"""
Vision-capable GPT-NeoX (LLaVA-style): vision encoder -> MLP projector -> image
tokens prepended to the text embeddings.

Fixes vs. the original draft:
  * torch.cat (not stack) along the sequence dim; image tokens are added to
    `inputs_embeds` BEFORE mask / position ids / rotary embeddings are built
  * attention mask is extended for image tokens (works in prefill and decode)
  * labels are left-padded with -100 for image positions
  * image is only encoded/prepended on prefill (empty KV cache)
  * vision settings live in the config, so save_pretrained / from_pretrained work
  * base_model_prefix / tied-weight key match the renamed submodule
"""

import torch
from torch import nn

from transformers import AutoConfig, AutoModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.image_processing_base import BatchFeature
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXLayer,
    GPTNeoXPreTrainedModel,
    GPTNeoXRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class VisualGPTNeoXConfig(GPTNeoXConfig):
    """GPTNeoXConfig + a nested vision config and projector size.

    NOTE: if your transformers version uses dataclass-style configs and this
    custom __init__ complains, move the two extra fields into that style.
    """

    model_type = "visual_gpt_neox"
    sub_configs = {"vision_config": AutoConfig}

    def __init__(self, vision_config=None, projector_hidden_size=None, use_projector=True, **kwargs):
        super().__init__(**kwargs)
        if isinstance(vision_config, dict):
            # dict must contain "model_type" (to_dict() output does)
            vision_config = AutoConfig.for_model(**vision_config)
        self.vision_config = vision_config
        # None -> same as LM hidden size
        self.projector_hidden_size = projector_hidden_size
        # False -> no MLP: vision tokens feed the LM directly (vision hidden size must equal LM hidden size)
        self.use_projector = use_projector

    @classmethod
    def from_llm_and_vision(cls, llm_config: GPTNeoXConfig, vision_config, **kwargs):
        d = llm_config.to_dict()
        for k in ("model_type", "transformers_version"):
            d.pop(k, None)
        d.update(kwargs)
        return cls(vision_config=vision_config, **d)


class VisualGPTNeoXPreTrainedModel(GPTNeoXPreTrainedModel):
    config_class = VisualGPTNeoXConfig
    base_model_prefix = "visual_gpt_neox"


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
class VisualGPTNeoXModel(VisualGPTNeoXPreTrainedModel):
    def __init__(self, config: VisualGPTNeoXConfig):
        super().__init__(config)
        if config.vision_config is None:
            raise ValueError("VisualGPTNeoXConfig.vision_config must be set")

        # Vision tower + projector (built from config; random init until you load weights)
        self.vision_encoder = AutoModel.from_config(config.vision_config)
        vis_dim = config.vision_config.hidden_size
        if config.use_projector:
            proj_dim = config.projector_hidden_size or config.hidden_size
            self.mlp = nn.Sequential(
                nn.Linear(vis_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, config.hidden_size),
            )
        else:
            if vis_dim != config.hidden_size:
                raise ValueError(
                    f"use_projector=False needs vision hidden size ({vis_dim}) == LM hidden size ({config.hidden_size})"
                )
            self.mlp = nn.Identity()

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = GPTNeoXRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def encode_images(self, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(B, C, H, W) -> (B, N_img, hidden_size)"""
        enc_dtype = next(self.vision_encoder.parameters()).dtype
        out = self.vision_encoder(pixel_values.to(enc_dtype))
        feats = getattr(out, "last_hidden_state", out)
        if feats.ndim != 3:
            raise ValueError(
                f"Vision encoder must return patch tokens (B, N, D); got shape {tuple(feats.shape)}"
            )
        # LLaVA-style tweak you may want: drop the CLS token -> feats = feats[:, 1:]
        return self.mlp(feats).to(dtype)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        vision_inputs: BatchFeature | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        vision_inputs (`BatchFeature`, *optional*):
            Output of the image processor; must contain `pixel_values`. Image tokens are prepended
            to the text only on prefill (empty KV cache); ignored on later decoding steps.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0

        # 1) Prepend image tokens (prefill only) BEFORE building mask/positions/rotary.
        if vision_inputs is not None and past_seen == 0:
            vis = self.encode_images(vision_inputs["pixel_values"], inputs_embeds.dtype)
            inputs_embeds = torch.cat([vis, inputs_embeds], dim=1)
            position_ids = None  # recompute for the combined length

        # 2) Make the 2D attention mask cover the image tokens that live in the sequence/cache.
        #    Handles prefill (text mask + image tokens) and decode steps (generate() only
        #    extends the text-length mask) with the same rule.
        if attention_mask is not None and attention_mask.ndim == 2:
            expected = past_seen + inputs_embeds.shape[1]
            missing = expected - attention_mask.shape[1]
            if missing > 0:
                img_mask = attention_mask.new_ones(attention_mask.shape[0], missing)
                attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = self.emb_dropout(inputs_embeds)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                layer_past=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.final_layer_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value


# --------------------------------------------------------------------------- #
# Causal LM
# --------------------------------------------------------------------------- #
class VisualGPTNeoXForCausalLM(VisualGPTNeoXPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "visual_gpt_neox.embed_in.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: VisualGPTNeoXConfig):
        super().__init__(config)
        self.visual_gpt_neox = VisualGPTNeoXModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ---- weight loading / freezing helpers --------------------------------- #
    @torch.no_grad()
    def load_language_model(self, lm: nn.Module):
        """Copy weights from a pretrained GPTNeoXForCausalLM (text transformer + head)."""
        missing, unexpected = self.visual_gpt_neox.load_state_dict(lm.base_model.state_dict(), strict=False)
        # Only the vision tower + projector should be missing
        bad = [k for k in missing if not k.startswith(("vision_encoder.", "mlp."))]
        if bad or unexpected:
            raise RuntimeError(f"LM weight mismatch. missing={bad} unexpected={unexpected}")
        self.lm_head.load_state_dict(lm.get_output_embeddings().state_dict())

    @torch.no_grad()
    def load_vision_encoder(self, vision_model: nn.Module):
        """Copy pretrained vision weights (e.g. CLIPVisionModel). Call AFTER construction,
        since post_init() randomly initializes everything."""
        self.visual_gpt_neox.vision_encoder.load_state_dict(vision_model.state_dict())

    def freeze_for_stage1(self):
        """Stage 1: train only the projector."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.visual_gpt_neox.mlp.parameters():
            p.requires_grad = True

    def unfreeze_lm(self):
        """Stage 2: also train the language model (vision encoder stays frozen)."""
        for name, p in self.named_parameters():
            if "vision_encoder" not in name:
                p.requires_grad = True

    # ---- generation -------------------------------------------------------- #
    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        cache = model_inputs.get("past_key_values")
        if cache is not None and cache.get_seq_length() > 0:
            model_inputs["vision_inputs"] = None  # image already lives in the KV cache
        return model_inputs

    # ---- forward ----------------------------------------------------------- #
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        vision_inputs: BatchFeature | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Text labels (use -100 to ignore). Image positions are padded with -100 automatically.
        vision_inputs (`BatchFeature`, *optional*):
            Output of the image processor; must contain `pixel_values`.
        """
        outputs: BaseModelOutputWithPast = self.visual_gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            vision_inputs=vision_inputs,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if labels is not None:
            # Hidden states include the image tokens; left-pad labels to match.
            n_extra = hidden_states.shape[1] - labels.shape[1]
            if n_extra > 0:
                pad = labels.new_full((labels.shape[0], n_extra), -100)
                labels = torch.cat([pad, labels], dim=1)
            labels = labels[:, slice_indices]

        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPVisionModel, GPTNeoXForCausalLM

    LM = "EleutherAI/pythia-160m"
    VIS = "openai/clip-vit-base-patch32"

    tok = AutoTokenizer.from_pretrained(LM)
    proc = AutoImageProcessor.from_pretrained(VIS)
    vision = CLIPVisionModel.from_pretrained(VIS)
    lm = GPTNeoXForCausalLM.from_pretrained(LM)

    cfg = VisualGPTNeoXConfig.from_llm_and_vision(lm.config, vision.config)
    model = VisualGPTNeoXForCausalLM(cfg)
    model.load_language_model(lm)       # after construction (post_init re-inits weights)
    model.load_vision_encoder(vision)
    model.freeze_for_stage1()

    img = Image.new("RGB", (224, 224))
    vision_inputs = proc(images=[img], return_tensors="pt")
    text = tok(["A photo of"], return_tensors="pt")

    # training step (labels = text ids; image positions are masked automatically)
    out = model(**text, labels=text["input_ids"], vision_inputs=vision_inputs)
    print("loss:", out.loss.item(), "logits:", tuple(out.logits.shape))

    # generation
    gen = model.generate(**text, vision_inputs=vision_inputs, max_new_tokens=20)
    print(tok.decode(gen[0]))

    # save / reload round trip
    model.save_pretrained("visual-neox-ckpt")
    model = VisualGPTNeoXForCausalLM.from_pretrained("visual-neox-ckpt")
