"""
Join the per-run presentation lists into one trial-ordered list per subject.

Usage (run from anywhere):
    python build_trial_list.py /path/to/BOLD5000_Stimuli/Stimuli_Presentation_Lists CSI1

Writes CSI1_trial_names.txt (one image name per line, in the order shown).
Then set  NAMES_TXT = "CSI1_trial_names.txt"  in fit_encoding.py.
"""
import re
import sys
from pathlib import Path

EXPECTED = {"CSI1": 5254, "CSI2": 5254, "CSI3": 5254, "CSI4": 3108}


def natural_key(path):
    """Sort by the numbers in the path: session 2 before session 10, run 2 before run 10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(path.relative_to(path.parents[2])))]


def main(root, subject):
    subj_dir = Path(root) / subject
    all_txt = list(subj_dir.rglob("*.txt"))
    files = sorted((f for f in all_txt if re.fullmatch(r"[A-Za-z0-9]*_sess\d+_run\d+\.txt", f.name)), key=natural_key)
    skipped = [f for f in all_txt if f not in files]
    for f in skipped:
        print(f"Skipping non-run file: {f.relative_to(subj_dir)}")
    if not files:
        sys.exit(f"No run files matched under {subj_dir} ({len(all_txt)} .txt files found, all skipped). "
                 "Paste me a few of the filenames.")

    print(f"{len(files)} run files for {subject}. First three, in the order they will be read:")
    for f in files[:3]:
        print("  ", f.relative_to(subj_dir))

    names = []
    for f in files:
        names += [line.strip() for line in open(f) if line.strip()]

    out = f"trial_lists/{subject}_trial_names.txt"
    Path(out).write_text("\n".join(names) + "\n")

    print(f"\nTotal trials: {len(names)}  (expected {EXPECTED[subject]})")
    print("First 5 names:", names[:5])
    print("Last 3 names: ", names[-3:])
    print(f"Wrote {out}")
    if len(names) != EXPECTED[subject]:
        print("\nWARNING: count doesn't match. Paste this output to me; don't fit yet.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
