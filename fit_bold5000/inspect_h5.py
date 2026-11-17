"""
Step 1: look inside an .h5 file.

Usage:
    python inspect_h5.py /cs-data-01/drd92/doi_10_5061_dryad_wpzgmsbtr__v20240621/CSI1_BOLD5000_ROIs.h5

An .h5 (HDF5) file is like a folder of named arrays. "Groups" are folders,
"datasets" are arrays. This prints everything in the file so you can see
the names (keys) and shapes, and previews the first few values of small
datasets (e.g. lists of image names).
"""
import sys

import h5py
import numpy as np


def describe(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}   shape={obj.shape}  dtype={obj.dtype}")
        # Preview short 1-D arrays (often names / labels)
        if obj.ndim == 1 and 0 < obj.shape[0] <= 100000:
            head = obj[:3]
            if obj.dtype.kind in ("S", "O"):
                head = [h.decode() if isinstance(h, bytes) else h for h in head]
            print(f"      first values: {list(head)}")
    else:
        print(f"{name}/  (group)")


if __name__ == "__main__":
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        print(f"Contents of {path}:\n")
        f.visititems(describe)
        if len(f.attrs):
            print("\nFile attributes:")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
    print("\nNext: note which dataset holds the responses (2-D, trials x voxels)")
    print("and whether the image names/order are stored in the file.")
