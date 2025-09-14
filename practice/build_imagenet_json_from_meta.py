# build_imagenet_json_from_meta.py
import json, os, argparse
import numpy as np
from scipy.io import loadmat

def _as_int(x):
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return int(x)
    if isinstance(x, np.ndarray): return int(x.item())
    if isinstance(x, (list, tuple)) and len(x) == 1: return _as_int(x[0])
    return int(x)

def _get(d, names):
    for n in names:
        if hasattr(d, n): return getattr(d, n)
        if hasattr(d, n.lower()): return getattr(d, n.lower())
        if hasattr(d, n.upper()): return getattr(d, n.upper())
    raise KeyError(f"Missing fields, tried: {names}")

def main(meta_path: str, out_path: str):
    meta_path = os.path.abspath(os.path.expanduser(meta_path))
    mat = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    syn = mat["synsets"]

    if not isinstance(syn, (list, tuple, np.ndarray)):
        syn = [syn]

    rows = []
    for s in syn:
        id1 = _as_int(_get(s, ["ILSVRC2012_ID", "ILSVRC2012ID", "id"]))
        wnid = str(_get(s, ["WNID"]))
        words_long = str(_get(s, ["words", "word"]))
        num_children = _as_int(_get(s, ["num_children", "numchildren", "children_count"]))
        if num_children == 0:  # leaf = the 1k classes
            rows.append((id1, wnid, words_long))

    if len(rows) < 1000:
        raise RuntimeError(f"Expected 1000 leaf synsets, found {len(rows)}")

    rows.sort(key=lambda r: r[0])  # by ILSVRC2012 1-based id
    # Use SHORT labels by default (compat with torchvision JSON).
    # If you want long labels, replace `label_short` with `words_long`.
    class_index = {
        str(i): [wnid, (rows[i][2].split(",")[0]).strip()]
        for i, (_, wnid, _) in enumerate(rows)
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(class_index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)
    print(f"Wrote {out_path} with {len(class_index)} classes.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Export torchvision-style JSON from ILSVRC2012 meta.mat")
    ap.add_argument("--meta", required=True, help="Path to ILSVRC2012_devkit_t12/data/meta.mat")
    ap.add_argument("--out",  required=True, help="Path to write imagenet_class_index.json")
    args = ap.parse_args()
    main(args.meta, args.out)
