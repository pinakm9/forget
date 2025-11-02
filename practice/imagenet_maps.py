# imagenet_maps.py
from __future__ import annotations
import os, json, re
from functools import lru_cache
from typing import Dict, Iterable, List, Union, Optional

WNID = str
ClassId = int

class ImagenetMappingError(RuntimeError): ...

@lru_cache(maxsize=1)
def _load_maps(json_path: Optional[str] = None) -> Dict[str, Dict]:
    # Priority: explicit path → torchvision’s bundled JSON → local file in CWD
    paths = []
    if json_path: paths.append(json_path)

    try:
        from torchvision import models as _models  # optional
        paths.append(os.path.join(os.path.dirname(_models.__file__), "imagenet_class_index.json"))
    except Exception:
        pass

    paths.append(os.path.abspath("imagenet_class_index.json"))

    for p in paths:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                class_idx = json.load(f)
            try:
                idx0_to_wnid = {int(k): v[0] for k, v in class_idx.items()}
                idx0_to_text = {int(k): v[1] for k, v in class_idx.items()}
            except Exception as e:
                raise ImagenetMappingError("Bad JSON schema; expected {'0': ['wnid','label'], ...}") from e

            if len(idx0_to_wnid) != 1000 or min(idx0_to_wnid) != 0 or max(idx0_to_wnid) != 999:
                raise ImagenetMappingError("JSON must define 1000 classes keyed 0..999")

            wnid_to_idx0 = {w: i for i, w in idx0_to_wnid.items()}
            wnid_to_text = {w: idx0_to_text[i] for i, w in idx0_to_wnid.items()}
            return {
                "idx0_to_wnid": idx0_to_wnid,
                "wnid_to_idx0": wnid_to_idx0,
                "idx0_to_text": idx0_to_text,
                "wnid_to_text": wnid_to_text,
            }

    raise ImagenetMappingError(
        "Could not find imagenet_class_index.json. "
        "Provide json_path=..., place it in CWD, or install torchvision."
    )

def wnids_to_class_ids(
    wnids: Union[WNID, Iterable[WNID]], *, json_path: Optional[str] = None
) -> Union[ClassId, List[ClassId]]:
    m = _load_maps(json_path)
    w2i = m["wnid_to_idx0"]
    if isinstance(wnids, str):
        return w2i[wnids]
    return [w2i[w] for w in wnids]

def wnids_to_text(
    wnids: Union[WNID, Iterable[WNID]], *, json_path: Optional[str] = None
) -> Union[str, List[str]]:
    m = _load_maps(json_path)
    w2t = m["wnid_to_text"]
    if isinstance(wnids, str):
        return w2t[wnids]
    return [w2t[w] for w in wnids]

def class_ids_to_wnids(
    class_ids: Union[int, Iterable[int]], *, json_path: Optional[str] = None
) -> Union[str, List[str]]:
    m = _load_maps(json_path)
    i2w = m["idx0_to_wnid"]
    if isinstance(class_ids, int):
        return i2w[int(class_ids)]
    return [i2w[int(i)] for i in class_ids]


def _normalize_label(s: str) -> str:
    # lowercase, remove punctuation, collapse whitespace
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@lru_cache(maxsize=1)
def _text_index(json_path: Optional[str] = None):
    """
    Build a normalized text -> [wnid, ...] index from the loaded JSON labels.
    """
    m = _load_maps(json_path)  # uses your JSON-first loader
    idx: dict[str, List[str]] = {}
    for wnid, label in m["wnid_to_text"].items():
        key = _normalize_label(label)
        idx.setdefault(key, []).append(wnid)
    return idx

def text_to_wnid(
    text: str,
    *,
    json_path: Optional[str] = None,
    strict: bool = True,
    topk: int = 5,
) -> Union[str, List[str]]:
    """
    Map a human label to WNID using the JSON labels.

    - strict=True  -> require exact (normalized) match; returns a single WNID.
    - strict=False -> fuzzy token-overlap; returns up to `topk` candidate WNIDs.
    """
    index = _text_index(json_path)
    key = _normalize_label(text)

    # Exact (normalized) match
    if key in index:
        return index[key][0] if strict else index[key][:topk]

    if strict:
        raise KeyError(f"No exact label match for: {text!r}")

    # Fuzzy: simple token-overlap score against all labels
    tokens = set(key.split())
    scored: List[tuple[float, str]] = []
    for label_key, wnids in index.items():
        ltokens = set(label_key.split())
        # overlap over query length (bias toward covering the query)
        score = len(tokens & ltokens) / (len(tokens) or 1)
        if score > 0:
            for w in wnids:
                scored.append((score, w))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:topk]]

def text_to_class_id(
    text: str,
    *,
    json_path: Optional[str] = None,
    strict: bool = True,
    topk: int = 5,
) -> Union[int, List[int]]:
    """
    Convenience: label -> class_id(s). Uses text_to_wnid under the hood.
    """
    w = text_to_wnid(text, json_path=json_path, strict=strict, topk=topk)
    if isinstance(w, str):
        return wnids_to_class_ids(w, json_path=json_path)
    return wnids_to_class_ids(w, json_path=json_path)


w2i = wnids_to_class_ids   # wnid(s) -> class id(s)
w2t = wnids_to_text        # wnid(s) -> text label(s)
i2w = class_ids_to_wnids   # class id(s) -> wnid(s)
t2w = text_to_wnid         # text label(s) -> wnid(s)
t2i = text_to_class_id     # text label(s) -> class id(s)
