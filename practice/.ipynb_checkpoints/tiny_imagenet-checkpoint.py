from pathlib import Path
from typing import List, Union, Iterable, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch

class TinyImageNetSimple(Dataset):
    """
    Minimal Tiny-ImageNet-200 Dataset.
      - Labels are 0..199 by the order in wnids.txt
      - train:  train/<WNID>/images/*.JPEG
      - val:    val/images/*.JPEG  (labels via val/val_annotations.txt)
      - test:   test/images/*.JPEG (returns target = -1)
    """
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        assert split in {"train", "val", "test"}
        self.split = split
        self.transform = transform or self._default_tf(split)

        # Build class mapping from wnids.txt (defines the 0..199 label order)
        with open(self.root / "wnids.txt", "r", encoding="utf-8") as f:
            self.wnids = [ln.strip() for ln in f if ln.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        # Gather samples
        self.samples = self._gather_samples()

    def _default_tf(self, split):
        if split == "train":
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def _gather_samples(self):
        items = []
        if self.split == "train":
            for wnid in self.wnids:
                img_dir = self.root / "train" / wnid / "images"
                for p in sorted(img_dir.glob("*.JPEG")):
                    items.append((p, self.class_to_idx[wnid]))
        elif self.split == "val":
            # filename -> wnid from val_annotations.txt
            val_map = {}
            with open(self.root / "val" / "val_annotations.txt", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split("\t")
                    if len(parts) == 1:
                        parts = line.split()  # tolerate space-separated
                    fname, wnid = parts[0], parts[1]
                    val_map[fname] = wnid
            img_dir = self.root / "val" / "images"
            for p in sorted(img_dir.glob("*.JPEG")):
                wnid = val_map[p.name]
                items.append((p, self.class_to_idx[wnid]))
        else:  # test (unlabeled)
            img_dir = self.root / "test" / "images"
            for p in sorted(img_dir.glob("*.JPEG")):
                items.append((p, -1))
        return items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target





class TinyImageNetOneClass(Dataset):
    """
    Minimal class-specific dataset for Tiny-ImageNet-200.

    Args:
      root: path to tiny-imagenet-200
      split: "train" or "val"
      class_idx: integer in [0,199] (position in wnids.txt)  OR
      wnid: string like "n02099601" (golden retriever)
      remap_label_to_zero: if True, returned target = 0; else the original tiny label (0..199)
      transform: torchvision transforms; small defaults provided

    Returns: (image_tensor, target)
    """
    def __init__(self, root, split="train", class_idx=None, wnid=None,
                 remap_label_to_zero=True, transform=None):
        self.root = Path(root)
        assert split in {"train", "val"}, "Use 'train' or 'val'. (Test has no labels.)"
        self.split = split
        self.remap = remap_label_to_zero

        # Tiny-ImageNet class order
        with open(self.root / "wnids.txt", "r", encoding="utf-8") as f:
            self.wnids = [ln.strip() for ln in f if ln.strip()]

        # Resolve target class
        if wnid is None:
            assert class_idx is not None, "Provide class_idx or wnid"
            self.wnid = self.wnids[class_idx]
            self.tiny_label = int(class_idx)
        else:
            assert wnid in self.wnids, f"WNID {wnid} not found in wnids.txt"
            self.wnid = wnid
            self.tiny_label = self.wnids.index(wnid)

        # Gather file list
        if self.split == "train":
            img_dir = self.root / "train" / self.wnid / "images"
            self.files = sorted(img_dir.glob("*.JPEG"))
        else:  # val
            ann = {}
            with open(self.root / "val" / "val_annotations.txt", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split("\t")
                    if len(parts) == 1:
                        parts = line.split()  # tolerate space-separated
                    ann[parts[0]] = parts[1]
            img_dir = self.root / "val" / "images"
            self.files = sorted([img_dir / fn for fn, w in ann.items() if w == self.wnid])

        # Tiny defaults (kept super simple)
        if transform is None:
            if split == "train":
                self.transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        target = 0 if self.remap else self.tiny_label
        return img, target




class TinyImageNetSubset(Dataset):
    """
    Tiny-ImageNet-200 subset dataset.

    Args:
      root: path to tiny-imagenet-200
      split: "train" or "val"
      classes: list of Tiny indices (ints 0..199) and/or WNIDs (str like "n02099601")
      remap_labels: if True, labels are 0..K-1 in the order of 'classes' (after resolving to WNIDs);
                    if False, labels are the original Tiny labels (0..199) from wnids.txt
      transform: torchvision transforms (simple defaults provided)

    Returns: (image_tensor, label)
    """
    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        classes: Iterable[Union[int, str]],
        remap_labels: bool = True,
        transform=None,
    ):
        self.root = Path(root)
        assert split in {"train", "val"}, "Use 'train' or 'val' (test has no labels)."
        self.split = split
        self.transform = transform or self._default_tf(split)

        # Load Tiny-ImageNet class order
        wnids_path = self.root / "wnids.txt"
        assert wnids_path.exists(), f"Missing {wnids_path}"
        with wnids_path.open("r", encoding="utf-8") as f:
            self.wnids = [ln.strip() for ln in f if ln.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}  # wnid -> tiny label

        # Resolve requested classes -> WNIDs (validate as we go)
        requested_wnids: List[str] = []
        for c in classes:
            if isinstance(c, int):
                assert 0 <= c < len(self.wnids), f"Index {c} out of range 0..{len(self.wnids)-1}"
                requested_wnids.append(self.wnids[c])
            elif isinstance(c, str):
                assert c in self.wnids, f"WNID {c} not found in wnids.txt"
                requested_wnids.append(c)
            else:
                raise TypeError(f"Unsupported class identifier type: {type(c)}")

        # De-dup while preserving order
        seen = set()
        self.target_wnids = [w for w in requested_wnids if (w not in seen and not seen.add(w))]

        # Label mapping
        if remap_labels:
            # map each selected WNID to 0..K-1 by the order given in `classes`
            self.label_map = {w: i for i, w in enumerate(self.target_wnids)}
        else:
            # keep original Tiny labels (0..199)
            self.label_map = {w: self.class_to_idx[w] for w in self.target_wnids}

        # Build file list
        self.samples: List[Tuple[Path, int]] = self._gather_samples()

    def _default_tf(self, split):
        if split == "train":
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def _gather_samples(self) -> List[Tuple[Path, int]]:
        items: List[Tuple[Path, int]] = []
        if self.split == "train":
            # train/<wnid>/images/*.JPEG
            for wnid in self.target_wnids:
                img_dir = self.root / "train" / wnid / "images"
                for p in sorted(img_dir.glob("*.JPEG")):
                    items.append((p, self.label_map[wnid]))
        else:  # val
            # val/val_annotations.txt: filename \t wnid ...
            ann_path = self.root / "val" / "val_annotations.txt"
            with ann_path.open("r", encoding="utf-8") as f:
                fname2wnid = {}
                for line in f:
                    parts = line.split("\t")
                    if len(parts) == 1:
                        parts = line.split()  # tolerate space-separated
                    if not parts:
                        continue
                    fname, wnid = parts[0], parts[1]
                    fname2wnid[fname] = wnid
            img_dir = self.root / "val" / "images"
            for p in sorted(img_dir.glob("*.JPEG")):
                wnid = fname2wnid.get(p.name)
                if wnid in self.label_map:
                    items.append((p, self.label_map[wnid]))
        return items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label





class TinyImageNetSimpleLoader():
    def __init__(self, root, split="train", transform=None):
        self.dataset = TinyImageNetSimple(root, split, transform)
    
    def get_dataloader(self, batch_size, num_workers=4, pin_mem=torch.cuda.is_available()):
         return DataLoader(self.dataset,
                        batch_size=batch_size,
                        shuffle=True,                 # shuffle for training
                        num_workers=num_workers,
                        pin_memory=pin_mem,
                        drop_last=True,               # optional: keep batch sizes uniform
                        persistent_workers=num_workers > 0,
                    )


class TinyImageNetOneClassLoader():
    def __init__(self, root, split="train", class_idx=None, wnid=None,
                 remap_label_to_zero=True, transform=None):
        dataset = TinyImageNetOneClass(root, split, class_idx, wnid, remap_label_to_zero, transform)

    def get_dataloader(self, batch_size, num_workers=4, pin_mem=torch.cuda.is_available()):
         return DataLoader(self.dataset,
                        batch_size=batch_size,
                        shuffle=True,                 # shuffle for training
                        num_workers=num_workers,
                        pin_memory=pin_mem,
                        drop_last=True,               # optional: keep batch sizes uniform
                        persistent_workers=num_workers > 0,
                    )


class TinyImageNetSubsetLoader():
    def __init__(self, root: Union[str, Path],
                split: str,
                classes: Iterable[Union[int, str]],
                remap_labels: bool = True,
                transform=None):
        self.dataset = TinyImageNetSubset(root, split, classes, remap_labels, transform)

    def get_dataloader(self, batch_size, num_workers=4, pin_mem=torch.cuda.is_available()):
         return DataLoader(self.dataset,
                        batch_size=batch_size,
                        shuffle=True,                 # shuffle for training
                        num_workers=num_workers,
                        pin_memory=pin_mem,
                        drop_last=True,               # optional: keep batch sizes uniform
                        persistent_workers=num_workers > 0,
                    )



# Show a grid of samples from a DataLoader (e.g., train_loader)

import math, os
import numpy as np
import matplotlib.pyplot as plt


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def show_samples(loader, n=16, cols=8, show_names=False, root=None):
    """
    loader: a PyTorch DataLoader (e.g., for TinyImageNetSubset)
    n:      how many images to show
    cols:   grid columns
    show_names: if True, read words.txt (needs root) to show human-readable names
    root:   path to tiny-imagenet-200 (only needed if show_names=True)
    """
    ds = loader.dataset

    # Build label -> WNID mapping:
    # - If your dataset remaps labels, it should have .label_map (wnid -> new_label)
    # - Otherwise fall back to .class_to_idx (wnid -> tiny_label 0..199)
    if hasattr(ds, "label_map"):
        idx_to_wnid = {v: k for k, v in ds.label_map.items()}
    elif hasattr(ds, "class_to_idx"):
        idx_to_wnid = {v: k for k, v in ds.class_to_idx.items()}
    else:
        idx_to_wnid = None  # titles will just show numeric labels

    # Optional: WNID -> human-readable name
    words = {}
    if show_names:
        assert root is not None, "Pass root='.../tiny-imagenet-200' when show_names=True"
        words_path = os.path.join(root, "words.txt")
        try:
            with open(words_path, "r", encoding="utf-8") as f:
                for ln in f:
                    if "\t" in ln:
                        wid, name = ln.strip().split("\t", 1)
                    else:
                        wid, name = ln.strip().split(" ", 1)
                    words[wid] = name
        except FileNotFoundError:
            pass  # fall back to WNIDs

    # Collect up to n images (across batches)
    imgs, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("DataLoader must yield (images, labels) tuples.")
        x = x.detach().cpu()
        y = y.detach().cpu()
        take = min(n - len(imgs), x.size(0))
        imgs.append(x[:take])
        labels.append(y[:take])
        if sum(t.size(0) for t in imgs) >= n:
            break

    if not imgs:
        print("No samples to show.")
        return

    imgs = torch.cat(imgs, dim=0)[:n]      # [n,3,H,W]
    labels = torch.cat(labels, dim=0)[:n]  # [n]

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        if i >= len(imgs):
            axes[i].axis("off")
            continue
        im = imgs[i].permute(1, 2, 0).numpy()
        im = (im * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)  # unnormalize for display

        label = int(labels[i])
        if idx_to_wnid is not None:
            wnid = idx_to_wnid.get(label, str(label))
            title = wnid
            if show_names:
                title = f"{words.get(wnid, wnid).split(',')[0]}\n{wnid} ({label})"
        else:
            title = str(label)

        axes[i].imshow(im)
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
