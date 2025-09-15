import os
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import List, Sequence
import imagenet_maps
from pathlib import Path
from PIL import Image



class SingleFolderDataset(Dataset):
    def __init__(self, folder: str,
                 label: int = 0,
                 exts: Sequence[str] = (".jpg",".jpeg",".png",".bmp",".webp",".jpeg",".JPEG",".JPG",".PNG")):
        self.folder = Path(folder)
        self.files = sorted([p for p in self.folder.iterdir() if p.suffix in exts])
        if not self.files:
            raise RuntimeError(f"No images found in {folder}")
        self.transform = transforms.Compose([
                                            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.CenterCrop(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                        ])
        self.label = label

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.label



def get_dataloader(
    root: str,                  
    class_id: int,                  # e.g., "n02110958"
    imagenet_json_path: str,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    Build a DataLoader that yields ONLY images from the given WNID.
    Assumes a folder-per-class layout: <dir_with_classes>/<WNID>/*.JPEG
    """
    dir_with_class = root + f'/{imagenet_maps.i2w(class_id, json_path=imagenet_json_path)}'
    ds = SingleFolderDataset(dir_with_class, class_id)
    pin_memory = torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=num_workers > 0,
    )







def get_dataloader_multi(
    root: str,
    class_ids: Sequence[int],
    imagenet_json_path: str,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    Build a DataLoader over ONLY the given ImageNet class_ids.
    Assumes a folder-per-class layout: <root>/<WNID>/*.JPEG
    Labels in the loader are the original class_ids you pass in.
    """
    datasets = []
    missing = []

    for cid in class_ids:
        wnid = imagenet_maps.i2w(cid, json_path=imagenet_json_path)
        dir_with_class = os.path.join(root, wnid)
        if not os.path.isdir(dir_with_class):
            missing.append((cid, wnid, dir_with_class))
            continue
        # Your SingleFolderDataset takes (folder, label)
        datasets.append(SingleFolderDataset(dir_with_class, cid))

    if not datasets:
        raise RuntimeError(
            "No valid class folders found. Missing entries:\n" +
            "\n".join(f"  class_id={cid} wnid={wnid} path={path}" for cid, wnid, path in missing)
        )
    if missing:
        print("[get_dataloader_multi] Warning: skipped missing classes:\n" +
              "\n".join(f"  class_id={cid} wnid={wnid} path={path}" for cid, wnid, path in missing))

    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    pin_memory = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=num_workers > 0,
    )
