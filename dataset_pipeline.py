import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

VKITTI_RGB_ROOT = "/media/rvcse22/CSERV/vdaproj/dataset/vkitti_2.0.3_rgb-001"
VKITTI_DEP_ROOT = "/media/rvcse22/CSERV/vdaproj/dataset/vkitti_2.0.3_depth-002"

IMG_H = 392
IMG_W = 518
SEQ_LEN = 4
MAX_DEPTH = 80.0


def _sorted_frames(directory, exts=("*.png", "*.jpg")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths.sort(key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)])
    return paths


class VKittiVideoDataset(Dataset):

    def __init__(self, seq_len=SEQ_LEN, stride=None,
                 img_h=IMG_H, img_w=IMG_W):

        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.clips = []

        rgb_camera_dirs = glob.glob(
            os.path.join(
                VKITTI_RGB_ROOT,
                "*",
                "*",
                "frames",
                "rgb",
                "Camera_*"
            )
        )

        for rgb_dir in sorted(rgb_camera_dirs):

            rel = os.path.relpath(rgb_dir, VKITTI_RGB_ROOT)
            parts = rel.split(os.path.sep)

            if len(parts) != 5:
                continue

            scene_name = parts[0]
            condition = parts[1]
            camera_name = parts[4]

            dep_dir = os.path.join(
                VKITTI_DEP_ROOT,
                scene_name,
                condition,
                "frames",
                "depth",
                camera_name
            )

            if not os.path.isdir(dep_dir):
                continue

            rgb_files = _sorted_frames(rgb_dir)
            dep_files = _sorted_frames(dep_dir)

            n = min(len(rgb_files), len(dep_files))

            if n < seq_len:
                continue

            rgb_files = rgb_files[:n]
            dep_files = dep_files[:n]

            for start in range(0, n - seq_len + 1, self.stride):

                rgb_clip = rgb_files[start:start + seq_len]
                dep_clip = dep_files[start:start + seq_len]

                self.clips.append((rgb_clip, dep_clip))

        print(f"[Dataset] {len(self.clips)} clips  |  seq_len={seq_len}  stride={self.stride}")

    def __len__(self):
        return len(self.clips)

    def _load_depth(self, path):

        dep = np.array(Image.open(path), dtype=np.float32) / 100.0
        dep = np.clip(dep, 0.0, MAX_DEPTH)

        dep = torch.from_numpy(dep).unsqueeze(0)

        dep = torch.nn.functional.interpolate(
            dep.unsqueeze(0),
            size=(IMG_H, IMG_W),
            mode="nearest"
        ).squeeze(0)

        return dep

    def __getitem__(self, idx):

        rgb_paths, dep_paths = self.clips[idx]

        rgbs = []
        deps = []

        for rp, dp in zip(rgb_paths, dep_paths):

            rgb = Image.open(rp).convert("RGB")
            rgb = self.rgb_transform(rgb)

            dep = self._load_depth(dp)

            rgbs.append(rgb)
            deps.append(dep)

        rgbs = torch.stack(rgbs)
        deps = torch.stack(deps)

        return rgbs, deps


def build_loaders(batch_size=2,
                  seq_len=SEQ_LEN,
                  num_workers=8,
                  val_split=0.1):

    full_ds = VKittiVideoDataset(seq_len=seq_len)

    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kw
    )

    return train_loader, val_loader


if __name__ == "__main__":

    tl, vl = build_loaders(batch_size=2)

    rgb, dep = next(iter(tl))

    print(f"rgb shape   : {rgb.shape}")
    print(f"depth shape : {dep.shape}")
    print(f"depth range : [{dep.min():.2f}, {dep.max():.2f}]")