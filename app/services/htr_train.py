"""Training script for the CRNN HTR model (CTC loss).

Dataset: a folder with `labels.tsv` lines of `<relative_image_path>\t<text>` where each
image is a single text line (e.g. IAM lines, or your own block-print samples).

    python -m app.services.htr_train --data data/lines --epochs 30 --out weights/crnn_htr.pt
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .htr import BLANK, CHARS, CRNN, IMG_H


class LineDataset(Dataset):
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "labels.tsv"), encoding="utf-8") as f:
            self.items = [l.rstrip("\n").split("\t", 1) for l in f if "\t" in l]
        self.items = [(p, "".join(c for c in t if c in CHARS)) for p, t in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        img = cv2.resize(img, (max(32, int(w * IMG_H / h)), IMG_H)).astype(np.float32)
        img = (255 - img) / 255.0
        target = [CHARS.index(c) + 1 for c in text]
        return torch.from_numpy(img)[None], torch.tensor(target, dtype=torch.long)


def collate(batch):
    imgs, tgts = zip(*batch)
    maxw = max(i.shape[-1] for i in imgs)
    x = torch.zeros(len(imgs), 1, IMG_H, maxw)
    for k, im in enumerate(imgs):
        x[k, :, :, : im.shape[-1]] = im
    lens = torch.tensor([len(t) for t in tgts])
    return x, torch.cat(tgts), lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="weights/crnn_htr.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    dl = DataLoader(LineDataset(a.data), batch_size=a.bs, shuffle=True, collate_fn=collate)

    for ep in range(a.epochs):
        total = 0.0
        for x, y, ylen in dl:
            x = x.to(dev)
            logp = model(x).permute(1, 0, 2)  # T,B,K
            xlen = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp, y, xlen, ylen)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        print(f"epoch {ep + 1}/{a.epochs} loss {total / max(1, len(dl)):.4f}")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save(model.state_dict(), a.out)


if __name__ == "__main__":
    main()
