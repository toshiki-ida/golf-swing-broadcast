"""train_clubhead — datasets/clubhead のラベルからクラブヘッド検出 (ヒートマップ回帰) を学習し、ONNX に書き出す。

    <torch 入り venv>/python.exe tools/train_clubhead.py [--dataset <保存先>] [--epochs 80] [--val-clips 1] [--arch tiny] [--out models/clubhead.onnx]

    --dataset の既定はリポジトリ直下の datasets/clubhead。現場 PC から持ち帰ったラベルは
    <プロジェクトフォルダ>/datasets/clubhead を指定する (複数ある場合は 1 つのフォルダにまとめてから)

- 学習データ: datasets/clubhead/*/labels.json の strong 点 (manual / auto_confirmed)。weak (interpolated) は既定で使わない (--use-weak で含める)
- 負例: negative (自動生成されたあと人が削除した点)。その座標の周り (半径 NEG_RADIUS セル) だけヒートマップを 0 に寄せる
  (切り出しの他の場所は本当のヘッドかもしれないので損失を掛けない)。--no-negatives で外す
- 分割: クリップ単位 (同じクリップのフレームは train / val のどちらか一方だけ)。--val-clips 本を検証に回す
- モデル: 自前の小さな CNN (約 0.3M パラメータ、入力 256、出力ヒートマップ 64x64)。
          ライセンスはこのリポジトリと同じ (外部モデルの重みを使わない)。データが増えたら YOLOX-Nano (Apache-2.0) に
          差し替えられるよう、入出力 (切り出し → ピーク) は ai/club_detector.py に閉じている
- 出力: models/clubhead.onnx と models/clubhead_train.json (データ数、分割、検証精度、学習時間)

検証精度: 検証クリップの各点で、ヒートマップ最大位置と正解の距離 (元解像度 px)。平均 / 中央値 / 15 px 以内の割合 (PCK@15)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.dataset import DATASET_DIR  # noqa: E402
from ai.club_detector import INPUT, STRIDE, ClubDetector  # noqa: E402

HM = INPUT // STRIDE
NEG_RADIUS = 4   # 負例の損失を掛ける半径 (ヒートマップのセル数)


# ---------------------------------------------------------------- データ

def load_records(use_weak: bool, root=DATASET_DIR, use_negatives: bool = True):
    """labels.json を読む。holdout として excluded の付いたクリップは外す。同一フレームに座標の違う強ラベルが複数ある
    場合は矛盾なので両方外し、同じ座標なら 1 つにまとめる (件数は stderr に出す)。各レコードに camera を付ける。"""
    recs = []
    n_dup = n_conflict = 0
    for lab in sorted(Path(root).glob("*/labels.json")):
        meta = json.loads(lab.read_text(encoding="utf-8"))
        if meta.get("excluded"):
            print(f"[skip] {meta['clip_id']}: excluded={meta['excluded']}")
            continue
        cam = str(meta.get("camera") or "unknown")
        seen = {}
        for r in meta["points"]:
            if (r.get("weak") and not use_weak) or not r.get("in_crop", True):
                continue
            if r.get("negative") and not use_negatives:
                continue
            rec = {"clip": meta["clip_id"], "camera": cam, "img": lab.parent / r["image"], "cx": r["cx"], "cy": r["cy"],
                   "crop_size": r.get("crop_size", 320), "crop": r["crop"], "weak": bool(r.get("weak")),
                   "negative": bool(r.get("negative")), "frame": r.get("frame")}
            if not rec["weak"] and not rec["negative"]:
                key = rec["frame"]
                if key in seen:
                    other = seen[key]
                    if abs(other["cx"] - rec["cx"]) <= 2 and abs(other["cy"] - rec["cy"]) <= 2:
                        n_dup += 1
                        continue
                    other["conflict"] = True
                    rec["conflict"] = True
                    n_conflict += 1
                seen[key] = rec
            recs.append(rec)
    out = [r for r in recs if not r.get("conflict")]
    if n_dup or n_conflict:
        print(f"[data] 同一フレームの重複 {n_dup} 点をまとめ、座標が矛盾する {n_conflict} 組 ({len(recs) - len(out)} 点) を除外")
    return out


def split_by_clip(recs, val_clips: int, seed: int = 0, val_ratio: float = 0.0):
    """クリップ単位の分割。val_ratio > 0 なら撮影方向 (camera) ごとに比率で検証へ回す (各方向から最低 1 本、
    方向が 1 本しかなければ学習側)。val_ratio == 0 なら従来どおり val_clips 本。"""
    rng = random.Random(seed)
    if val_ratio > 0:
        by_cam = {}
        for c in sorted({(r["clip"], r["camera"]) for r in recs}):
            by_cam.setdefault(c[1], []).append(c[0])
        val = set()
        for cam, clips in sorted(by_cam.items()):
            rng.shuffle(clips)
            if len(clips) < 2:
                continue
            k = max(1, int(round(len(clips) * val_ratio)))
            val.update(clips[:k])
    else:
        clips = sorted({r["clip"] for r in recs})
        rng.shuffle(clips)
        val = set(clips[:val_clips]) if len(clips) > val_clips else set(clips[:1]) if len(clips) > 1 else set()
    train = [r for r in recs if r["clip"] not in val]
    valid = [r for r in recs if r["clip"] in val]
    return train, valid, sorted(val)


def make_sample(rec, augment: bool):
    img = cv2.imread(str(rec["img"]))
    s = INPUT / rec["crop_size"]
    img = cv2.resize(img, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    cx, cy = rec["cx"] * s, rec["cy"] * s
    if augment:
        # 平行移動 (±12%)、明るさ・コントラスト、左右反転
        dx, dy = np.random.uniform(-0.12, 0.12, 2) * INPUT
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (INPUT, INPUT), borderMode=cv2.BORDER_REFLECT)
        cx, cy = cx + dx, cy + dy
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            cx = INPUT - 1 - cx
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    hm = np.zeros((HM, HM), np.float32)
    mask = np.ones((HM, HM), np.float32)
    # 中心は整数画素に丸める (ピークが正確に 1 になり focal loss の正例として数えられる。CenterNet と同じ)
    gx, gy = int(round(cx / STRIDE)), int(round(cy / STRIDE))
    if rec.get("negative"):
        # 負例: 目標は 0、損失はその座標の周りだけ
        mask[:] = 0.0
        if 0 <= gx < HM and 0 <= gy < HM:
            yy, xx = np.mgrid[0:HM, 0:HM]
            mask = (((xx - gx) ** 2 + (yy - gy) ** 2) <= NEG_RADIUS ** 2).astype(np.float32)
    elif 0 <= gx < HM and 0 <= gy < HM:
        yy, xx = np.mgrid[0:HM, 0:HM]
        hm = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
        hm[gy, gx] = 1.0
    return x, hm[None], (cx, cy), mask[None]


# ---------------------------------------------------------------- モデル

def build_model(arch: str):
    import torch.nn as nn

    def block(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    c = {"tiny": (16, 32, 64, 64), "small": (24, 48, 96, 96)}[arch]
    return nn.Sequential(
        block(3, c[0]), block(c[0], c[0], 2),          # 256 -> 128
        block(c[0], c[1]), block(c[1], c[1], 2),       # 128 -> 64
        block(c[1], c[2]), block(c[2], c[2], 2),       # 64 -> 32
        block(c[2], c[3]), block(c[3], c[3]),
        nn.Upsample(scale_factor=2, mode="nearest"),   # 32 -> 64
        block(c[3], c[1]), block(c[1], c[0]),
        nn.Conv2d(c[0], 1, 1), nn.Sigmoid(),
    )


def init_head_bias(model, prior: float = 0.01):
    """最終層のバイアスを事前確率 prior に合わせる (CenterNet/RetinaNet と同じ)。背景 4096 画素に対し正例 1 画素なので、
    これが無いと学習初期は全画素 0.5 から始まり、少ないステップ数では背景を抑えきれない"""
    import math
    import torch.nn as nn
    last = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    nn.init.constant_(last.bias, -math.log((1 - prior) / prior))


def focal_loss(pred, gt, mask=None):
    import torch
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    if mask is None:
        mask = torch.ones_like(gt)
    pos_loss = (torch.log(pred) * (1 - pred) ** 2 * pos * mask).sum()
    neg_loss = (torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg * mask).sum()
    npos = pos.sum().clamp(min=1)
    return -(pos_loss + neg_loss) / npos


# ---------------------------------------------------------------- 評価

def evaluate(model, recs, device):
    import torch
    errs = []
    model.eval()
    with torch.no_grad():
        for r in recs:
            if r.get("negative"):
                continue
            x, _, (cx, cy), _ = make_sample(r, augment=False)
            hm = model(torch.from_numpy(x[None]).to(device))[0, 0].cpu().numpy()
            pk = ClubDetector.peaks(hm, 0, 0, INPUT)
            if not pk:
                errs.append(float("nan"))
                continue
            px, py, _ = pk[0]
            # 元解像度 px に戻す (切り出し一辺 / INPUT)
            scale = r["crop"][2] / INPUT
            errs.append(float(np.hypot(px - cx, py - cy)) * scale)
    e = np.array(errs)
    ok = e[~np.isnan(e)]
    return {"n": int(len(e)), "detected": int(len(ok)), "mean_px": round(float(ok.mean()), 1) if len(ok) else None,
            "median_px": round(float(np.median(ok)), 1) if len(ok) else None,
            "pck15": round(float((ok <= 15).mean()), 3) if len(ok) else None,
            "pck30": round(float((ok <= 30).mean()), 3) if len(ok) else None}


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR), help="labels.json を含むフォルダ群の親")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-steps", type=int, default=1500,
                    help="最低限回す勾配ステップ数。データが少ないときは epochs を自動で増やす")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-clips", type=int, default=1)
    ap.add_argument("--val-ratio", type=float, default=0.0, help="クリップ単位の検証比率 (撮影方向で層化)。0 = --val-clips を使う")
    ap.add_argument("--name", default="", help="出力名。既定は clubhead_<日付>_<強ラベル数>.onnx (--out 指定時は無視)")
    ap.add_argument("--arch", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--use-weak", action="store_true")
    ap.add_argument("--no-negatives", action="store_true", help="negative (人が削除した自動点) を学習に使わない")
    ap.add_argument("--out", default="", help="ONNX の出力先。省略時は models/clubhead_<日付>_<点数>.onnx (現行の clubhead.onnx は上書きしない)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = load_records(a.use_weak, a.dataset, use_negatives=not a.no_negatives)
    n_neg = sum(1 for r in recs if r["negative"])
    if len(recs) - n_neg < 4:
        print(f"ラベルが足りません ({len(recs)} 点 in {a.dataset})。書き出しでラベルを貯めるか --dataset を確認してください")
        return 1
    train, valid, val_clips = split_by_clip(recs, a.val_clips, a.seed, a.val_ratio)
    n_strong_total = sum(1 for r in recs if not r["negative"] and not r["weak"])
    if not a.out:
        stem = a.name or f"clubhead_{time.strftime('%Y%m%d')}_{n_strong_total}"
        a.out = str(ROOT / "models" / f"{stem}.onnx")
    cams = {}
    for r in recs:
        cams.setdefault(r["camera"], {"train": set(), "val": set()})["val" if r["clip"] in val_clips else "train"].add(r["clip"])
    print("撮影方向ごとのクリップ数 (学習/検証): " + ", ".join(f"{c} {len(v['train'])}/{len(v['val'])}" for c, v in sorted(cams.items())))
    print(f"device={device} 学習 {len(train)} 点 (うち負例 {sum(1 for r in train if r['negative'])}) / "
          f"検証 {len(valid)} 点 (検証クリップ: {val_clips})  クリップ数 {len({r['clip'] for r in recs})}")

    steps_per_epoch = max(1, (len(train) + a.batch - 1) // a.batch)
    epochs = max(a.epochs, (a.min_steps + steps_per_epoch - 1) // steps_per_epoch)
    if epochs != a.epochs:
        print(f"データが少ないため epochs を {a.epochs} → {epochs} に増やします ({steps_per_epoch} ステップ/epoch)")
    a.epochs = epochs
    model = build_model(a.arch).to(device)
    init_head_bias(model)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), a.batch):
            batch = train[i:i + a.batch]
            samples = [make_sample(r, augment=True) for r in batch]
            x = torch.from_numpy(np.stack([smp[0] for smp in samples])).to(device)
            y = torch.from_numpy(np.stack([smp[1] for smp in samples])).to(device)
            m = torch.from_numpy(np.stack([smp[3] for smp in samples])).to(device)
            loss = focal_loss(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(batch)
        sched.step()
        if (ep + 1) % max(10, a.epochs // 8) == 0 or ep == a.epochs - 1:
            ev = evaluate(model, valid, device) if valid else {}
            hist.append({"epoch": ep + 1, "loss": round(tot / len(train), 4), **ev})
            print(f"epoch {ep + 1:3d} loss {tot / len(train):.4f} val {ev}")
    secs = time.time() - t0

    # ONNX
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, 3, INPUT, INPUT)
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["heatmap"], opset_version=13,
                      dynamo=False)
    final_val = evaluate(model, valid, "cpu") if valid else {}
    final_train = evaluate(model, train, "cpu")
    info = {"trained": time.strftime("%Y-%m-%d %H:%M:%S"), "arch": a.arch, "params": n_params,
            "input": INPUT, "stride": STRIDE, "epochs": a.epochs, "device": device, "seconds": round(secs, 1),
            "n_train": len(train), "n_val": len(valid), "val_clips": val_clips,
            "n_negative": n_neg, "use_negatives": not a.no_negatives,
            "clips": sorted({r["clip"] for r in recs}), "use_weak": a.use_weak, "dataset": str(a.dataset),
            "val": final_val, "train_fit": final_train, "history": hist,
            "onnx": str(out), "onnx_mb": round(out.stat().st_size / 1e6, 2)}
    info["cameras"] = {c: {"train": sorted(v["train"]), "val": sorted(v["val"])} for c, v in cams.items()}
    info["n_strong_total"] = n_strong_total
    out.with_suffix(".train.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
