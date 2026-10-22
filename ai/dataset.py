"""ai.dataset — 書き出し時のヘッド軌道を「承認済みラベル」として datasets/clubhead/<clip_id>/ に蓄積する。

保存する点:
  - 制御点のうち、人が置いた点 (manual) と、自動生成後に人が修正しなかった点 (auto_confirmed)
  - スプライン補間で埋めた点 (interpolated) は weak として別扱い (既定では学習に使わない)
  - 自動生成されたあと人が削除した点 (negative): その座標は「ヘッドではない」。学習では負例 (その位置のヒートマップを 0 に
    寄せる。切り出し内の他の場所は本当のヘッドかもしれないので損失を掛けない)

各点: frame (clip.in_frame 基準の相対フレーム), x, y (元解像度 px), source, 画像サイズ、
      切り出し (手首中心・身長代替値 ×2.4 の正方形) の位置と、切り出し内の座標。
画像: frames/<frame>.jpg に切り出し画像 (元フレームから)。strong / weak とも保存するが weak は labels.json で区別する。

除外: tests/real/split.yaml の holdout に含まれるクリップ (元ファイル名で判定)。
同じクリップを再書き出ししたらフォルダごと上書きする。

保存先:
  アプリ (編集タブの書き出し) … <プロジェクトフォルダ>/datasets/clubhead/<clip_id>/   ← exe フォルダの外。exe を差し替えても残る
  開発ツール (import/train) の既定 … <リポジトリ>/datasets/clubhead/   (--dataset で指定可)
レイアウト:
  <保存先>/<clip_id>/labels.json
  <保存先>/<clip_id>/frames/<frame>.jpg
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "clubhead"      # 開発ツールの既定 (exe では _internal 配下になるので使わない)


def dataset_dir(project_dir=None) -> Path:
    """ラベルの保存先。project_dir を渡すと <project_dir>/datasets/clubhead (アプリはこちら)。無ければ開発用の既定。"""
    if project_dir:
        return Path(project_dir) / "datasets" / "clubhead"
    return DATASET_DIR
CROP_RATIO = 2.4            # 切り出し正方形の一辺 (身長代替値比)
CROP_SIZE = 320             # 保存する切り出し画像の一辺 (px)
WEAK_STEP_SEC = 0.1         # 補間点を何秒おきに記録するか


def holdout_stems() -> set:
    p = ROOT / "tests" / "real" / "split.yaml"
    if not p.exists():
        return set()
    try:
        import yaml
        sp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return {Path(k).stem for k in (sp.get("holdout") or {})}
    except Exception:
        return set()


def is_holdout(source_path) -> bool:
    stem = Path(str(source_path)).stem.replace("_trim", "")
    return stem in holdout_stems()


def _crop_box(center, body_h, W, H):
    side = int(round(CROP_RATIO * body_h))
    side = max(64, min(side, W, H))   # 縦位置素材でも正方形がフレーム内に収まるように
    x0 = int(round(center[0] - side / 2))
    y0 = int(round(center[1] - side / 2))
    x0 = max(0, min(x0, W - side)) if W >= side else 0
    y0 = max(0, min(y0, H - side)) if H >= side else 0
    return x0, y0, side


class _PoseHelper:
    """骨格 (landmarks) が渡されなかったフレーム用。MediaPipe Pose (画像モード) で手首中点と身長代替値を求める。
    複数人いるときはラベル点 (ヘッド位置) に手首が最も近い人を選ぶ。"""

    def __init__(self):
        self.lm = None
        self.mp = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            from ai import default_pose_model
            mp_path = default_pose_model()
            if mp_path and Path(mp_path).exists():
                opts = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(mp_path)),
                    running_mode=vision.RunningMode.IMAGE, num_poses=4, min_pose_detection_confidence=0.3)
                self.lm = vision.PoseLandmarker.create_from_options(opts)
                self.mp = mp
        except Exception as e:  # pragma: no cover
            print(f"[dataset] MediaPipe を使えません ({e})。切り出しはラベル点中心になります")

    def wrists(self, frame_bgr, label_xy):
        """(手首中点 (x, y), 身長代替値) か None"""
        if self.lm is None:
            return None
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        best = None
        for lm in res.pose_landmarks or []:
            pts = np.array([(p.x * W, p.y * H) for p in lm], dtype=np.float32)
            wc = (pts[15] + pts[16]) / 2
            d = float(np.hypot(wc[0] - label_xy[0], wc[1] - label_xy[1]))
            body_h = float(pts[:, 1].max() - pts[:, 1].min())
            if best is None or d < best[0]:
                best = (d, (float(wc[0]), float(wc[1])), max(body_h, 40.0))
        return None if best is None else best[1:]

    def close(self):
        if self.lm is not None:
            self.lm.close()


def _sample_interpolated(swings, real_fps: float):
    """制御点の間をスプラインで埋めた点を WEAK_STEP_SEC おきに返す [(x, y, frame), ...]"""
    from trajectory import TimedSpline
    step = max(1, int(round(WEAK_STEP_SEC * real_fps)))
    out = []
    for sw in swings:
        pts = sorted(sw.points, key=lambda p: p[2])
        if len(pts) < 2:
            continue
        ctrl_frames = {p[2] for p in pts}
        spline = TimedSpline(pts, 600, handles=getattr(sw, "handles", None) or None)
        f0, f1 = int(np.ceil(pts[0][2])), int(np.floor(pts[-1][2]))
        for f in range(f0, f1 + 1, step):
            if f in ctrl_frames:
                continue
            curve = spline.get_curve_at_frame(f)
            if curve:
                out.append((int(curve[-1][0]), int(curve[-1][1]), f))
    return out


def collect_clip_labels(clip_id: str, source_path, in_frame: int, swings: list, *,
                        corrections: dict | None, landmarks: dict | None, body_h: float | None,
                        real_fps: float, labeler: str = "operator", note: str = "",
                        out_root=None) -> dict | None:
    """書き出し時に呼ぶ。承認済みラベルを <out_root>/<clip_id>/ に保存して要約を返す (holdout なら None)。
    out_root 省略時は開発用の既定 (DATASET_DIR)。アプリは dataset_dir(project_dir) を渡す。"""
    if is_holdout(source_path):
        return None
    pts_all = sorted({tuple(p) for sw in swings for p in sw.points}, key=lambda p: p[2])
    if len(pts_all) < 2:
        return None
    corr = (corrections or {}).get("frames", {})

    def source_of(f):
        c = corr.get(str(f))
        if c is None:
            return "manual"              # 自動生成の記録が無い点 = 人が置いた点
        return {"auto": "auto_confirmed", "moved": "manual", "manual": "manual"}.get(c["source"], "manual")

    strong = [{"frame": f, "x": int(x), "y": int(y), "source": source_of(f)} for x, y, f in pts_all]
    weak = [{"frame": f, "x": x, "y": y, "source": "interpolated"}
            for x, y, f in _sample_interpolated(swings, real_fps)]
    # 人が削除した自動点 = 誤検出。同じフレームに人が置き直した点があってもその座標は負例として有効
    negatives = [{"frame": int(f), "x": int(x), "y": int(y), "source": "negative"}
                 for x, y, f in (corrections or {}).get("deleted", [])]

    out_dir = (Path(out_root) if out_root else DATASET_DIR) / clip_id
    if out_dir.exists():
        shutil.rmtree(out_dir)              # 再書き出しは上書き
    (out_dir / "frames").mkdir(parents=True)
    cap = cv2.VideoCapture(str(source_path))
    W, H = int(cap.get(3)), int(cap.get(4))
    n_total = int(cap.get(7))
    lm = landmarks or {}
    bh = float(body_h) if body_h else None
    pose = None

    def wrist_center(f, fr, label_xy):
        """(手首中点, 身長代替値)。渡された骨格 → 無ければ MediaPipe → それも無ければラベル点中心"""
        nonlocal pose
        l = lm.get(f) or lm.get(int(f)) or lm.get(float(f))
        if l is not None and len(l) > 16:
            return ((l[15][0] + l[16][0]) / 2, (l[15][1] + l[16][1]) / 2), bh or H * 0.45
        if pose is None:
            pose = _PoseHelper()
        got = pose.wrists(fr, label_xy)
        if got is not None:
            return got[0], (bh or got[1])
        return (label_xy[0], label_xy[1]), bh or H * 0.45

    records = []
    for rec in strong + weak + negatives:
        f = rec["frame"]
        abs_f = in_frame + int(f)
        if abs_f < 0 or abs_f >= n_total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_f)
        ok, fr = cap.read()
        if not ok:
            continue
        center, bh_f = wrist_center(f, fr, (rec["x"], rec["y"]))
        x0, y0, side = _crop_box(center, bh_f, W, H)
        crop = fr[y0:y0 + side, x0:x0 + side]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        scale = CROP_SIZE / side
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        name = f"{int(f):05d}.jpg"
        cv2.imwrite(str(out_dir / "frames" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cx, cy = (rec["x"] - x0) * scale, (rec["y"] - y0) * scale
        inside = 0 <= cx < CROP_SIZE and 0 <= cy < CROP_SIZE
        records.append({**rec, "abs_frame": abs_f, "image": f"frames/{name}", "crop": [x0, y0, side],
                        "crop_size": CROP_SIZE, "cx": round(cx, 1), "cy": round(cy, 1), "in_crop": inside,
                        "weak": rec["source"] == "interpolated", "negative": rec["source"] == "negative"})
    cap.release()
    if pose is not None:
        pose.close()
    bh = bh or (float(np.median([r["crop"][2] for r in records]) / CROP_RATIO) if records else H * 0.45)
    meta = {
        "clip_id": clip_id, "source_path": str(source_path), "source_stem": Path(str(source_path)).stem,
        "width": W, "height": H, "in_frame": int(in_frame), "real_fps": float(real_fps),
        "body_h": bh, "labeler": labeler, "note": note, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_strong": sum(1 for r in records if not r["weak"] and not r["negative"]),
        "n_weak": sum(1 for r in records if r["weak"]),
        "n_negative": sum(1 for r in records if r["negative"]),
        "points": records,
    }
    (out_dir / "labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"dir": str(out_dir), "n_strong": meta["n_strong"], "n_weak": meta["n_weak"], "n_negative": meta["n_negative"],
            "sources": {s: sum(1 for r in records if r["source"] == s)
                        for s in ("manual", "auto_confirmed", "interpolated", "negative")}}


def load_all(strong_only: bool = True, root=None) -> list:
    """保存先配下の全ラベルを [(clip_id, record, image_path), ...] で返す。"""
    out = []
    for lab in sorted((Path(root) if root else DATASET_DIR).glob("*/labels.json")):
        meta = json.loads(lab.read_text(encoding="utf-8"))
        for r in meta["points"]:
            if strong_only and (r.get("weak") or r.get("negative")):
                continue
            if not r.get("in_crop", True):
                continue
            out.append((meta["clip_id"], r, lab.parent / r["image"]))
    return out


def summary(root=None) -> dict:
    clips = {}
    for lab in sorted((Path(root) if root else DATASET_DIR).glob("*/labels.json")):
        meta = json.loads(lab.read_text(encoding="utf-8"))
        clips[meta["clip_id"]] = {"strong": meta["n_strong"], "weak": meta["n_weak"],
                                  "negative": meta.get("n_negative", 0), "labeler": meta.get("labeler"),
                                  "source_stem": meta.get("source_stem")}
    return {"clips": clips, "n_clips": len(clips), "n_strong": sum(c["strong"] for c in clips.values()),
            "n_weak": sum(c["weak"] for c in clips.values()), "n_negative": sum(c["negative"] for c in clips.values())}
