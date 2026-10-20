"""inventory_real — tests/real/ の素材の棚卸し (解像度・fps・尺・インターレース・音声・HFR 判定・
撮影方向の自動判定・カメラの固定/パン・ズーム・画面内の人数) を Markdown 表にする。

    python tools/inventory_real.py [--out tests/real/results/<日時>_inventory.md] [--samples 12]

判定方法:
  - HFR 判定      : ai.config.realtime_factor (fps が 29.97 なら HFR 2x とみなす)。fps だけの推定なので
                    「要確認」。実時間倍率は eval_club --all の骨格解析 (スイング所要フレーム数) で裏を取る
  - 撮影方向      : 等間隔に標本化したフレームで MediaPipe Pose を実行し、人物 1 人目の
                    肩幅/胴長の比 (ai.swing_analyzer.estimate_camera_angle) の中央値で判定
  - カメラの動き  : 標本フレーム間で、人物を含まない縁の領域 (上 20% / 左右 12%) の ORB 特徴点を
                    対応付けて相似変換を推定。平行移動 (px) と倍率の変化を報告。
                    平行移動 > 1% の画面幅 or 倍率変化 > 2% を「パン/ズームあり」とする
  - 人数          : 標本フレームごとの Pose 検出数 (num_poses=5) の最大 / 中央値
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai import default_pose_model  # noqa: E402
from ai.config import realtime_factor  # noqa: E402
from ai.swing_analyzer import DTL_MAX_RATIO, FACE_MIN_RATIO, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER  # noqa: E402

REAL_DIR = ROOT / "tests" / "real"
VIDEO_EXTS = (".mp4", ".mov", ".mxf", ".mkv", ".avi")


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,codec_name,width,height,r_frame_rate,field_order,nb_frames,sample_rate,channels",
           "-show_entries", "format=duration,size", "-of", "json", str(path)]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    num, den = v["r_frame_rate"].split("/")
    return {"codec": v["codec_name"], "w": int(v["width"]), "h": int(v["height"]),
            "fps": int(num) / int(den), "field": v.get("field_order", "?"),
            "frames": int(v.get("nb_frames", 0) or 0), "dur": float(d["format"]["duration"]),
            "audio": f"{a[0]['codec_name']} {a[0].get('channels', '?')}ch" if a else "なし",
            "size_mb": int(d["format"]["size"]) / 1e6}


def make_landmarker(num_poses=5):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(default_pose_model())),
        running_mode=vision.RunningMode.IMAGE, num_poses=num_poses,
        min_pose_detection_confidence=0.3)
    return mp, vision.PoseLandmarker.create_from_options(opts)


def border_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    m[: int(h * 0.20), :] = 255
    m[:, : int(w * 0.12)] = 255
    m[:, w - int(w * 0.12):] = 255
    return m


def camera_motion(gray_a, gray_b, mask, orb):
    """縁領域の特徴点対応から (平行移動 px, 倍率) を推定。対応が足りなければ None。"""
    ka, da = orb.detectAndCompute(gray_a, mask)
    kb, db = orb.detectAndCompute(gray_b, mask)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(da, db), key=lambda m: m.distance)[:200]
    if len(matches) < 12:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None or inl.sum() < 8:
        return None
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    shift = float(np.hypot(M[0, 2], M[1, 2]))
    return shift, scale


def analyze_video(path: Path, samples: int, mp, landmarker, orb) -> dict:
    info = ffprobe(path)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or info["frames"]
    W, H = int(cap.get(3)), int(cap.get(4))
    idxs = np.linspace(0, max(0, n - 1), samples).round().astype(int)
    small_w = 640
    sc = small_w / W
    mask = border_mask(int(H * sc), small_w)
    people, ratios, grays = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        people.append(len(res.pose_landmarks))
        if res.pose_landmarks:
            # 画面内で最も大きい人物 (肩〜腰が長い) を主役とみなす
            best = max(res.pose_landmarks,
                       key=lambda lm: abs(lm[L_HIP].y - lm[L_SHOULDER].y))
            sh_l = np.array([best[L_SHOULDER].x * W, best[L_SHOULDER].y * H])
            sh_r = np.array([best[R_SHOULDER].x * W, best[R_SHOULDER].y * H])
            hip = (np.array([best[L_HIP].x * W, best[L_HIP].y * H]) + np.array([best[R_HIP].x * W, best[R_HIP].y * H])) / 2
            torso = np.linalg.norm((sh_l + sh_r) / 2 - hip) + 1e-6
            ratios.append(float(np.linalg.norm(sh_l - sh_r) / torso))
        grays.append(cv2.cvtColor(cv2.resize(fr, (small_w, int(H * sc))), cv2.COLOR_BGR2GRAY))
    cap.release()

    shifts, scales, fails = [], [], 0
    for a, b in zip(grays[:-1], grays[1:]):
        r = camera_motion(a, b, mask, orb)
        if r is None:
            fails += 1
        else:
            shifts.append(r[0] / sc)
            scales.append(r[1])
    if ratios:
        ratio = float(np.median(ratios))
        angle = "dtl" if ratio < DTL_MAX_RATIO else "face_on" if ratio > FACE_MIN_RATIO else "unknown(斜め)"
        conf = (min(1.0, (DTL_MAX_RATIO - ratio) / DTL_MAX_RATIO + 0.5) if angle == "dtl"
                else min(1.0, (ratio - FACE_MIN_RATIO) / 0.35 + 0.5) if angle == "face_on" else 0.0)
    else:
        ratio, angle, conf = float("nan"), "検出なし", 0.0
    max_shift = max(shifts) if shifts else float("nan")
    max_zoom = max(abs(s - 1.0) for s in scales) if scales else float("nan")
    if shifts:
        moving = max_shift > 0.01 * W or max_zoom > 0.02
        cam = ("パン/ズームあり" if moving else "固定") + f" (移動 {max_shift:.0f}px, 倍率 ±{max_zoom * 100:.1f}%)"
    else:
        cam = "判定不能 (特徴点不足)"
    factor = realtime_factor(info["fps"], "auto")
    return {**info, "name": path.name, "factor": factor,
            "hfr": "HFR 2x (fps=29.97, 要確認)" if factor > 1 else "通常 (実時間)",
            "angle": angle, "ratio": ratio, "conf": conf, "camera": cam,
            "people_max": max(people) if people else 0, "people_med": float(np.median(people)) if people else 0,
            "people_frames": f"{sum(1 for p in people if p > 1)}/{len(people)}"}


def main() -> int:
    args = sys.argv[1:]
    samples = int(args[args.index("--samples") + 1]) if "--samples" in args else 12
    out = Path(args[args.index("--out") + 1]) if "--out" in args else \
        REAL_DIR / "results" / f"{datetime.datetime.now():%Y%m%d_%H%M}_inventory.md"
    videos = sorted(p for p in REAL_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"素材がありません: {REAL_DIR}")
        return 1
    mp, landmarker = make_landmarker()
    orb = cv2.ORB_create(1500)
    rows = []
    for v in videos:
        print(f"[inventory] {v.name}", flush=True)
        try:
            rows.append(analyze_video(v, samples, mp, landmarker, orb))
        except Exception as e:
            rows.append({"name": v.name, "error": str(e)})
    landmarker.close()

    lines = [f"# tests/real 棚卸し ({datetime.datetime.now():%Y-%m-%d %H:%M}, {len(rows)} 本, 標本 {samples} フレーム/本)", "",
             "| 素材 | 解像度 | fps | 尺 | フレーム数 | 走査 | 音声 | HFR 判定 | 撮影方向 (肩幅/胴長, 信頼度) | カメラ | 人数 (最大 / 中央値 / 2 人以上の標本) | サイズ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | エラー: {r['error']} |" + " |" * 10)
            continue
        lines.append(f"| {r['name']} | {r['w']}x{r['h']} | {r['fps']:.2f} | {r['dur']:.1f}s | {r['frames']} | {r['field']} | {r['audio']} | "
                     f"{r['hfr']} | {r['angle']} ({r['ratio']:.2f}, {r['conf']:.2f}) | {r['camera']} | "
                     f"{r['people_max']} / {r['people_med']:.0f} / {r['people_frames']} | {r['size_mb']:.0f} MB |")
    lines += ["", "判定方法はスクリプト冒頭の docstring を参照。HFR 判定は fps のみによる推定。",
              "撮影方向は標本フレームの中央値 (アドレス時ではない) なので、eval_club --all の解析結果と併せて見る。"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
