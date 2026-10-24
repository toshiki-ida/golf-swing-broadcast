"""run_club_tracker — ai/club_tracker.py を評価用に実行し、予測 JSON と確認動画を出力する。

    python tools/run_club_tracker.py <動画> <出力dir> [--in N] [--out N] [--interlaced] [--debug]
                                     [--no-detector | --detector | --model <onnx>]   (省略時: models/clubhead.onnx があれば使う)

出力:
    <出力dir>/club_tracker.pred.json     tools/eval_club.py 用 (全フレーム点 + 信頼度 + 確信点)
    <出力dir>/club_tracker_overlay.mp4   元映像に軌道を重ねた確認動画
        緑丸   = 観測できたヘッド位置 (信頼度 >= 1.5)
        橙丸   = 低信頼 / 補間
        水色   = 手首中点と探索円環
        黄→赤  = confident_points() をベースの trajectory.py でスプライン描画
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):   # Windows の cp932 コンソールでも落ちないように
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.club_tracker import track_club, CONF_OK, R_MIN_RATIO, R_MAX_RATIO  # noqa: E402
from ai.store import save_club_track  # noqa: E402
from clip_manager import TrajectoryData  # noqa: E402
from trajectory import render_trajectory_on_frame  # noqa: E402

video = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
args = sys.argv[3:]
in_f = int(args[args.index("--in") + 1]) if "--in" in args else 0
out_f = int(args[args.index("--out") + 1]) if "--out" in args else -1
interlaced = "--interlaced" in args
use_det = False if "--no-detector" in args else (True if "--detector" in args else None)
det_model = Path(args[args.index("--model") + 1]) if "--model" in args else None
if det_model is not None:
    use_det = True
debug_dir = out_dir / "debug" if "--debug" in args else None
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[club_tracker] {video.name}  in={in_f} out={out_f} interlaced={interlaced}")
t0 = time.time()
res = track_club(str(video), in_f, out_f, interlaced=interlaced, debug_dir=debug_dir, use_detector=use_det,
                 detector_model=det_model,
                 progress=lambda r, m: print(f"  {r * 100:3.0f}% {m}", flush=True))
cp = res.confident_points()
weak = res.weak_ranges()
print(f"[club_tracker] {res.seconds:.1f}s (pose {res.pose_seconds:.1f}s)  frames={res.frames_total}"
      f"  observed={res.frames_detected}  confident={sum(1 for v in res.conf.values() if v >= CONF_OK)}"
      f"  control={len(cp)}  body_h={res.body_h:.0f}px  reliable={res.reliable} quality={res.quality}  weak={weak[:4]}")

pred = {
    "method": "club_tracker" + (f"+det:{getattr(res, 'detector_name', '')}" if getattr(res, "detector_used", False) else ""),
    "video": video.name,
    "points": [[x, y, f] for x, y, f in res.points],
    "confident_points": [[x, y, f] for x, y, f in cp],
    "conf": {str(f): round(c, 2) for f, c in res.conf.items()},
    "weak_ranges": weak,
    "quality": res.quality, "reliable": res.reliable, "real_fps": res.real_fps,
    "raw": {str(f): [round(x, 1), round(y, 1), round(c, 2)] for f, (x, y, c) in res.raw.items()},
    "seconds": round(res.seconds, 2),
    "pose_seconds": round(res.pose_seconds, 2),
    "note": "CPU mediapipe",
    "trace": {str(f): v for f, v in res.trace.items()},
}
# 観測の無い区間を推定で埋めた軌道 (--no-fill で省略)。評価は tools/eval_prior.py
if "--no-fill" not in args:
    try:
        from ai.trajectory_prior import fill_trajectory, load_prior, swing_frame_range, estimate_camera_from_landmarks
        _cap = cv2.VideoCapture(str(video))
        _W, _H = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()
        _obs = {int(f): (x, y) for seg, q, ok in res.segment_quality() if ok for x, y, f in seg}
        _angle = estimate_camera_from_landmarks(res.landmarks, _W, _H, res.real_fps)
        _prior = load_prior(_angle) if _angle in ("dtl", "face_on") else None
        _rng = swing_frame_range(res.landmarks, _W, _H, res.real_fps) or ((min(_obs), max(_obs)) if _obs else None)
        if _rng is not None:
            if _obs:
                _rng = (min(_rng[0], min(_obs)), max(_rng[1], max(_obs)))
            _filled = fill_trajectory(_obs, res.wrist, res.body_h, res.real_fps, list(range(_rng[0], _rng[1] + 1)), prior=_prior)
            pred["filled"] = [[round(x, 1), round(y, 1), f, s] for x, y, f, s in _filled]
            pred["fill_info"] = {"angle": _angle, "prior": bool(_prior), "range": list(_rng),
                                 "estimated": sum(1 for p in _filled if p[3] == "estimated")}
            (out_dir / "club_tracker.pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[club_tracker] fill: {pred['fill_info']}")
    except Exception as _e:
        print(f"[club_tracker] fill failed: {_e}")
# 編集タブと同じ形式 (手首・骨格・接続フラグ入り) も残す。閾値の再計算 (tools/gate_tune.py) に使う
from ai.store import save_club_track  # noqa: E402
save_club_track(out_dir, res, 0, str(video))
# 編集タブと同じ形式 (in_frame 基準の相対フレーム + 骨格) も残す → ai.sns_render の --traj に渡せる
save_club_track(out_dir, res, in_f, str(video))

# --- 確認動画 (インターレース指定時はフィールド展開した 2 倍フレームで書く)
cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
tmp, final = out_dir / "club_tracker_overlay_tmp.mp4", out_dir / "club_tracker_overlay.mp4"
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps * (2 if interlaced else 1), (W, H))
swing = TrajectoryData(points=[(x, y, f) for x, y, f in cp], thickness=4)
by_frame = {f: (x, y) for x, y, f in res.points}
r_min, r_max = int(R_MIN_RATIO * res.body_h), int(R_MAX_RATIO * res.body_h)


def draw(fr, fno):
    if len(cp) >= 2:
        render_trajectory_on_frame(fr, [swing], fno)
    w = res.wrist.get(fno)
    if w:
        cv2.circle(fr, (int(w[0]), int(w[1])), r_min, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(fr, (int(w[0]), int(w[1])), r_max, (255, 255, 0), 1, cv2.LINE_AA)
    step = 0.5 if interlaced else 1
    for k in range(6, -1, -1):
        f = fno - k * step
        if f in by_frame:
            c = (0, 220, 0) if res.conf.get(f, 0) >= CONF_OK else (0, 165, 255)
            cv2.circle(fr, by_frame[f], 7 if k == 0 else 3, c, 2 if k == 0 else -1, cv2.LINE_AA)
    cv2.rectangle(fr, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(fr, f"club_tracker f={fno} conf={res.conf.get(fno, 0):.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return fr


idx = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if interlaced:
        from field_processor import build_high_frame_rate_frames
        a, b = build_high_frame_rate_frames(fr, True)
        writer.write(draw(a, float(idx)))
        writer.write(draw(b, idx + 0.5))
    else:
        writer.write(draw(fr, idx))
    idx += 1
cap.release()
writer.release()
ff = shutil.which("ffmpeg")
if ff:
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", str(final)], check=True)
    tmp.unlink()
else:
    tmp.rename(final)
print(f"[club_tracker] overlay -> {final}")
