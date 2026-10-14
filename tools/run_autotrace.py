"""run_autotrace — origin/feature/autotrace-slowmo の autotrace.py を評価用に実行する。

autotrace ブランチを git worktree で展開し、その専用 venv (torch / ultralytics 入り) から呼ぶ:

    <worktree>/.venv/Scripts/python.exe tools/run_autotrace.py <worktree> <動画> <出力dir> [--in N] [--out N]

出力:
    <出力dir>/autotrace.pred.json   tools/eval_club.py 用の予測 (全フレーム点 + 信頼度 + 確信点)
    <出力dir>/autotrace_overlay.mp4 元映像に軌道を重ねた確認動画
        赤丸   = 各フレームの推定ヘッド位置 (信頼度 >= 1.5)
        橙丸   = 信頼度 < 1.5 (補間区間)
        黄→赤  = confident_points() をベースの trajectory.py でスプライン描画 (編集タブに流れる形)
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2

worktree = Path(sys.argv[1]).resolve()
video = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()
args = sys.argv[4:]
in_f = int(args[args.index("--in") + 1]) if "--in" in args else 0
out_f = int(args[args.index("--out") + 1]) if "--out" in args else -1
sys.path.insert(0, str(worktree))

import autotrace  # noqa: E402  (worktree)
from trajectory import render_trajectory_on_frame  # noqa: E402  (worktree, main と同一)
from clip_manager import TrajectoryData  # noqa: E402

out_dir.mkdir(parents=True, exist_ok=True)
note = platform.processor()
try:
    import torch
    if torch.cuda.is_available():
        note = f"GPU {torch.cuda.get_device_name(0)}"
    else:
        note = "CPU (torch)"
except Exception:
    pass

print(f"[autotrace] {video.name}  in={in_f} out={out_f}  {note}")
t0 = time.time()
res = autotrace.autotrace(str(video), in_f, out_f,
                          progress=lambda r, m: print(f"  {r * 100:3.0f}% {m}", flush=True))
sec = time.time() - t0
cp = res.confident_points()
weak = res.weak_ranges()
print(f"[autotrace] {sec:.1f}s  points={len(res.points)}  confident={len(cp)}  "
      f"impact={res.impact_frame}  ball={res.ball_xy}  weak={weak}")

pred = {
    "method": "autotrace",
    "video": video.name,
    "points": [[int(x), int(y), int(f)] for x, y, f in res.points],
    "confident_points": [[int(x), int(y), int(f)] for x, y, f in cp],
    "conf": {str(f): float(c) for f, c in res.conf.items()},
    "impact_frame": res.impact_frame,
    "ball_xy": list(res.ball_xy) if res.ball_xy else None,
    "weak_ranges": weak,
    "seconds": round(sec, 2),
    "note": note,
}
(out_dir / "autotrace.pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1),
                                             encoding="utf-8")

# --- 確認動画
cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(str(out_dir / "autotrace_overlay_tmp.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
swing = TrajectoryData(points=[(x, y, f) for x, y, f in cp], thickness=4)
by_frame = {f: (x, y) for x, y, f in res.points}
idx = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if len(cp) >= 2:
        render_trajectory_on_frame(fr, [swing], idx)
    for f in range(max(0, idx - 6), idx + 1):  # 直近の推定点を軌跡として
        if f in by_frame:
            c = (0, 0, 255) if res.conf.get(f, 0) >= 1.5 else (0, 165, 255)
            cv2.circle(fr, by_frame[f], 6 if f == idx else 3, c, 2 if f == idx else -1, cv2.LINE_AA)
    cv2.rectangle(fr, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(fr, f"autotrace f={idx} conf={res.conf.get(idx, 0):.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    writer.write(fr)
    idx += 1
cap.release()
writer.release()

# H.264 に変換 (再生互換のため)。ffmpeg が無ければ mp4v のまま残す
import shutil, subprocess  # noqa: E402
ff = shutil.which("ffmpeg")
tmp, final = out_dir / "autotrace_overlay_tmp.mp4", out_dir / "autotrace_overlay.mp4"
if ff:
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", str(final)], check=True)
    tmp.unlink()
else:
    tmp.rename(final)
print(f"[autotrace] overlay -> {final}")
