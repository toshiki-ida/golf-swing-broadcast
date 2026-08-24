"""フレーム補間によるスローモーション生成

収録した動画を、コマ落ちのない滑らかなスローにする。
単純なコマ伸ばし (同じフレームを2回出す) だとカクつくので、
ffmpeg の minterpolate で中間フレームを生成する。

  2倍スロー = 元が29.97pなら 59.94p に補間 → 29.97p で書き出す
              (全フレームが残るので、尺が2倍・速度が1/2になる)

使い方:
    from slowmo import make_slowmo
    make_slowmo("in.mp4", "out.mp4", factor=2, preset="quality")

CLI:
    python slowmo.py IN.mp4 OUT.mp4 [--factor 2] [--preset quality]
                     [--in-frame N] [--out-frame N]
"""

import subprocess
import sys
import time
from pathlib import Path

import cv2

# 動きの探索範囲(px)。クラブヘッドは1フレームで数百px動くので、
# 速い被写体ほど大きな値が要る。ただし処理時間はこれにほぼ比例する。
PRESETS = {
    # 体や背景だけ滑らかになれば良い場合。速い被写体は二重像が残る
    "fast":    dict(search_param=32,  mc_mode="obmc",  vsbmc=0),
    "normal":  dict(search_param=64,  mc_mode="aobmc", vsbmc=1),
    # ゴルフスイングのような高速被写体向け。時間はかかる
    "quality": dict(search_param=200, mc_mode="aobmc", vsbmc=1),
}


def find_ffmpeg():
    """ffmpeg の実行パスを返す (アプリ同梱 → PATH → imageio_ffmpeg同梱)"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ffmpeg_writer import find_ffmpeg as _f
        p = _f()
        if p:
            return p
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def make_slowmo(src, dst, factor=2, preset="quality", crf=16,
                in_frame=None, out_frame=None, progress=None):
    """フレーム補間でスローモーションを作る

    factor: 何倍スローにするか (2 なら半分の速度)
    preset: fast / normal / quality
    in_frame, out_frame: 元動画のフレーム番号で範囲指定 (省略時は全体)
    progress: callable(ratio 0..1, message)
    """
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg が見つかりません")
    if preset not in PRESETS:
        raise ValueError(f"preset は {list(PRESETS)} のいずれか")
    cfg = PRESETS[preset]
    fps, total, w, h = probe(src)

    pre = []
    if in_frame is not None:
        pre += ["-ss", f"{in_frame / fps:.6f}"]
    if out_frame is not None:
        span = (out_frame - (in_frame or 0) + 1) / fps
        pre += ["-t", f"{span:.6f}"]

    vf = (f"minterpolate=fps={fps * factor:.6f}:mi_mode=mci"
          f":mc_mode={cfg['mc_mode']}:me_mode=bidir"
          f":vsbmc={cfg['vsbmc']}:search_param={cfg['search_param']}"
          f",setpts={factor}*PTS")

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    cmd += pre + ["-i", str(src), "-vf", vf, "-r", f"{fps:.6f}",
                  "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                  "-pix_fmt", "yuv420p", "-an", str(dst)]

    t0 = time.time()
    if progress:
        progress(0.0, "補間を開始")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗:\n{r.stderr[-2000:]}")
    if progress:
        progress(1.0, "完了")
    return dict(seconds=time.time() - t0, fps=fps, size=(w, h),
                src_frames=total, factor=factor, preset=preset)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="フレーム補間スローモーション")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--preset", default="quality", choices=list(PRESETS))
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--in-frame", type=int, default=None)
    ap.add_argument("--out-frame", type=int, default=None)
    a = ap.parse_args(argv)
    info = make_slowmo(a.src, a.dst, a.factor, a.preset, a.crf,
                       a.in_frame, a.out_frame,
                       progress=lambda r, m: print(f"{r * 100:3.0f}% {m}"))
    print(f"{a.dst}  {info['seconds']:.1f}s  "
          f"{info['src_frames']}f -> x{info['factor']} slow "
          f"({info['preset']})")


if __name__ == "__main__":
    main(sys.argv[1:])
