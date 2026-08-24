"""RIFE (GPU) によるフレーム補間スローモーション

ffmpeg の minterpolate はブロックマッチングなので、クラブヘッドのように
1フレームで数百px動く被写体では探索範囲を広げる必要があり、非常に遅い。
RIFE はニューラルネットで中間フレームを直接生成するため、速くて破綻も少ない。

モデル: Practical-RIFE v4.25 (models/rife/train_log/flownet.pkl)
バックエンド: TensorRT fp16 エンジンがあればそれを使う (PyTorch の約1.8倍速)。
             エンジンは解像度・timestep 固定なので、条件が合わなければ
             自動的に PyTorch fp16 に落ちる。

TensorRT エンジンの作り方 (一度だけ、約6分):
    python -c "import rife_trt; rife_trt.export_onnx()"
    trtexec --onnx=models/rife/rife425_1088x1920.onnx --fp16             --saveEngine=models/rife/rife425_fp16.engine             --memPoolSize=workspace:512 --builderOptimizationLevel=1
    ※ --builderOptimizationLevel を上げると VRAM 4GB では落ちる

使い方:
    from rife_slowmo import make_slowmo_rife
    make_slowmo_rife("in.mp4", "out.mp4", factor=2)

CLI:
    python rife_slowmo.py IN.mp4 OUT.mp4 [--factor 2] [--scale 1.0]
                          [--in-frame N] [--out-frame N] [--fp32]
"""

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_MODEL_DIR = Path(__file__).parent / "models" / "rife"
_net = None
_dev = None
_half = False
_TRT_OK = None    # None=未判定 / True=利用中 / False=使えない


def _load(fp16=True):
    """IFNet を読み込む (RIFE_HDv3.Model は学習用の依存があるので直接使う)"""
    global _net, _dev, _half
    if _net is not None:
        return _net, _dev, _half
    import torch
    sys.path.insert(0, str(_MODEL_DIR))
    from train_log.IFNet_HDv3 import IFNet

    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 入力サイズが固定なので、cudnn に最適アルゴリズムを選ばせる
    torch.backends.cudnn.benchmark = True
    net = IFNet()
    sd = torch.load(_MODEL_DIR / "train_log" / "flownet.pkl",
                    map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval().to(_dev)
    _half = bool(fp16 and _dev.type == "cuda")
    if _half:
        net.half()
    if _dev.type == "cuda":
        # channels_last にすると fp16 の畳み込みが Tensor Core に乗る
        net.to(memory_format=torch.channels_last)
    _net = net
    return _net, _dev, _half


def _to_tensor(img, dev, half, pad):
    import torch
    t = torch.from_numpy(img.transpose(2, 0, 1)).to(dev)
    t = (t.half() if half else t.float()) / 255.0
    t = t.unsqueeze(0)
    t = torch.nn.functional.pad(t, pad)
    if t.is_cuda:
        t = t.contiguous(memory_format=torch.channels_last)
    return t


def interpolate(img0, img1, timesteps, scale=1.0, fp16=True):
    """img0 と img1 の間のフレームを timesteps (0<t<1) の位置で生成する

    条件が合えば TensorRT エンジンを使う (PyTorch の約1.8倍速)。
    エンジンは解像度と timestep を固定してビルドされているので、
    2倍スロー・その解像度のときだけ使える。合わなければ PyTorch で動く。
    """
    import torch
    h, w = img0.shape[:2]
    if scale == 1.0 and list(timesteps) == [0.5]:
        out = _try_trt(img0, img1, h, w)
        if out is not None:
            return out
    net, dev, half = _load(fp16)
    # IFNet は 64/scale の倍数を要求する (scale を下げると粗い階層から
    # 始まるので、その分だけ大きな倍数に揃える必要がある)
    unit = max(64, int(round(64 / scale)))
    ph = ((h - 1) // unit + 1) * unit
    pw = ((w - 1) // unit + 1) * unit
    pad = (0, pw - w, 0, ph - h)
    t0 = _to_tensor(img0, dev, half, pad)
    t1 = _to_tensor(img1, dev, half, pad)
    sl = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    out = []
    with torch.no_grad():
        for ts in timesteps:
            _, _, merged = net(torch.cat((t0, t1), 1), ts, sl)
            m = merged[-1][0, :, :h, :w]
            m = (m.float().clamp(0, 1) * 255).byte().cpu().numpy()
            out.append(m.transpose(1, 2, 0))
    return out


def _try_trt(img0, img1, h, w):
    """TensorRT で1枚生成する。使えない条件なら None を返す"""
    global _TRT_OK
    if _TRT_OK is False:
        return None
    try:
        import torch
        import rife_trt
        unit = 64
        ph = ((h - 1) // unit + 1) * unit
        pw = ((w - 1) // unit + 1) * unit
        if not rife_trt.available(ph, pw):
            _TRT_OK = False
            return None
        dev = torch.device("cuda")
        pad = (0, pw - w, 0, ph - h)

        def t(img):
            x = torch.from_numpy(img.transpose(2, 0, 1)).to(dev).float() / 255.
            return torch.nn.functional.pad(x.unsqueeze(0), pad)

        y = rife_trt.infer(torch.cat((t(img0), t(img1)), 1))
        m = (y[0, :, :h, :w].clamp(0, 1) * 255).byte().cpu().numpy()
        _TRT_OK = True
        return [m.transpose(1, 2, 0)]
    except Exception:
        _TRT_OK = False
        return None


def _find_ffmpeg():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ffmpeg_writer import find_ffmpeg
        p = find_ffmpeg()
        if p:
            return p
    except Exception:
        pass
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _encoder_args(ff, encoder, crf):
    """NVENC が使えるなら使う (CPU を空けて、供給を止めないため)"""
    if encoder == "auto":
        try:
            out = subprocess.run([ff, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
            encoder = "h264_nvenc" if "h264_nvenc" in out else "libx264"
        except Exception:
            encoder = "libx264"
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0"], encoder
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"], encoder


def make_slowmo_rife(src, dst, factor=2, scale=1.0, crf=16,
                     in_frame=None, out_frame=None, fp16=True, progress=None,
                     encoder="auto"):
    """RIFE でスローモーションを作る

    factor: 何倍スローにするか (2 なら間に1枚, 4 なら3枚生成)
    scale : 動きが大きいときは 0.5 にすると粗い階層から探すので追従しやすい。
            処理も速くなる
    encoder: auto / h264_nvenc / libx264

    デコード・推論・エンコードを別スレッドで回す。GPU推論の裏で次のフレームを
    読み、書き出しも待たないので、実測で 2 割ほど短くなる。
    """
    import queue
    import threading

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lo = in_frame or 0
    hi = total - 1 if out_frame is None else out_frame

    ff = _find_ffmpeg()
    enc_args, enc_name = _encoder_args(ff, encoder, crf)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-"] + enc_args +           ["-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    read_q = queue.Queue(maxsize=8)
    write_q = queue.Queue(maxsize=16)

    def reader():
        idx = 0
        while True:
            ok, f = cap.read()
            if not ok or idx > hi:
                break
            if idx >= lo:
                read_q.put(f)
            idx += 1
        read_q.put(None)

    def writer():
        while True:
            buf = write_q.get()
            if buf is None:
                break
            proc.stdin.write(buf)

    rt = threading.Thread(target=reader, daemon=True)
    wt = threading.Thread(target=writer, daemon=True)
    rt.start()
    wt.start()

    ts = [i / factor for i in range(1, factor)]
    t0 = time.time()
    prev, n_out, done = None, 0, 0
    try:
        while True:
            f = read_q.get()
            if f is None:
                break
            if prev is not None:
                for mid in interpolate(prev, f, ts, scale, fp16):
                    write_q.put(mid.tobytes())
                    n_out += 1
            write_q.put(f.tobytes())
            n_out += 1
            prev = f
            done += 1
            if progress and done % 10 == 0:
                progress(done / max(hi - lo + 1, 1),
                         f"補間 {done}/{hi - lo + 1}")
    finally:
        write_q.put(None)
        wt.join()
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 失敗:\n" + err[-2000:])
    if progress:
        progress(1.0, "完了")
    return dict(seconds=time.time() - t0, frames_in=hi - lo + 1,
                frames_out=n_out, fps=fps, size=(w, h), factor=factor,
                encoder=enc_name)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="RIFE スローモーション")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--in-frame", type=int, default=None)
    ap.add_argument("--out-frame", type=int, default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--encoder", default="auto",
                    choices=["auto", "h264_nvenc", "libx264"])
    a = ap.parse_args(argv)
    info = make_slowmo_rife(a.src, a.dst, a.factor, a.scale, a.crf,
                            a.in_frame, a.out_frame, not a.fp32,
                            progress=lambda r, m: print(f"\r{r * 100:5.1f}% {m}",
                                                        end="", flush=True))
    print(f"\n{a.dst}  {info['seconds']:.1f}s  "
          f"{info['frames_in']}f -> {info['frames_out']}f "
          f"(x{info['factor']} slow)")


if __name__ == "__main__":
    main(sys.argv[1:])
