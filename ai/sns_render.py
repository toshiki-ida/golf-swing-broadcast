"""ai.sns_render — クラブヘッド軌道を主役にした SNS 向け縦型動画 (1080x1920 / 30fps / H.264) の生成。

swingcut/render.py の構成 (通常速 → スロー → サマリーカード) を、ベースの軌道描画
(trajectory.py のスプライン・グラデーション) を主役にして作り直したもの。

構成:
  1. 通常速度 (アドレス少し前 〜 フィニッシュ)。軌道をフレーム同期で伸ばしながら描く
  2. 1/N スロー (既定 1/4)。軌道 + 骨格線 (任意)。トップとインパクトで静止して局面名を表示
  3. サマリーカード (任意、既定 ON): テンポ (バックスイング : ダウンスイング) 等

素材の fps と実時間:
  realtime_factor = 素材 1 フレームが実時間の何フレーム分か。通常 59.94p は 1.0、
  HFR 2x (29.97fps 記録で 1/2 速) は 4.0。出力は常に 30fps 実時間で、
  通常速は real_fps/30 フレームおきに、スローは real_fps/(30×N) おきにソースを進める。
  59.94p 素材なら 1/4 スローは 2 枚繰り返しで済み、HFR 素材なら全フレームが使われる。

レイアウト (横位置 16:9 素材 → 縦 9:16):
  auto: 軌道と人物の範囲が 9:16 のクロップに収まればクロップ、収まらなければ
        全体を幅 1080 に縮めてぼかし背景の上に置く (pad)。crop / pad で固定もできる。

使い方 (CLI):
    python -m ai.sns_render <動画> --traj <trajectory.json> --out short.mp4 [--in N --out-frame N]
                            [--track club_track.json] [--no-skeleton] [--no-summary] [--realtime 1.0]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_manager import TrajectoryData  # noqa: E402
from ffmpeg_writer import _build_encoder_args, find_ffmpeg  # noqa: E402
from trajectory import TimedSpline, draw_gradient_trail, hex_to_bgr  # noqa: E402
from ai import resource_root  # noqa: E402
from ai.config import AI_DEFAULTS  # noqa: E402

OUT_W, OUT_H, FPS = 1080, 1920, 30
SAFE_TOP = int(OUT_H * 0.15)      # SNS の UI に被らない領域
SAFE_BOTTOM = int(OUT_H * 0.80)
MAX_ZOOM = 1.6
CONTENT_MARGIN = 0.06
SPLINE_RES = 600
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]
AUDIO_SR = 48000


# ---------------------------------------------------------------- テキスト

_fonts: dict = {}


def _font(size: int):
    from PIL import ImageFont
    if size in _fonts:
        return _fonts[size]
    path = resource_root() / "assets" / "fonts" / "NotoSansJP.ttf"
    try:
        f = ImageFont.truetype(str(path), size)
        try:
            f.set_variation_by_axes([700])
        except Exception:
            pass
    except Exception:
        print(f"[sns_render] フォントがありません: {path} (setup_assets.py を実行)。既定フォントで代用")
        f = ImageFont.load_default()
    _fonts[size] = f
    return f


def draw_texts(frame_bgr, items):
    """白文字 + 黒縁取り。items: [(text, (x, y), size, anchor, fill_rgb), ...]"""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, size, anchor, fill in items:
        d.text(xy, text, font=_font(size), fill=fill, anchor=anchor,
               stroke_width=max(2, size // 14), stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def dark_band(img, y0, y1, alpha=0.5):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (OUT_W, y1), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ---------------------------------------------------------------- 局面 (軌道から推定)

def derive_phases(swings, in_frame: int, dense_points=None) -> dict:
    """軌道から 4 局面のソースフレーム番号を推定する。

    dense_points: club_track.json の全フレーム点 [(x, y, rel_frame), ...] があればそれを優先する
    (制御点 18 点に間引くとインパクトの瞬間が落ちることがあるため)。

    address = 動き出しの直前の点 (先頭の静止区間は除く), finish = 最後の点,
    impact  = 前半 1/4 を過ぎてからアドレス位置に最も近づいた点 (ヘッドはボール位置に一度だけ戻る),
    top     = アドレスとインパクトの間で速度が最も落ちる点 (切り返し)。
    クリップは In/Out でスイング 1 回に切られている前提 (編集タブと同じ)。
    """
    fs_l, xs_l, ys_l = [], [], []
    if dense_points:
        for x, y, f in sorted(dense_points, key=lambda q: q[2]):
            fs_l.append(float(f))
            xs_l.append(float(x))
            ys_l.append(float(y))
    else:
        main = max(swings, key=lambda s: len(s.points))
        pts = sorted(main.points, key=lambda p: p[2])
        # 制御点そのものではなく、スプラインを 1 フレームごとに標本化した位置で判定する
        spline = TimedSpline(pts, SPLINE_RES, handles=getattr(main, "handles", None) or None)
        f0, f1 = int(np.floor(pts[0][2])), int(np.ceil(pts[-1][2]))
        for f in range(f0, f1 + 1):
            curve = spline.get_curve_at_frame(f)
            if curve:
                fs_l.append(f)
                xs_l.append(curve[-1][0])
                ys_l.append(curve[-1][1])
    fs = np.array(fs_l, dtype=float)
    xs = np.array(xs_l, dtype=float)
    ys = np.array(ys_l, dtype=float)
    n = len(fs)
    fin_i = n - 1
    # アドレス = 動き出しの直前 (静止保持区間を除く)。速度が最大値の 8% を超えた最初の点の 1 つ前
    spd_all = np.hypot(np.gradient(xs), np.gradient(ys)) / np.maximum(np.gradient(fs), 1e-6)
    s_thr = 0.08 * float(spd_all.max()) if n > 2 else 0.0
    onset = next((i for i in range(n) if spd_all[i] > s_thr), 0)
    addr_i = max(0, onset - 1)
    lo = addr_i + max(1, int((n - addr_i) * 0.25))
    dist = np.hypot(xs - xs[addr_i], ys - ys[addr_i])
    # インパクト = 前半 1/4 を過ぎてから最初にアドレス位置へ戻った極小点 (しきい値: 最大距離の 25%)。
    # 後ろの誤追跡点がたまたまボール付近にあっても拾わない
    dmax = float(dist.max())
    thr = 0.25 * dmax
    # まずヘッドが十分遠ざかった点 (バックスイング) を探し、その後で最初にアドレス位置へ戻った極小点をインパクトにする。
    # ゆっくりしたテイクバック中の「まだ近い点」を拾わないため
    far_i = next((i for i in range(lo, n) if dist[i] >= 0.6 * dmax), lo)
    imp_i = None
    for i in range(far_i + 1, n):
        nxt = dist[i + 1] if i + 1 < n else np.inf
        if dist[i] <= thr and dist[i] <= nxt:
            imp_i = i
            break
    if imp_i is None:
        imp_i = int(far_i + np.argmin(dist[far_i:]))
    if imp_i <= addr_i + 1:
        imp_i = min(addr_i + 2, fin_i)
    # トップ = 切り返し (アドレス→インパクトの 40〜95% の区間で速度が最も落ちる点)。
    # DTL では「最も高い点」はテイクバック途中になるため高さでは決めない
    if imp_i > addr_i + 2:
        lo_t = addr_i + max(1, int((imp_i - addr_i) * 0.4))
        hi_t = max(lo_t + 1, addr_i + int((imp_i - addr_i) * 0.95))
        spd = []
        for i in range(lo_t, hi_t + 1):
            a, b = max(addr_i, i - 1), min(imp_i, i + 1)
            dt = max(1e-6, fs[b] - fs[a])
            spd.append(np.hypot(xs[b] - xs[a], ys[b] - ys[a]) / dt)
        top_i = lo_t + int(np.argmin(spd))
    else:
        top_i = min(addr_i + 1, imp_i)
    return {"address": in_frame + int(round(fs[addr_i])), "top": in_frame + int(round(fs[top_i])),
            "impact": in_frame + int(round(fs[imp_i])), "finish": in_frame + int(round(fs[fin_i]))}


# ---------------------------------------------------------------- レイアウト

def compute_layout(W, H, bbox, mode="auto"):
    """(kind, rect) を返す。kind='crop' なら rect=(x0, y0, w, h) の固定クロップ、'pad' なら全体を縮小して置く。"""
    x0, y0, x1, y1 = bbox
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    need_w = bw / (1 - 2 * CONTENT_MARGIN)
    need_h = bh / (1 - 2 * CONTENT_MARGIN)
    crop_h = max(need_h, need_w * OUT_H / OUT_W, H / MAX_ZOOM)
    crop_h = min(crop_h, H)
    crop_w = crop_h * OUT_W / OUT_H
    fits = crop_w <= W + 1e-6
    if mode == "pad" or (mode == "auto" and not fits):
        return "pad", None
    crop_w = min(crop_w, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx = float(np.clip(cx - crop_w / 2, 0, W - crop_w))
    ry = float(np.clip(cy - crop_h / 2, 0, H - crop_h))
    return "crop", (int(round(rx)), int(round(ry)), int(round(crop_w)), int(round(crop_h)))


def make_transform(W, H, kind, rect):
    """ソースフレーム → 1080x1920 の変換関数を返す。"""
    if kind == "crop":
        x0, y0, w, h = rect

        def tf(frame):
            return cv2.resize(frame[y0:y0 + h, x0:x0 + w], (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        return tf

    fg_h = int(round(H * OUT_W / W))

    def tf(frame):
        # 背景: 全体を覆うように拡大してぼかし、暗くする
        s = max(OUT_W / W, OUT_H / H)
        bw, bh = int(W * s) + 1, int(H * s) + 1
        bg = cv2.resize(frame, (bw, bh), interpolation=cv2.INTER_LINEAR)
        bx, by = (bw - OUT_W) // 2, (bh - OUT_H) // 2
        bg = bg[by:by + OUT_H, bx:bx + OUT_W]
        bg = cv2.GaussianBlur(bg, (0, 0), 25)
        bg = cv2.convertScaleAbs(bg, alpha=0.45, beta=0)
        fg = cv2.resize(frame, (OUT_W, fg_h), interpolation=cv2.INTER_LANCZOS4)
        y = (OUT_H - fg_h) // 2
        bg[y:y + fg_h] = fg
        return bg
    return tf


# ---------------------------------------------------------------- 描画

class SwingDrawer:
    """TrajectoryData をフレーム同期で描く (編集タブの書き出しと同じスプライン/グラデーション)。"""

    def __init__(self, swings, color_pos=(0.0, 1.0)):
        self.items = []
        self.cps, self.cpe = color_pos
        for s in swings:
            if len(s.points) < 2:
                continue
            self.items.append({
                "spline": TimedSpline(s.points, SPLINE_RES, handles=getattr(s, "handles", None) or None),
                "c0": hex_to_bgr(s.color_start_hex), "c1": hex_to_bgr(s.color_end_hex),
                "thick": s.thickness, "alpha": getattr(s, "alpha", 0.85),
                "blur": getattr(s, "blur", 0), "end": getattr(s, "end_frame", -1),
            })

    def draw(self, frame, rel_frame):
        for it in self.items:
            if it["end"] >= 0 and rel_frame > it["end"]:
                continue
            curve = it["spline"].get_curve_at_frame(rel_frame)
            if not curve or len(curve) < 2:
                continue
            ratio = len(curve) / max(len(it["spline"]._curve), 1)
            draw_gradient_trail(frame, curve, it["c0"], it["c1"], it["thick"], it["alpha"],
                                blur=it["blur"], color_pos_start=self.cps / ratio if ratio else 0.0,
                                color_pos_end=self.cpe / ratio if ratio else 1.0)
        return frame


def draw_skeleton(frame, lm, W):
    t = max(2, int(round(W / 540)))
    for a, b in SKELETON:
        pa, pb = lm[a], lm[b]
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 200, 0), t, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------- フレーム供給

class FrameReader:
    """順方向読みを基本にした簡易ランダムアクセス。"""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"動画を開けません: {path}")
        self.pos = -1
        self.last = None

    def get(self, idx):
        idx = int(idx)
        if idx == self.pos and self.last is not None:
            return self.last
        if idx < self.pos or idx > self.pos + 12:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx - 1
        while self.pos < idx:
            ok, fr = self.cap.read()
            if not ok:
                return self.last
            self.pos += 1
            self.last = fr
        return self.last

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------- 音声

def _audio_inputs(mode, bgm_path, src_path, src_start_sec, src_len_sec, total_sec, tmp_dir):
    """ffmpeg の音声入力引数と音声オプションを返す。"""
    ff = find_ffmpeg()
    if mode == "bgm" and bgm_path and Path(bgm_path).exists():
        return ["-stream_loop", "-1", "-i", str(bgm_path)], ["-af", f"afade=t=out:st={max(0, total_sec - 1):.2f}:d=1"]
    if mode == "source" and ff:
        wav = Path(tmp_dir) / "sns_src_audio.wav"
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{src_start_sec:.3f}", "-t", f"{src_len_sec:.3f}",
                            "-i", str(src_path), "-vn", "-ac", "2", "-ar", str(AUDIO_SR), str(wav)],
                           capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000:
            return ["-i", str(wav)], ["-af", f"apad=whole_dur={total_sec:.3f}"]
    return ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_SR}"], []


# ---------------------------------------------------------------- 本体

def render_sns(video_path, out_path, *, in_frame: int = 0, out_frame: int = -1,
               swings: list, landmarks: Optional[dict] = None, phases: Optional[dict] = None,
               realtime_factor: float = 1.0, options: Optional[dict] = None,
               extra_lines: Optional[list] = None, dense_points: Optional[list] = None,
               progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """縦型動画を書き出す。swings の点のフレーム番号は in_frame 基準の相対値 (編集タブと同じ)。"""
    opt = dict(AI_DEFAULTS["sns"])
    opt.update(options or {})
    t_start = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def report(r, m):
        if progress:
            progress(r, m)

    cap = cv2.VideoCapture(str(video_path))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if out_frame < 0 or out_frame >= total:
        out_frame = total - 1
    real_fps = fps * realtime_factor
    swings = [s for s in swings if len(s.points) >= 2]
    if not swings:
        raise ValueError("軌道がありません")
    landmarks = landmarks or {}
    if phases is None:
        phases = derive_phases(swings, in_frame, dense_points)
    addr, top, imp, fin = (phases["address"], phases["top"], phases["impact"], phases["finish"])
    addr, fin = max(in_frame, addr), min(out_frame, fin)

    # --- レイアウト: 軌道 + 骨格の範囲から固定クロップ
    xs = [p[0] for s in swings for p in s.points]
    ys = [p[1] for s in swings for p in s.points]
    if landmarks:
        for lm in landmarks.values():
            xs.extend(p[0] for p in lm)
            ys.extend(p[1] for p in lm)
    m = 0.04
    bbox = (max(0, min(xs) - m * W), max(0, min(ys) - m * H), min(W, max(xs) + m * W), min(H, max(ys) + m * H))
    kind, rect = compute_layout(W, H, bbox, opt.get("layout", "auto"))
    tf = make_transform(W, H, kind, rect)
    drawer = SwingDrawer(swings)
    reader = FrameReader(video_path)
    slow = max(1, int(opt.get("slow_factor", 4)))
    freeze_n = int(round(float(opt.get("freeze_sec", 0.7)) * FPS))
    summary_n = int(round(float(opt.get("summary_sec", 3.0)) * FPS)) if opt.get("summary", True) else 0
    use_skel = bool(opt.get("skeleton", True)) and bool(landmarks)
    pre = int(round(float(opt.get("pre_frames", 6)) * real_fps / FPS))
    s1_start = max(in_frame, addr - pre)

    # --- タイムライン (出力フレーム → ソースフレーム)
    plan = []   # (src_frame, section, label)
    step1 = real_fps / FPS
    k = 0
    while True:
        src = s1_start + int(round(k * step1))
        if src > fin:
            break
        plan.append((src, "normal", None))
        k += 1
    step2 = real_fps / (FPS * slow)
    k = 0
    frozen = set()
    while True:
        src = addr + int(round(k * step2))
        if src > fin:
            break
        plan.append((src, "slow", None))
        for key, name in ((top, "トップ"), (imp, "インパクト")):
            if src >= key and key not in frozen:
                frozen.add(key)
                plan.extend([(src, "freeze", name)] * freeze_n)
        k += 1
    n_video = len(plan)
    total_frames = n_video + summary_n
    total_sec = total_frames / FPS

    # --- サマリーの数値
    back_s = max(0.0, (top - addr) / real_fps)
    down_s = max(0.0, (imp - top) / real_fps)
    tempo = back_s / down_s if down_s > 0 else None

    # --- エンコーダ
    ff = find_ffmpeg()
    if not ff:
        raise FileNotFoundError("ffmpeg が見つかりません")
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    a_in, a_opts = _audio_inputs(opt.get("audio_mode", "mute"), opt.get("bgm_path", ""), video_path,
                                 s1_start / fps, (fin - s1_start + 1) / fps, total_sec, out_path.parent)
    enc = _build_encoder_args(int(opt.get("crf", 18)), "medium", hw_encode=bool(opt.get("hw_encode", True)))
    cmd = ([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "pipe:0"] + a_in
           + ["-map", "0:v:0", "-map", "1:a:0", "-shortest"] + enc
           + ["-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-ac", "2"] + a_opts + [str(tmp)])
    log_path = out_path.parent / "ffmpeg_sns.log"
    n_out = 0
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def emit(img):
            nonlocal n_out
            proc.stdin.write(img.tobytes())
            n_out += 1

        try:
            last_key = None
            cache = None
            for i, (src, section, label) in enumerate(plan):
                key = (src, section, label)
                if key != last_key:
                    fr = reader.get(src)
                    if fr is None:
                        break
                    fr = fr.copy()
                    rel = src - in_frame
                    if section == "slow" or section == "freeze":
                        if use_skel and rel in landmarks:
                            draw_skeleton(fr, landmarks[rel], W)
                    drawer.draw(fr, rel)
                    big = tf(fr)
                    if section == "slow":
                        big = draw_texts(big, [(f"×1/{slow} スロー", (OUT_W - 60, SAFE_TOP + 60), 56, "rm", (255, 255, 255))])
                    elif section == "freeze":
                        big = dark_band(big, SAFE_TOP + 110, SAFE_TOP + 260, 0.5)
                        big = draw_texts(big, [(label, (OUT_W // 2, SAFE_TOP + 185), 76, "mm", (255, 255, 255))])
                    cache, last_key = big, key
                emit(cache)
                if i % 30 == 0:
                    report(0.05 + 0.85 * i / max(1, total_frames), "縦型動画を書き出し")

            if summary_n:
                base = reader.get(addr)
                card = cv2.convertScaleAbs(tf(base.copy()), alpha=0.30, beta=0)
                items = [("スイング解析", (OUT_W // 2, SAFE_TOP + 72), 76, "mm", (255, 255, 255)),
                         ("テンポ (バックスイング : ダウンスイング)", (OUT_W // 2, 560), 40, "mm", (190, 190, 190)),
                         (f"{back_s:.2f} s : {down_s:.2f} s", (OUT_W // 2, 640), 56, "mm", (255, 255, 255)),
                         (f"= {tempo:.2f} : 1" if tempo else "= -", (OUT_W // 2, 720), 56, "mm", (255, 255, 255)),
                         ("スイング時間 (アドレス → フィニッシュ)", (OUT_W // 2, 860), 40, "mm", (190, 190, 190)),
                         (f"{(fin - addr) / real_fps:.2f} s", (OUT_W // 2, 940), 56, "mm", (255, 255, 255))]
                y = 1080
                for line in (extra_lines or [])[:4]:
                    items.append((line, (90, y), 50, "lm", (255, 255, 255)))
                    y += 70
                card = draw_texts(card, items)
                for _ in range(summary_n):
                    emit(card)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            rc = proc.wait()
            reader.close()
    if rc != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg がエラー終了 (exit {rc})。ログ: {log_path}")
    tmp.replace(out_path)
    report(1.0, "完了")
    return {"out_path": str(out_path), "frames": n_out, "seconds": time.time() - t_start,
            "layout": kind, "rect": rect, "phases": phases, "real_fps": real_fps,
            "skeleton": use_skel, "summary": bool(summary_n), "tempo": tempo}


# ---------------------------------------------------------------- CLI

def _load_swings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("swings", data) if isinstance(data, dict) else data
    out = []
    for d in items:
        out.append(TrajectoryData(points=[tuple(p) for p in d["points"]],
                                  color_start_hex=d.get("color_start_hex", "#FFFF00"),
                                  color_end_hex=d.get("color_end_hex", "#FF0000"),
                                  thickness=d.get("thickness", 3), end_frame=d.get("end_frame", -1),
                                  blur=d.get("blur", 0), alpha=d.get("alpha", 0.85),
                                  handles=[tuple(h) if h else None for h in d.get("handles", [])]))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="クラブヘッド軌道を主役にした縦型動画 (1080x1920/30fps)")
    ap.add_argument("video")
    ap.add_argument("--traj", required=True, help="trajectory.json (編集タブ形式) または club_track.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="in_frame", type=int, default=0)
    ap.add_argument("--out-frame", type=int, default=-1)
    ap.add_argument("--track", help="club_track.json (骨格線用)")
    ap.add_argument("--realtime", type=float, default=1.0, help="実時間倍率 (59.94p=1, HFR 2x=4)")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--layout", default="auto", choices=["auto", "crop", "pad"])
    ap.add_argument("--audio", default="mute", choices=["mute", "bgm", "source"])
    ap.add_argument("--bgm", default="")
    a = ap.parse_args(argv)

    from ai.store import landmarks_from_track
    traj = json.loads(Path(a.traj).read_text(encoding="utf-8"))
    dense = None
    if "confident_points" in traj:   # club_track.json / *.pred.json
        swings = [TrajectoryData(points=[tuple(p) for p in traj["confident_points"]], thickness=4)]
        landmarks = landmarks_from_track(traj)
        dense = [tuple(p) for p in traj.get("points", [])]
    else:
        swings = _load_swings(a.traj)
        track = json.loads(Path(a.track).read_text(encoding="utf-8")) if a.track else None
        landmarks = landmarks_from_track(track)
        dense = [tuple(p) for p in track.get("points", [])] if track else None
    opt = {"skeleton": not a.no_skeleton, "summary": not a.no_summary, "layout": a.layout,
           "audio_mode": a.audio, "bgm_path": a.bgm}
    rep = render_sns(a.video, a.out, in_frame=a.in_frame, out_frame=a.out_frame, swings=swings,
                     landmarks=landmarks, realtime_factor=a.realtime, options=opt, dense_points=dense,
                     progress=lambda r, m: print(f"  {r * 100:3.0f}% {m}", flush=True))
    print(json.dumps({k: v for k, v in rep.items() if k != "rect"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
