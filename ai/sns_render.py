"""ai.sns_render — クラブヘッド軌道を主役にした SNS 向け縦型動画 (1080x1920 / 30fps / H.264) の生成。

swingcut/render.py の構成 (通常速 → スロー → サマリーカード) を、ベースの軌道描画
(trajectory.py のスプライン・グラデーション) を主役にして作り直したもの。

構成:
  1. 通常速度 (アドレス少し前 〜 フィニッシュ)。軌道をフレーム同期で伸ばしながら描く
  2. 1/N スロー (既定 1/4)。軌道 + 骨格線 (任意、軌道より細く半透明)。
     トップとインパクトで静止し、撮影方向に応じた補助線と指標を表示
       DTL : 背骨ライン (黄) + 垂直基準線 (白)、前傾角
       正面: 頭の基準位置の縦線 (白) + 軸ライン (黄)、頭の左右ブレ / 軸の傾き
  3. サマリーカード (任意、既定 ON): テンポ・スイング時間・撮影方向別の指標・rules.yaml のコメント

局面:
  - 軌道 (クラブヘッド) 由来: derive_phases()
  - 骨格 (手首 / ボール ROI) 由来: ai.swing_analyzer (骨格があるとき)
  両方あるときは、トップ / インパクトそれぞれで差が実時間 PHASE_TOL_SEC (50 ms) 以内なら軌道側、
  それ以上ずれたら警告をログに出して骨格側を採用する。アドレス / フィニッシュは骨格側。

素材の fps と実時間:
  realtime_factor = 素材 1 フレームが実時間の何フレーム分か。通常 59.94p は 1.0、
  HFR 2x (29.97fps 記録で 1/2 速) は 4.0。出力は常に 30fps 実時間。

使い方 (CLI):
    python -m ai.sns_render <動画> --traj <club_track.json | trajectory.json> --out short.mp4
        [--in N --out-frame N] [--track club_track.json] [--no-skeleton] [--no-summary]
        [--realtime 1.0] [--camera auto|dtl|face_on] [--stills DIR]
"""
from __future__ import annotations

import json
import logging
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

log = logging.getLogger("sns_render")

OUT_W, OUT_H, FPS = 1080, 1920, 30
SAFE_TOP = int(OUT_H * 0.15)      # SNS の UI に被らない領域 (上 15% / 下 20% には文字を置かない)
SAFE_BOTTOM = int(OUT_H * 0.80)
MAX_ZOOM = 1.6
CONTENT_MARGIN = 0.05             # 枠の中で内容の外側に取る余白 (枠の幅・高さに対する比、上下左右)
PAN_TAU_SEC = 0.45                # 動的クロップ (pan) の枠移動の時定数 (大きいほどゆっくり)
PAN_LOOKAHEAD_SEC = 0.35          # pan の目標位置を決めるときに先読みする時間 (ヘッドが枠から出る前に動き出す)
PAN_MAX_SPEED = 0.8               # pan の枠移動の上限 (ソース幅 / 秒)
SPLINE_RES = 600
PHASE_TOL_SEC = 0.05              # 軌道由来と骨格由来の局面差の許容 (実時間 50 ms。59.94p で 3f、HFR 2x で 6f)
AUDIO_SR = 48000
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 0, 11, 12, 23, 24, 27, 28
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]
WHITE, GRAY, YELLOW = (255, 255, 255), (190, 190, 190), (255, 235, 59)


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
        log.warning("フォントがありません: %s (setup_assets.py を実行)。既定フォントで代用", path)
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


def wrap_text(text: str, size: int, max_width: int) -> list:
    font = _font(size)
    lines, cur = [], ""
    for ch in text:
        if font.getlength(cur + ch) > max_width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


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
    impact  = バックスイングで十分遠ざかった後、最初にアドレス位置へ戻った極小点,
    top     = アドレスとインパクトの間で速度が最も落ちる点 (切り返し)。
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
        spline = TimedSpline(pts, SPLINE_RES, handles=getattr(main, "handles", None) or None)
        for f in range(int(np.floor(pts[0][2])), int(np.ceil(pts[-1][2])) + 1):
            curve = spline.get_curve_at_frame(f)
            if curve:
                fs_l.append(f)
                xs_l.append(curve[-1][0])
                ys_l.append(curve[-1][1])
    fs, xs, ys = (np.array(v, dtype=float) for v in (fs_l, xs_l, ys_l))
    n = len(fs)
    fin_i = n - 1
    spd_all = np.hypot(np.gradient(xs), np.gradient(ys)) / np.maximum(np.gradient(fs), 1e-6)
    s_thr = 0.08 * float(spd_all.max()) if n > 2 else 0.0
    onset = next((i for i in range(n) if spd_all[i] > s_thr), 0)
    addr_i = max(0, onset - 1)
    lo = addr_i + max(1, int((n - addr_i) * 0.25))
    dist = np.hypot(xs - xs[addr_i], ys - ys[addr_i])
    dmax = float(dist.max())
    thr = 0.25 * dmax
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
    if imp_i > addr_i + 2:
        lo_t = addr_i + max(1, int((imp_i - addr_i) * 0.4))
        hi_t = max(lo_t + 1, addr_i + int((imp_i - addr_i) * 0.95))
        spd = []
        for i in range(lo_t, hi_t + 1):
            a, b = max(addr_i, i - 1), min(imp_i, i + 1)
            spd.append(np.hypot(xs[b] - xs[a], ys[b] - ys[a]) / max(1e-6, fs[b] - fs[a]))
        top_i = lo_t + int(np.argmin(spd))
    else:
        top_i = min(addr_i + 1, imp_i)
    return {"address": in_frame + int(round(fs[addr_i])), "top": in_frame + int(round(fs[top_i])),
            "impact": in_frame + int(round(fs[imp_i])), "finish": in_frame + int(round(fs[fin_i]))}


def fuse_phases(traj: dict, skel: Optional[dict], warnings: list, real_fps: float = 30.0):
    """軌道由来と骨格由来の局面を統合する (差が実時間 PHASE_TOL_SEC 以内なら軌道側、以上なら骨格側)。"""
    if not skel:
        return dict(traj), {"top": "trajectory", "impact": "trajectory"}
    tol = max(1, int(round(PHASE_TOL_SEC * real_fps)))
    out = {"address": skel["address"], "finish": max(skel["finish"], traj["finish"])}
    src = {}
    for k in ("top", "impact"):
        d = abs(traj[k] - skel[k])
        if d <= tol:
            out[k], src[k] = traj[k], "trajectory"
        else:
            out[k], src[k] = skel[k], "skeleton"
            msg = f"{k}: 軌道由来 {traj[k]} と骨格由来 {skel[k]} が {d} フレーム ({d / real_fps * 1000:.0f} ms, 許容 {tol}f) ずれ → 骨格側を採用"
            log.warning(msg)
            warnings.append(msg)
    if out["top"] >= out["impact"]:
        out["top"] = max(out["address"] + 1, out["impact"] - 1)
    return out, src


# ---------------------------------------------------------------- レイアウト

def swing_curve_points(swings):
    """描かれるスプライン全体の点 (観測 + 推定 + 手直し後の制御点を通る曲線)。制御点だけだと曲線のふくらみ分が外に出る。"""
    pts = []
    for s in swings:
        if len(s.points) >= 2:
            try:
                spl = TimedSpline(s.points, SPLINE_RES, handles=getattr(s, "handles", None) or None)
                pts.extend((float(c[0]), float(c[1])) for c in spl._curve)
                continue
            except Exception:
                pass
        pts.extend((float(p[0]), float(p[1])) for p in s.points)
    return pts


def content_bbox(swings, landmarks, W, H, ball=None, frames=None):
    """枠に収めたい内容の外接矩形 (ソース座標、フレーム外にはみ出していてもそのまま)。
    軌道の全点 (スプライン全体) + 骨格 (frames を渡せばその範囲のフレームだけ) + ボール。"""
    xs, ys = [], []
    for x, y in swing_curve_points(swings):
        xs.append(x)
        ys.append(y)
    for f, lm in (landmarks or {}).items():
        if frames is not None and f not in frames:
            continue
        xs.extend(p[0] for p in lm)
        ys.extend(p[1] for p in lm)
    if ball:
        bx, by, br = ball[0], ball[1], (ball[2] if len(ball) > 2 else 0)
        xs += [bx - br, bx + br]
        ys += [by - br, by + br]
    if not xs:
        return (W * 0.4, H * 0.3, W * 0.6, H * 0.7)
    return (min(xs), min(ys), max(xs), max(ys))


def compute_layout(W, H, bbox, mode="auto"):
    """(kind, rect, fits)。
    kind='crop': rect=(x0, y0, w, h) の固定クロップ / 'pad': 全体を縮小して置く / 'pan': rect=(w, h) の枠を軌道に合わせて動かす。
    fits: 内容 (bbox) に上下左右 CONTENT_MARGIN の余白を付けたものが 9:16 の固定枠 (高さは最大でソース高) に収まるか。
    mode: auto (収まれば crop、収まらなければ pad) / crop (収まらなくても固定枠) / pan (収まらなければ動的クロップ) / pad。"""
    x0, y0, x1, y1 = bbox
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    need_w = bw / (1 - 2 * CONTENT_MARGIN)
    need_h = bh / (1 - 2 * CONTENT_MARGIN)
    crop_h = min(max(need_h, need_w * OUT_H / OUT_W, H / MAX_ZOOM), H)
    crop_w = crop_h * OUT_W / OUT_H
    fits = (need_w <= crop_w + 1e-6) and (need_h <= crop_h + 1e-6) and (crop_w <= W + 1e-6)
    if mode == "pad" or (mode == "auto" and not fits):
        # pad: 内容 (余白込み) がちょうど収まる領域を切り出して 1080 幅 (または 1920 高) に縮小し、ぼかした背景の上に置く。
        # ソース全体を縮小するより被写体が大きく映る
        pw, ph = min(W, max(need_w, crop_w)), min(H, max(need_h, H / MAX_ZOOM))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        px = float(np.clip(cx - pw / 2, 0, W - pw))
        py = float(np.clip(cy - ph / 2, 0, H - ph))
        return "pad", (int(round(px)), int(round(py)), int(round(pw)), int(round(ph))), fits
    if mode == "pan" and not fits:
        return "pan", (int(round(min(crop_w, W))), int(round(crop_h))), fits
    crop_w = min(crop_w, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx = float(np.clip(cx - crop_w / 2, 0, W - crop_w))
    ry = float(np.clip(cy - crop_h / 2, 0, H - crop_h))
    return "crop", (int(round(rx)), int(round(ry)), int(round(crop_w)), int(round(crop_h))), fits


def pan_rects(W, H, size, frames, head_xy, body_xy, real_fps):
    """動的クロップの枠位置 {frame: (x0, y0, w, h)}。
    各フレームの目標 = 先読み窓 (±PAN_LOOKAHEAD_SEC) 内のヘッド位置と体の中心を全部含む範囲の中心。
    それを前後方向の指数平滑 (時定数 PAN_TAU_SEC、位相遅れ無し) で滑らかにし、移動速度を PAN_MAX_SPEED に制限する。"""
    w, h = size
    frames = sorted(frames)
    if not frames:
        return {}
    la = max(1, int(round(PAN_LOOKAHEAD_SEC * real_fps)))
    tx, ty = [], []
    for f in frames:
        xs, ys = [], []
        for g in range(f - la // 3, f + la + 1):
            if g in head_xy:
                xs.append(head_xy[g][0])
                ys.append(head_xy[g][1])
        if f in body_xy:
            xs.append(body_xy[f][0])
            ys.append(body_xy[f][1])
        if not xs:
            xs, ys = [W / 2], [H / 2]
        tx.append((min(xs) + max(xs)) / 2)
        ty.append((min(ys) + max(ys)) / 2)
    tx, ty = np.array(tx, dtype=np.float64), np.array(ty, dtype=np.float64)

    def smooth(v):
        a = 1.0 - np.exp(-1.0 / max(PAN_TAU_SEC * real_fps, 1.0))
        fwd = v.copy()
        for i in range(1, len(v)):
            fwd[i] = fwd[i - 1] + a * (v[i] - fwd[i - 1])
        bwd = v.copy()
        for i in range(len(v) - 2, -1, -1):
            bwd[i] = bwd[i + 1] + a * (v[i] - bwd[i + 1])
        out = (fwd + bwd) / 2
        vmax = PAN_MAX_SPEED * W / max(real_fps, 1.0)
        for i in range(1, len(out)):
            d = out[i] - out[i - 1]
            if abs(d) > vmax:
                out[i] = out[i - 1] + np.sign(d) * vmax
        return out
    cx, cy = smooth(tx), smooth(ty)
    rects = {}
    for i, f in enumerate(frames):
        rx = float(np.clip(cx[i] - w / 2, 0, max(0, W - w)))
        ry = float(np.clip(cy[i] - h / 2, 0, max(0, H - h)))
        rects[f] = (int(round(rx)), int(round(ry)), int(w), int(h))
    return rects


class Layout:
    """ソースフレーム → 1080x1920 の変換。kind='crop'/'pad' は固定、'pan' はフレームごとに枠が動く。"""

    def __init__(self, W, H, kind, rect, rects=None):
        self.W, self.H, self.kind, self.rect = W, H, kind, rect
        self.rects = rects or {}
        # pad: rect の領域 (無ければソース全体) を縦横比を保って収まる大きさに縮小し、中央に置く
        self._pad_rect = tuple(rect) if (kind == "pad" and rect) else (0, 0, W, H)
        px, py, pw, ph = self._pad_rect
        self._pad_s = min(OUT_W / max(pw, 1), OUT_H / max(ph, 1))
        self._pad_fg_w = max(1, int(round(pw * self._pad_s)))
        self._pad_fg_h = max(1, int(round(ph * self._pad_s)))
        self._pad_ox = (OUT_W - self._pad_fg_w) // 2
        self._pad_oy = (OUT_H - self._pad_fg_h) // 2

    def rect_for(self, src):
        """固定クロップ / pan の枠 (x0, y0, w, h)。pad は None (tf / map_pt が別扱い)。"""
        if self.kind == "crop":
            return self.rect
        if self.kind == "pan":
            if src in self.rects:
                return self.rects[src]
            if self.rects:
                near = min(self.rects, key=lambda k: abs(k - src))
                return self.rects[near]
            return (0, 0, self.W, self.H)
        return None

    def tf(self, frame, src=None):
        r = self.rect_for(src)
        if r is not None:
            x0, y0, w, h = r
            return cv2.resize(frame[y0:y0 + h, x0:x0 + w], (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        W, H = self.W, self.H
        sc = max(OUT_W / W, OUT_H / H)
        bw, bh = int(W * sc) + 1, int(H * sc) + 1
        bg = cv2.resize(frame, (bw, bh), interpolation=cv2.INTER_LINEAR)
        bx, by = (bw - OUT_W) // 2, (bh - OUT_H) // 2
        bg = bg[by:by + OUT_H, bx:bx + OUT_W]
        bg = cv2.GaussianBlur(bg, (0, 0), 25)
        bg = cv2.convertScaleAbs(bg, alpha=0.45, beta=0)
        px, py, pw, ph = self._pad_rect
        fg = cv2.resize(frame[py:py + ph, px:px + pw], (self._pad_fg_w, self._pad_fg_h), interpolation=cv2.INTER_LANCZOS4)
        bg[self._pad_oy:self._pad_oy + self._pad_fg_h, self._pad_ox:self._pad_ox + self._pad_fg_w] = fg
        return bg

    def map_pt(self, p, src=None):
        r = self.rect_for(src)
        if r is not None:
            x0, y0, w, h = r
            return (p[0] - x0) * OUT_W / w, (p[1] - y0) * OUT_H / h
        px, py, _, _ = self._pad_rect
        return (p[0] - px) * self._pad_s + self._pad_ox, (p[1] - py) * self._pad_s + self._pad_oy

    def pad_region(self):
        """pad のときに表示されるソース領域 (x0, y0, w, h)。プレビュー用"""
        return self._pad_rect if self.kind == "pad" else None

    def inside(self, p, src=None, tol=2.0):
        x, y = self.map_pt(p, src)
        return -tol <= x < OUT_W + tol and -tol <= y < OUT_H + tol


def make_transform(W, H, kind, rect):
    """後方互換: 固定レイアウトの (tf, map_pt)。"""
    lay = Layout(W, H, kind, rect)
    return (lambda frame: lay.tf(frame)), (lambda p: lay.map_pt(p))


def plan_layout(W, H, swings, landmarks, mode, *, ball=None, frames=None, real_fps=30.0, dense_points=None):
    """内容の外接矩形 → レイアウト種別 → (Layout, info)。pan のときはフレームごとの枠も作る。
    frames: 書き出しに使うソースフレーム (相対) の集合。骨格と pan の対象をこれに限る。"""
    bbox = content_bbox(swings, landmarks, W, H, ball=ball, frames=frames)
    kind, rect, fits = compute_layout(W, H, bbox, mode)
    rects = None
    if kind == "pan":
        head_xy = {}
        if dense_points:
            for x, y, f in dense_points:
                head_xy[int(f)] = (float(x), float(y))
        else:
            for s in swings:
                if len(s.points) >= 2:
                    spl = TimedSpline(s.points, SPLINE_RES, handles=getattr(s, "handles", None) or None)
                    for c, cf in zip(spl._curve, getattr(spl, "_curve_frames", [])):
                        head_xy[int(round(cf))] = (float(c[0]), float(c[1]))
                for p in s.points:
                    head_xy.setdefault(int(p[2]), (float(p[0]), float(p[1])))
        body_xy = {}
        for f, lm in (landmarks or {}).items():
            if len(lm) > 24:
                body_xy[f] = ((lm[23][0] + lm[24][0]) / 2, (lm[11][1] + lm[12][1] + lm[23][1] + lm[24][1]) / 4)
        fr = sorted(frames) if frames else sorted(set(head_xy) | set(body_xy))
        rects = pan_rects(W, H, rect, fr, head_xy, body_xy, real_fps)
    info = {"bbox": [round(v, 1) for v in bbox], "fits": fits, "kind": kind, "rect": rect, "mode": mode}
    return Layout(W, H, kind, rect, rects), info


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

    @property
    def thickness(self):
        return max((it["thick"] for it in self.items), default=3)

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


def draw_skeleton(frame, lm, thickness, alpha=0.45):
    """骨格線を軌道より細く・半透明で描く (ソース座標)。"""
    overlay = frame.copy()
    for a, b in SKELETON:
        pa, pb = lm[a], lm[b]
        cv2.line(overlay, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 200, 0), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_guides(big, tracks, idx, angle, map_pt, head_ref_x=None):
    """静止中の補助線 (出力座標)。DTL: 背骨ライン + 垂直基準線。正面: 頭の基準縦線 + 軸ライン。"""
    hip = (tracks[L_HIP][idx] + tracks[R_HIP][idx]) / 2
    sh = (tracks[L_SHOULDER][idx] + tracks[R_SHOULDER][idx]) / 2
    ankle_y = (tracks[L_ANKLE][idx][1] + tracks[R_ANKLE][idx][1]) / 2
    p1 = map_pt(hip + 1.5 * (sh - hip))
    p2 = map_pt(hip - 0.3 * (sh - hip))
    hx = map_pt(hip)[0]
    y_top = int(min(p1[1], p2[1]) - 40)
    y_bot = int(min(OUT_H - 1, map_pt((hip[0], ankle_y))[1] + 30))
    if angle == "dtl":
        cv2.line(big, (int(hx), y_top), (int(hx), y_bot), (230, 230, 230), 3, cv2.LINE_AA)
    elif head_ref_x is not None:
        rx = int(map_pt((head_ref_x, 0))[0])
        cv2.line(big, (rx, y_top - 80), (rx, y_bot), (230, 230, 230), 3, cv2.LINE_AA)
    cv2.line(big, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 5, cv2.LINE_AA)
    return big


def phase_metric_text(analysis, phase_key, angle):
    """静止中に局面名の下へ出す指標 1 行。"""
    m = analysis["metrics"] if analysis else None
    if not m:
        return None
    if angle == "dtl" and "spine_angle_deg" in m:
        return f"前傾 {m['spine_angle_deg'][phase_key]:.1f}°"
    if angle == "face_on" and "head_sway_pct" in m:
        if phase_key == "top":
            v = m["head_sway_pct"]["top_back_pct"]
            return f"頭 {abs(v):.1f}% {'後ろ' if v >= 0 else 'ターゲット側'}"
        v = m["axis_tilt_deg"]["impact"]
        return f"軸 {abs(v):.1f}° {'後ろ' if v >= 0 else 'ターゲット側'}"
    return None


def legend_items(angle):
    if angle == "dtl":
        return [("── 背骨ライン", (90, 690), 28, "lm", YELLOW),
                ("── 垂直基準線 (前傾角の基準)", (360, 690), 28, "lm", (235, 235, 235))]
    if angle == "face_on":
        return [("── 軸ライン", (90, 690), 28, "lm", YELLOW),
                ("── 頭の基準位置", (330, 690), 28, "lm", (235, 235, 235))]
    return []


# ---------------------------------------------------------------- サマリーカード

def summary_items(analysis, angle, back_s, down_s, tempo, swing_s, comments):
    """文字が下部 20% (SAFE_BOTTOM) に入らないように段組みする。"""
    cx = OUT_W // 2
    items = [("スイング解析", (cx, SAFE_TOP + 72), 76, "mm", WHITE),
             ("テンポ (バックスイング : ダウンスイング)", (cx, 500), 38, "mm", GRAY),
             (f"{back_s:.2f} s : {down_s:.2f} s   =  {tempo:.2f} : 1" if tempo else f"{back_s:.2f} s : {down_s:.2f} s",
              (cx, 566), 54, "mm", WHITE),
             ("スイング時間 (アドレス → フィニッシュ)", (cx, 660), 38, "mm", GRAY),
             (f"{swing_s:.2f} s", (cx, 722), 54, "mm", WHITE)]
    y = 820
    m = analysis["metrics"] if analysis else None
    if m and angle == "dtl" and "spine_angle_deg" in m:
        sp, hd = m["spine_angle_deg"], m["head_movement_pct"]
        v, fw = hd["vertical_pct"], hd["forward_pct"]
        rows = [("前傾角 (アドレス → トップ → インパクト)",
                 f"{sp['address']:.1f}°  →  {sp['top']:.1f}°  →  {sp['impact']:.1f}°"),
                ("頭の移動 (アドレス → インパクト・身長比)",
                 f"{'下' if v >= 0 else '上'} {abs(v):.1f}%   /   {'ボール側' if fw >= 0 else 'ボール逆側'} {abs(fw):.1f}%"),
                ("腰の突っ込み (ボール側・身長比)", f"{m['early_extension_pct']:+.1f}%")]
    elif m and angle == "face_on" and "head_sway_pct" in m:
        hs, ax = m["head_sway_pct"], m["axis_tilt_deg"]
        rows = [("頭の左右ブレ (身長比)",
                 f"トップ 後ろ {hs['top_back_pct']:+.1f}%   /   インパクト ターゲット側 {hs['impact_target_pct']:+.1f}%"),
                ("腰の左右移動 (アドレス → インパクト)", f"ターゲット側 {m['hip_shift_pct']:+.1f}%"),
                ("インパクト時の軸の傾き", f"{'後ろ' if ax['impact'] >= 0 else 'ターゲット側'} {abs(ax['impact']):.1f}°")]
    else:
        rows = []
        if analysis and analysis.get("camera", {}).get("angle") == "unknown":
            rows = [("撮影方向", "斜めのため指標は省略 (局面とテンポのみ)")]
    for label, value in rows:
        items.append((label, (cx, y), 34, "mm", GRAY))
        items.append((value, (cx, y + 52), 44, "mm", WHITE))
        y += 120
    y += 10
    size, line_h, max_w = 44, 58, OUT_W - 160
    for c in comments or []:
        lines = wrap_text("・" + c, size, max_w)
        if y + line_h * len(lines) > SAFE_BOTTOM - 20:
            break   # 下部 20% に入るコメントは丸ごと省く
        for ln in lines:
            items.append((ln, (80, y), size, "lm", WHITE))
            y += line_h
        y += 14
    return items


# ---------------------------------------------------------------- フレーム供給

class FrameReader:
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
               camera_angle: str = "auto", dense_points: Optional[list] = None,
               stills_dir=None, trajectory_note: Optional[str] = None,
               progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """縦型動画を書き出す。swings / landmarks / dense_points のフレーム番号は in_frame 基準の相対値。"""
    opt = dict(AI_DEFAULTS["sns"])
    opt.update(options or {})
    t_start = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stills_dir = Path(stills_dir) if stills_dir else out_path.parent / "preview"
    stills_dir.mkdir(parents=True, exist_ok=True)
    warnings: list = []

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
    landmarks = landmarks or {}
    if not swings and not landmarks:
        raise ValueError("軌道も骨格もありません")
    # 軌道の注記 (「信頼度不足」「一部区間のみ」) は動画に入れない。描ける区間があれば描き、無ければ何も描かない。
    # 状態は編集タブの右パネルにだけ出す (trajectory_note は互換のため受け取るが使わない)
    trajectory_note = None

    # --- 局面: 軌道由来 + 骨格由来 → 統合
    report(0.02, "局面を解析")
    traj_phases = derive_phases(swings, in_frame, dense_points) if (swings or dense_points) else None
    analysis = None
    skel_phases = None
    if landmarks:
        try:
            from ai.swing_analyzer import analyze
            analysis = analyze(landmarks, W, H, fps, real_fps, video_path=video_path, in_frame=in_frame,
                               camera_angle=camera_angle, dense_points=dense_points)
            skel_phases = {k: in_frame + v for k, v in analysis["phases"].items()}
            warnings.extend(analysis["warnings"])
            log.info("骨格由来の局面: %s (インパクト: %s)", skel_phases, analysis["impact"]["chosen"])
        except Exception as e:
            log.warning("骨格解析をスキップ: %s", e)
            warnings.append(f"骨格解析をスキップ: {e}")
    log.info("軌道由来の局面: %s", traj_phases)
    if phases is None:
        if traj_phases is None and skel_phases is None:
            raise ValueError("局面を決められません (軌道も骨格解析も無い)")
        if traj_phases is None:
            phases, phase_src = dict(skel_phases), {"top": "skeleton", "impact": "skeleton"}
        else:
            phases, phase_src = fuse_phases(traj_phases, skel_phases, warnings, real_fps)
    else:
        phase_src = {"top": "given", "impact": "given"}
    addr, top, imp, fin = phases["address"], phases["top"], phases["impact"], phases["finish"]
    addr, fin = max(in_frame, addr), min(out_frame, fin)
    angle = analysis["camera"]["angle"] if analysis else "unknown"
    tracks = analysis["tracks"] if analysis else None
    frame_to_idx = {f: i for i, f in enumerate(analysis["frames"])} if analysis else {}
    head_ref_x = None
    if tracks is not None and angle == "face_on":
        head_ref_x = float(tracks[NOSE][frame_to_idx.get(addr - in_frame, 0)][0])

    drawer = SwingDrawer(swings)
    skel_thick = max(1, drawer.thickness - 1)
    reader = FrameReader(video_path)
    slow = max(1, int(opt.get("slow_factor", 4)))
    freeze_n = int(round(float(opt.get("freeze_sec", 0.7)) * FPS))
    summary_n = int(round(float(opt.get("summary_sec", 3.0)) * FPS)) if opt.get("summary", True) else 0
    use_skel = bool(opt.get("skeleton", True)) and bool(landmarks)
    pre = int(round(float(opt.get("pre_frames", 6)) * real_fps / FPS))
    s1_start = max(in_frame, addr - pre)

    # --- タイムライン (出力フレーム → ソースフレーム)
    plan = []
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
        for key, name in ((top, "top"), (imp, "impact")):
            if src >= key and key not in frozen:
                frozen.add(key)
                plan.extend([(key, "freeze", name)] * freeze_n)
        k += 1
    total_frames = len(plan) + summary_n
    total_sec = total_frames / FPS

    # --- レイアウト: 軌道の全点 (スプライン全体) + 書き出し範囲の骨格 + ボール の外接矩形から決める
    used_frames = {src - in_frame for src, _, _ in plan}
    ball_xy = None
    try:
        b = (analysis or {}).get("impact", {}).get("ball")
        if b:
            ball_xy = (float(b[0]), float(b[1]), float(b[2]) if len(b) > 2 else 0.0)
    except Exception:
        ball_xy = None
    layout, layout_info = plan_layout(W, H, swings, landmarks, opt.get("layout", "auto"), ball=ball_xy,
                                      frames=used_frames, real_fps=real_fps, dense_points=dense_points)
    kind, rect = layout.kind, layout.rect
    log.info("layout: %s", layout_info)
    if kind == "crop" and not layout_info["fits"]:
        warnings.append("9:16 の固定枠に収まらない内容を crop 設定で書き出しています (はみ出します)")

    back_s = max(0.0, (top - addr) / real_fps)
    down_s = max(0.0, (imp - top) / real_fps)
    tempo = back_s / down_s if down_s > 0 else None
    comments = analysis["comments"] if analysis else []

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
    stills = {}
    names = {"top": "トップ", "impact": "インパクト"}
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logf,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def emit(img):
            nonlocal n_out
            proc.stdin.write(img.tobytes())
            n_out += 1

        try:
            last_key, cache = None, None
            for i, (src, section, label) in enumerate(plan):
                key = (src, section, label)
                if key != last_key:
                    fr = reader.get(src)
                    if fr is None:
                        break
                    fr = fr.copy()
                    rel = src - in_frame
                    if section in ("slow", "freeze") and use_skel and rel in landmarks:
                        draw_skeleton(fr, landmarks[rel], skel_thick)
                    drawer.draw(fr, rel)
                    big = layout.tf(fr, rel)
                    note_items = [(trajectory_note, (60, SAFE_TOP + 60), 40, "lm", (255, 160, 120))] if trajectory_note else []
                    if section == "normal" and note_items:
                        big = draw_texts(big, note_items)
                    if section == "slow":
                        big = draw_texts(big, [(f"×1/{slow} スロー", (OUT_W - 60, SAFE_TOP + 60), 56, "rm", WHITE),
                                               *note_items, *legend_items(angle if use_skel else "")])
                    elif section == "freeze":
                        if tracks is not None and angle in ("dtl", "face_on"):
                            big = draw_guides(big, tracks, frame_to_idx.get(rel, 0), angle,
                                              lambda p, _r=rel: layout.map_pt(p, _r), head_ref_x)
                        big = dark_band(big, SAFE_TOP + 110, SAFE_TOP + 300, 0.5)
                        txt = [(names[label], (OUT_W // 2, SAFE_TOP + 170), 72, "mm", WHITE)]
                        mt = phase_metric_text(analysis, label, angle)
                        if mt:
                            txt.append((mt, (OUT_W // 2, SAFE_TOP + 250), 50, "mm", WHITE))
                        big = draw_texts(big, txt + legend_items(angle if use_skel else ""))
                        if label not in stills:
                            stills[label] = str(stills_dir / f"{label}_freeze.png")
                            cv2.imwrite(stills[label], big)
                    cache, last_key = big, key
                emit(cache)
                if i % 30 == 0:
                    report(0.05 + 0.85 * i / max(1, total_frames), "縦型動画を書き出し")

            if summary_n:
                base = reader.get(addr)
                card = cv2.convertScaleAbs(layout.tf(base.copy(), addr - in_frame), alpha=0.30, beta=0)
                card = draw_texts(card, summary_items(analysis, angle, back_s, down_s, tempo,
                                                      (fin - addr) / real_fps, comments))
                stills["summary"] = str(stills_dir / "summary_card.png")
                cv2.imwrite(stills["summary"], card)
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
    # --- はみ出し検査: 各ソースフレームで描かれている軌道 (そのフレームまでの曲線) が出力枠に入っているか
    # 「軌跡 (そのフレームまでの曲線全体)」と「ヘッド (曲線の先端)」を分けて数える。pan は枠が動くので軌跡の後ろ側が
    # 枠から出るのは仕様。ヘッドが出ていれば layout によらず警告する。
    outside_frames, outside_pts, checked, head_out = 0, 0, 0, 0
    for src in sorted(used_frames):
        bad = 0
        for it in drawer.items:
            curve = it["spline"].get_curve_at_frame(src) or []
            for c in curve:
                checked += 1
                if not layout.inside(c, src):
                    bad += 1
            if curve and not layout.inside(curve[-1], src):
                head_out += 1
        if bad:
            outside_frames += 1
            outside_pts += bad
    rep_outside = {"frames": outside_frames, "points": outside_pts, "checked": checked, "head_frames": head_out}
    if head_out or (outside_pts and kind != "pan"):
        msg = (f"軌道が出力枠からはみ出しています: 軌跡 {outside_frames} フレーム / {outside_pts} 点, "
               f"ヘッド {head_out} フレーム (layout={kind})")
        log.warning(msg)
        warnings.append(msg)
    elif outside_pts:
        log.info("pan: 枠が動くため軌跡の後ろ側が %d フレームで枠外 (ヘッドは常に枠内)", outside_frames)
    report(1.0, "完了")
    rep = {"out_path": str(out_path), "frames": n_out, "seconds": time.time() - t_start,
           "layout": kind, "rect": rect, "layout_info": layout_info, "trajectory_outside": rep_outside,
           "phases": phases, "phase_source": phase_src,
           "trajectory_phases": traj_phases, "skeleton_phases": skel_phases, "real_fps": real_fps,
           "skeleton": use_skel, "summary": bool(summary_n), "tempo": tempo,
           "camera": analysis["camera"] if analysis else None,
           "metrics": analysis["metrics"] if analysis else None,
           "comments": comments, "warnings": warnings, "stills": stills,
           "trajectory_note": trajectory_note, "trajectory_drawn": bool(swings)}
    out_path.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
    return rep


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
    ap.add_argument("--traj", required=True, help="club_track.json または trajectory.json (編集タブ形式)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="in_frame", type=int, default=0)
    ap.add_argument("--out-frame", type=int, default=-1)
    ap.add_argument("--track", help="trajectory.json を使うときの club_track.json (骨格・全フレーム点)")
    ap.add_argument("--realtime", type=float, default=1.0, help="実時間倍率 (59.94p=1, HFR 2x=4)")
    ap.add_argument("--camera", default="auto", choices=["auto", "dtl", "face_on"])
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--layout", default="auto", choices=["auto", "crop", "pan", "pad"])
    ap.add_argument("--audio", default="mute", choices=["mute", "bgm", "source"])
    ap.add_argument("--bgm", default="")
    ap.add_argument("--stills", default=None, help="静止画 (top/impact/summary) の出力先")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    from ai.store import landmarks_from_track
    traj = json.loads(Path(a.traj).read_text(encoding="utf-8"))
    if "confident_points" in traj:
        swings = [TrajectoryData(points=[tuple(p) for p in traj["confident_points"]], thickness=4)]
        track = traj
    else:
        swings = _load_swings(a.traj)
        track = json.loads(Path(a.track).read_text(encoding="utf-8")) if a.track else None
    landmarks = landmarks_from_track(track)
    dense = [tuple(p) for p in track.get("points", [])] if track else None
    opt = {"skeleton": not a.no_skeleton, "summary": not a.no_summary, "layout": a.layout,
           "audio_mode": a.audio, "bgm_path": a.bgm}
    rep = render_sns(a.video, a.out, in_frame=a.in_frame, out_frame=a.out_frame, swings=swings,
                     landmarks=landmarks, realtime_factor=a.realtime, options=opt, camera_angle=a.camera,
                     dense_points=dense, stills_dir=a.stills,
                     progress=lambda r, m: print(f"  {r * 100:3.0f}% {m}", flush=True))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rect", "metrics")}, ensure_ascii=False,
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
