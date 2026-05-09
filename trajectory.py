"""
軌道描画モジュール

スプライン補間によるクラブヘッド軌道の描画。
golf-swing-trackerの描画エンジンを再利用。
"""

import numpy as np
import cv2
from scipy.interpolate import splprep, splev


# =============================================================================
# 色変換ユーティリティ
# =============================================================================
def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def lerp_color_bgr(c1, c2, ratio):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * ratio) for i in range(3))


# =============================================================================
# スプライン補間（フレーム同期版）
# =============================================================================
class TimedSpline:
    """フレーム番号付きポイントからスプライン曲線を構築
    handles が指定されている場合はキュービックベジェ補間を使用"""

    def __init__(self, timed_points, resolution=300, handles=None):
        self.points = timed_points
        self.resolution = resolution
        self.handles = handles or []
        self._curve = []
        self._curve_frames = []
        self._build()

    def _has_any_handle(self):
        """ハンドルが1つでも設定されているか"""
        for h in self.handles:
            if h is not None:
                return True
        return False

    def _get_handle(self, idx):
        """指定indexのハンドルを返す (None なら None)"""
        if idx < len(self.handles):
            return self.handles[idx]
        return None

    def _compute_spline_tangents(self, xs, ys, u_pts):
        """splprepの微分から各点の接線ベクトルを計算して返す。
        ベジェモード時、ハンドルなしの点に使い曲線形状を保つ。"""
        n = len(xs)
        tangents = [(0.0, 0.0)] * n
        if n == 2:
            dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
            tangents[0] = (dx, dy)
            tangents[1] = (dx, dy)
            return tangents
        try:
            k = min(3, n - 1)
            tck, _ = splprep([xs, ys], u=u_pts, s=0, k=k)
            # 各制御点のu値での微分を取得
            dxs, dys = splev(u_pts, tck, der=1)
            for i in range(n):
                tangents[i] = (float(dxs[i]), float(dys[i]))
        except Exception:
            # フォールバック: Catmull-Rom
            for i in range(n):
                if i == 0:
                    tangents[i] = (float(xs[1] - xs[0]), float(ys[1] - ys[0]))
                elif i == n - 1:
                    tangents[i] = (float(xs[-1] - xs[-2]), float(ys[-1] - ys[-2]))
                else:
                    tangents[i] = ((xs[i+1] - xs[i-1]) * 0.5, (ys[i+1] - ys[i-1]) * 0.5)
        return tangents

    def _build(self):
        n = len(self.points)
        if n == 0:
            return
        if n == 1:
            self._curve = [(self.points[0][0], self.points[0][1])]
            self._curve_frames = [self.points[0][2]]
            return

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        frames = [p[2] for p in self.points]

        dists = [0.0]
        for i in range(1, n):
            d = np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            dists.append(dists[-1] + max(d, 1e-6))
        total_len = dists[-1]
        u_pts = [d / total_len for d in dists]

        if self._has_any_handle():
            # splprepの接線を事前計算し、ハンドルなしの点に使う
            spline_tangents = self._compute_spline_tangents(xs, ys, u_pts)
            self._build_bezier(xs, ys, frames, u_pts, spline_tangents)
        elif n == 2:
            u_new = np.linspace(0, 1, self.resolution)
            cx = xs[0] + (xs[1] - xs[0]) * u_new
            cy = ys[0] + (ys[1] - ys[0]) * u_new
            curve_frames = np.interp(u_new, u_pts, frames)
            self._curve = [(int(round(x)), int(round(y))) for x, y in zip(cx, cy)]
            self._curve_frames = curve_frames.tolist()
        else:
            try:
                k = min(3, n - 1)
                tck, _ = splprep([xs, ys], u=u_pts, s=0, k=k)
                u_new = np.linspace(0, 1, self.resolution)
                cx, cy = splev(u_new, tck)
            except Exception:
                u_new = np.linspace(0, 1, self.resolution)
                cx = np.interp(u_new, u_pts, xs)
                cy = np.interp(u_new, u_pts, ys)
            curve_frames = np.interp(u_new, u_pts, frames)
            self._curve = [(int(round(x)), int(round(y))) for x, y in zip(cx, cy)]
            self._curve_frames = curve_frames.tolist()

    def _build_bezier(self, xs, ys, frames, u_pts, spline_tangents):
        """ベジェ曲線によるセグメント単位の補間
        spline_tangents: splprepから計算した各点の接線 (ハンドルなし点で使用)"""
        n = len(xs)
        all_pts = []
        all_frames = []

        for i in range(n - 1):
            seg_frac = u_pts[i + 1] - u_pts[i]
            seg_res = max(int(self.resolution * seg_frac), 4)

            p0 = np.array([xs[i], ys[i]], dtype=float)
            p3 = np.array([xs[i + 1], ys[i + 1]], dtype=float)

            h0 = self._get_handle(i)
            h1 = self._get_handle(i + 1)

            # 出射制御点 (p0側)
            if h0 is not None:
                p1 = p0 + np.array([h0[2], h0[3]], dtype=float)
            else:
                # splprep接線をセグメント長でスケール → ベジェ制御点
                tx, ty = spline_tangents[i]
                scale = seg_frac / 3.0
                p1 = p0 + np.array([tx * scale, ty * scale])

            # 入射制御点 (p3側)
            if h1 is not None:
                p2 = p3 + np.array([h1[0], h1[1]], dtype=float)
            else:
                tx, ty = spline_tangents[i + 1]
                scale = seg_frac / 3.0
                p2 = p3 - np.array([tx * scale, ty * scale])

            # キュービックベジェ
            ts = np.linspace(0, 1, seg_res, endpoint=(i == n - 2))
            for t in ts:
                mt = 1 - t
                pt = mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3
                f = frames[i] + (frames[i + 1] - frames[i]) * t
                all_pts.append((int(round(pt[0])), int(round(pt[1]))))
                all_frames.append(f)

        self._curve = all_pts
        self._curve_frames = all_frames

    def get_curve_at_frame(self, current_frame):
        """current_frameまでの曲線を返す"""
        if not self._curve:
            return []
        first_frame = self.points[0][2]
        last_frame = self.points[-1][2]
        if current_frame < first_frame:
            return []
        if current_frame >= last_frame:
            return list(self._curve)

        cut_idx = 0
        for i, f in enumerate(self._curve_frames):
            if f <= current_frame:
                cut_idx = i
            else:
                break

        # 先端補間
        if cut_idx < len(self._curve_frames) - 1:
            f0 = self._curve_frames[cut_idx]
            f1 = self._curve_frames[cut_idx + 1]
            if f1 > f0:
                t = min(max((current_frame - f0) / (f1 - f0), 0), 1)
                p0 = self._curve[cut_idx]
                p1 = self._curve[cut_idx + 1]
                head = (int(round(p0[0] + (p1[0] - p0[0]) * t)),
                        int(round(p0[1] + (p1[1] - p0[1]) * t)))
            else:
                head = self._curve[cut_idx]
        else:
            head = self._curve[cut_idx]

        visible = self._curve[:cut_idx + 1]
        if head != visible[-1]:
            visible = visible + [head]
        return visible

    def get_full_curve(self):
        return list(self._curve)


# =============================================================================
# 描画関数
# =============================================================================
def draw_gradient_trail(frame, curve_points, color_start_bgr, color_end_bgr,
                        thickness, alpha=0.85, blur=0):
    """グラデーション付きスプライン曲線を描画

    Args:
        blur: エッジぼかし量 (0=なし). 大きいほどトレイルが滑らかに減衰する。
    """
    if len(curve_points) < 2 or alpha <= 0.0:
        return

    if blur <= 0:
        overlay = frame.copy()
        total = len(curve_points) - 1
        for i in range(total):
            ratio = i / total
            color = lerp_color_bgr(color_start_bgr, color_end_bgr, ratio)
            cv2.line(overlay, curve_points[i], curve_points[i + 1],
                     color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return

    # ぼかしモード: トレイルを黒キャンバスに描画 → ブラー → マスク合成
    h, w = frame.shape[:2]
    trail = np.zeros_like(frame)
    mask = np.zeros((h, w), dtype=np.uint8)
    total = len(curve_points) - 1
    for i in range(total):
        ratio = i / total
        color = lerp_color_bgr(color_start_bgr, color_end_bgr, ratio)
        cv2.line(trail, curve_points[i], curve_points[i + 1],
                 color, thickness, cv2.LINE_AA)
        cv2.line(mask, curve_points[i], curve_points[i + 1],
                 255, thickness, cv2.LINE_AA)

    k = int(blur) * 2 + 1  # 奇数カーネル
    trail = cv2.GaussianBlur(trail, (k, k), 0)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    # マスクを 0-1 に正規化して alpha を掛ける
    mask_f = (mask.astype(np.float32) / 255.0) * alpha
    mask_f = mask_f[:, :, np.newaxis]
    frame_f = frame.astype(np.float32)
    trail_f = trail.astype(np.float32)
    np.copyto(frame, (frame_f * (1.0 - mask_f) + trail_f * mask_f).astype(np.uint8))


def draw_markers(frame, timed_points, color_start_bgr, color_end_bgr, radius=6):
    """マーカー描画"""
    n = len(timed_points)
    for i, pt in enumerate(timed_points):
        ratio = i / max(n - 1, 1)
        color = lerp_color_bgr(color_start_bgr, color_end_bgr, ratio)
        cv2.circle(frame, (pt[0], pt[1]), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (pt[0], pt[1]), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def render_trajectory_on_frame(frame, swings, current_frame=None):
    """フレームに全スイングの軌道を描画

    current_frame=None の場合は全体を描画
    current_frame=数値 の場合はフレーム同期で部分描画

    swing.end_frame >= 0 の場合、current_frame > end_frame では描画しない
    """
    for swing in swings:
        if len(swing.points) < 2:
            continue
        # 終了フレーム以降は描画しない
        end_f = getattr(swing, 'end_frame', -1)
        if current_frame is not None and end_f >= 0 and current_frame > end_f:
            continue

        c_start = hex_to_bgr(swing.color_start_hex)
        c_end = hex_to_bgr(swing.color_end_hex)
        handles = getattr(swing, 'handles', None) or None
        spline = TimedSpline(swing.points, 300, handles=handles)

        if current_frame is not None:
            curve = spline.get_curve_at_frame(current_frame)
            if curve and len(curve) >= 2:
                full_len = len(spline._curve)
                partial_ratio = len(curve) / max(full_len, 1)
                c_partial_end = lerp_color_bgr(c_start, c_end, partial_ratio)
                draw_gradient_trail(frame, curve, c_start, c_partial_end,
                                    swing.thickness, 0.85)
        else:
            curve = spline.get_full_curve()
            if curve and len(curve) >= 2:
                draw_gradient_trail(frame, curve, c_start, c_end,
                                    swing.thickness, 0.85)


def compute_smooth_curve(timed_points, resolution=300, handles=None):
    """全体スプライン曲線を返す"""
    if len(timed_points) < 2:
        return [(p[0], p[1]) for p in timed_points]
    spline = TimedSpline(timed_points, resolution, handles=handles)
    return spline.get_full_curve()
