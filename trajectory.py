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


def remap_color_ratio(ratio, pos_start=0.0, pos_end=1.0):
    """色変化位置に応じてratioをリマップ"""
    if pos_start >= pos_end:
        return 0.0 if ratio < pos_start else 1.0
    if ratio <= pos_start:
        return 0.0
    if ratio >= pos_end:
        return 1.0
    return (ratio - pos_start) / (pos_end - pos_start)


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
            self._build_catmull_rom(xs, ys, frames, u_pts)

    def _build_catmull_rom(self, xs, ys, frames, u_pts):
        """Centripetal Catmull-Rom スプライン補間

        ローカル補間のため、ポイント追加/移動の影響が隣接セグメントに限定され、
        グローバルスプライン (splprep) で起きる意図しない膨らみを抑制する。
        alpha=0.5 (centripetal) はカスプや自己交差を保証的に回避する。
        """
        n = len(xs)
        alpha = 0.5  # centripetal
        all_pts = []
        all_frames = []

        for i in range(n - 1):
            # 4点ウィンドウ: P0, P1, P2, P3
            # セグメントは P1→P2 を補間
            i0 = max(i - 1, 0)
            i1 = i
            i2 = i + 1
            i3 = min(i + 2, n - 1)

            p0 = np.array([xs[i0], ys[i0]], dtype=np.float64)
            p1 = np.array([xs[i1], ys[i1]], dtype=np.float64)
            p2 = np.array([xs[i2], ys[i2]], dtype=np.float64)
            p3 = np.array([xs[i3], ys[i3]], dtype=np.float64)

            # Centripetal パラメータ化
            def _knot(pa, pb):
                d = np.linalg.norm(pb - pa)
                return max(d ** alpha, 1e-6)

            t0 = 0.0
            t1 = t0 + _knot(p0, p1)
            t2 = t1 + _knot(p1, p2)
            t3 = t2 + _knot(p2, p3)

            seg_frac = u_pts[i2] - u_pts[i1]
            seg_res = max(int(self.resolution * seg_frac), 4)
            ts = np.linspace(t1, t2, seg_res, endpoint=(i == n - 2))

            for t in ts:
                # De Boor-Cox 式
                a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1 if t1 != t0 else p1
                a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2 if t2 != t1 else p1
                a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3 if t3 != t2 else p2

                b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2 if t2 != t0 else a1
                b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3 if t3 != t1 else a2

                pt = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2 if t2 != t1 else b1

                frac = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
                f = frames[i1] + (frames[i2] - frames[i1]) * frac
                all_pts.append((int(round(pt[0])), int(round(pt[1]))))
                all_frames.append(f)

        self._curve = all_pts
        self._curve_frames = all_frames

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
                # /1.5 にすることでカーブを強調 (標準は /3.0)
                tx, ty = spline_tangents[i]
                scale = seg_frac / 1.5
                p1 = p0 + np.array([tx * scale, ty * scale])

            # 入射制御点 (p3側)
            if h1 is not None:
                p2 = p3 + np.array([h1[0], h1[1]], dtype=float)
            else:
                tx, ty = spline_tangents[i + 1]
                scale = seg_frac / 1.5
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
def _precompute_colors(total, color_start_bgr, color_end_bgr,
                       color_pos_start, color_pos_end):
    """numpy でグラデーション色を一括計算 (Python ループ排除)"""
    cs = np.array(color_start_bgr, dtype=np.float32)
    ce = np.array(color_end_bgr, dtype=np.float32)
    ratios = np.linspace(0, 1, total, dtype=np.float32)
    # remap (pos_end > 1.0 は部分カーブ補正時に発生しうる)
    if color_pos_start != 0.0 or color_pos_end != 1.0:
        span = max(color_pos_end - color_pos_start, 1e-6)
        ratios = np.clip((ratios - color_pos_start) / span, 0.0, 1.0)
    # vectorised lerp → (total, 3) uint8
    colors = (cs[None, :] + (ce - cs)[None, :] * ratios[:, None]).astype(np.int32)
    return colors          # ndarray (total, 3)


def draw_gradient_trail(frame, curve_points, color_start_bgr, color_end_bgr,
                        thickness, alpha=0.85, blur=0,
                        color_pos_start=0.0, color_pos_end=1.0):
    """グラデーション付きスプライン曲線を描画

    Args:
        blur: エッジぼかし量 (0=なし). 大きいほどトレイルが滑らかに減衰する。
    """
    if len(curve_points) < 2 or alpha <= 0.0:
        return

    total = len(curve_points) - 1
    colors = _precompute_colors(total, color_start_bgr, color_end_bgr,
                                color_pos_start, color_pos_end)

    if blur <= 0:
        # ROI限定: フルフレームcopy を回避
        h, w = frame.shape[:2]
        margin = thickness + 2
        pts_arr = np.array(curve_points)
        x_min = max(0, int(pts_arr[:, 0].min()) - margin)
        y_min = max(0, int(pts_arr[:, 1].min()) - margin)
        x_max = min(w, int(pts_arr[:, 0].max()) + margin + 1)
        y_max = min(h, int(pts_arr[:, 1].max()) + margin + 1)

        roi = frame[y_min:y_max, x_min:x_max]
        overlay = roi.copy()
        for i in range(total):
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            p0 = (curve_points[i][0] - x_min, curve_points[i][1] - y_min)
            p1 = (curve_points[i + 1][0] - x_min, curve_points[i + 1][1] - y_min)
            cv2.line(overlay, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        return

    # ぼかしモード: ROI限定で処理 (フル解像度float32を回避)
    h, w = frame.shape[:2]
    k = int(blur) * 2 + 1  # 奇数カーネル
    margin = k + thickness  # ブラー + 線幅分のマージン

    # カーブの座標バウンディングボックスを計算
    pts_arr = np.array(curve_points)
    x_min = max(0, int(pts_arr[:, 0].min()) - margin)
    y_min = max(0, int(pts_arr[:, 1].min()) - margin)
    x_max = min(w, int(pts_arr[:, 0].max()) + margin + 1)
    y_max = min(h, int(pts_arr[:, 1].max()) + margin + 1)
    rw, rh = x_max - x_min, y_max - y_min

    # ROIサイズの小さなキャンバスに描画 (座標をオフセット)
    trail = np.zeros((rh, rw, 3), dtype=np.uint8)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    for i in range(total):
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        p0 = (curve_points[i][0] - x_min, curve_points[i][1] - y_min)
        p1 = (curve_points[i + 1][0] - x_min, curve_points[i + 1][1] - y_min)
        cv2.line(trail, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.line(mask, p0, p1, 255, thickness, cv2.LINE_AA)

    trail = cv2.GaussianBlur(trail, (k, k), 0)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    # uint8演算で合成 (float32一時配列を回避、cv2 C++ SIMD使用)
    mask_a = cv2.convertScaleAbs(mask, alpha=alpha)  # mask * alpha → uint8
    mask3 = cv2.merge([mask_a, mask_a, mask_a])
    inv_mask3 = cv2.bitwise_not(mask3)  # 255 - mask3
    roi = frame[y_min:y_max, x_min:x_max]
    weighted_frame = cv2.multiply(roi, inv_mask3, scale=1.0 / 255.0)
    weighted_trail = cv2.multiply(trail, mask3, scale=1.0 / 255.0)
    cv2.add(weighted_frame, weighted_trail, dst=roi)


def draw_markers(frame, timed_points, color_start_bgr, color_end_bgr, radius=6,
                 color_pos_start=0.0, color_pos_end=1.0):
    """マーカー描画"""
    n = len(timed_points)
    if n <= 1:
        for pt in timed_points:
            cv2.circle(frame, (pt[0], pt[1]), radius, color_start_bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, (pt[0], pt[1]), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        return
    colors = _precompute_colors(n, color_start_bgr, color_end_bgr,
                                color_pos_start, color_pos_end)
    for i, pt in enumerate(timed_points):
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        cv2.circle(frame, (pt[0], pt[1]), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (pt[0], pt[1]), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def render_trajectory_on_frame(frame, swings, current_frame=None,
                               color_pos_start=0.0, color_pos_end=1.0):
    """フレームに全スイングの軌道を描画

    current_frame=None の場合は全体を描画
    current_frame=数値 の場合はフレーム同期で部分描画
    color_pos_start/end はグラデーション位置 (グローバル設定)

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
                # グラデーション位置をフルカーブ基準→可視カーブ基準にリマップ
                vis_cps = color_pos_start / partial_ratio if partial_ratio > 0 else 0.0
                vis_cpe = color_pos_end / partial_ratio if partial_ratio > 0 else 1.0
                draw_gradient_trail(frame, curve, c_start, c_end,
                                    swing.thickness, 0.85,
                                    color_pos_start=vis_cps,
                                    color_pos_end=vis_cpe)
        else:
            curve = spline.get_full_curve()
            if curve and len(curve) >= 2:
                draw_gradient_trail(frame, curve, c_start, c_end,
                                    swing.thickness, 0.85,
                                    color_pos_start=color_pos_start,
                                    color_pos_end=color_pos_end)


def compute_smooth_curve(timed_points, resolution=300, handles=None):
    """全体スプライン曲線を返す"""
    if len(timed_points) < 2:
        return [(p[0], p[1]) for p in timed_points]
    spline = TimedSpline(timed_points, resolution, handles=handles)
    return spline.get_full_curve()
