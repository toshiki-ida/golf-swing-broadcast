"""club_tracker — 骨格 (MediaPipe Pose) を基点にしたクラブヘッド追跡 (試作)。

torch / ultralytics は使わない。依存は mediapipe + opencv + numpy (+ scipy があれば補間に使う)。

考え方
------
1. MediaPipe Pose で毎フレームの手首位置 (両手首の中点) を得る。
2. 動く物だけを見る: 3 フレーム差分  motion = min(|f_t - f_{t-k}|, |f_t - f_{t+k}|)
   (BGR 最大チャネル差)。前後両方と違う画素 = 今まさにそこにある動く物。
   背景の中央値は使わない (静止姿勢が長いクリップでは中央値が人物ごと背景になるため)。
   人物自身は骨格から作った太い線のマスクで除く。
3. 探索は手首中点を中心に、半径 R_min〜R_max の円環内だけ。
     R_max = 1.0 × 身長代替値 (鼻〜足首中点距離)、R_min = 0.22 × 同
   クラブはその外に出られないので、ギャラリーや背景の動きの大半を最初から見ない。
4. 円環内の連結成分ごとに「手首から最も遠い画素」までの距離を求め、
   手に繋がっている (シャフトが写っている) 成分と、前フレームからの予測位置に近い成分を優先する。
   ヘッド位置 = 選んだ成分の最遠端付近の重心。ダウンスイングで残像の筋になっても
   「筋の先端」がそのままヘッドになる。
5. 動きが無いフレーム (アドレス・トップ・フィニッシュ) ではヘッドは動いていないので、
   直近の観測位置を保持する。短い欠測は PCHIP で埋め、信頼度 0 として返す
   (編集タブでは確信点だけを制御点にする)。

インターレース (1080i59.94) は interlaced=True でフィールド単位に処理し、
フレーム番号は f と f+0.5 (119.88fps 相当) で返す。ベースの HFR 録画は
既にフィールド展開済みなので interlaced=False のまま各フレームを処理すればよい。

出力はベースの TrajectoryData 互換 [(x, y, frame), ...] (元解像度のピクセル)。

使い方:
    from ai.club_tracker import track_club
    res = track_club("clip.mp4", progress=lambda r, m: print(r, m))
    res.points              # 全フレーム (補間含む)
    res.confident_points()  # 編集タブ用の制御点
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# MediaPipe Pose 33 点のうち使うもの
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32
BODY_LINES = [
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER), (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]

ANALYSIS_MAX_W = 1280       # 解析解像度 (長辺がこれ以下になるよう縮小)。960 だと 1080p の細いシャフトが消える
MOTION_GAP = 2              # 3 フレーム差分の間隔 (フィールド処理時は 2 倍にする)
MOTION_GAP_SLOW = 6         # 動きが小さく検出できない時に広げる間隔
MOTION_THR = 22             # 動き差分のしきい値 (BGR 最大チャネル差)
R_MIN_RATIO = 0.10          # 探索円環の内径 (身長代替値に対する比)。DTL のトップではヘッドが手のすぐ上に来る
R_MAX_RATIO = 1.15          # 探索円環の外径 (腕を伸ばした状態のクラブ長に余裕を持たせる)
BODY_WIDTH_RATIO = 0.17     # 人物マスクの線の太さ (服の縁の動きまで覆う)
HAND_RATIO = 0.08           # 手首まわりの除外半径 (手+グリップ)
HEAD_EXCL_RATIO = 0.20      # 頭部 (鼻中心) の除外半径。キャップ・髪・襟の動きを拾わないため
HEAD_RATIO = 0.045          # ヘッド径 (先端重心を取る範囲)
ANGLE_BIN = 3               # 角度ヒストグラムの刻み (度)
WEDGE_DEG = 5               # ヘッド抽出に使う扇の半角 (度)
ANGLE_GATE = 30             # 前フレームからの角度変化の許容 (度/フレーム, 予測込み)
MIN_PX_RATIO = 0.15         # 最小動き画素数 (ヘッド径^2 に対する比)
MISS_RESET = 3              # 連続でこの回数見失ったら追跡状態を捨てて再取得する
GROUND_MARGIN = 0.04        # 足の最下点からこの分より下は地面 (影) として除外
CONF_OK = 1.5               # これ以上を「確信点」とする
WRIST_MIN_VIS = 0.5         # これ未満の visibility の手首は欠測として補間
WRIST_MAX_JUMP = 0.33       # 手首が 1 フレームでこれ (身長代替値比) 以上跳んだら欠測扱い
HOLD_CONF = 0.8             # 静止区間で保持した位置の信頼度
BODY_H_WINDOW = 0.4         # 身長代替値を決めるのに使うクリップ前半の割合 (アドレス付近。終盤のズームを除外)


@dataclass
class ClubTrackResult:
    points: list                      # [(x, y, frame), ...] 全フレーム (補間含む)
    conf: dict                        # {frame: 信頼度}  0 = 補間, HOLD_CONF = 静止保持
    wrist: dict = field(default_factory=dict)   # {frame: (x, y)} 手首中点 (元解像度)
    body_h: float = 0.0               # 身長代替値 (元解像度 px)
    seconds: float = 0.0
    pose_seconds: float = 0.0
    scale: float = 1.0                # 解析解像度 / 元解像度
    frames_total: int = 0
    frames_detected: int = 0

    def confident_points(self, min_conf: float = CONF_OK, max_pts: int = 18,
                         min_gap: float = 2.0) -> list:
        """確信点だけを、編集タブの制御点として使える点数に間引いて返す。"""
        good = [p for p in self.points if self.conf.get(p[2], 0.0) >= min_conf]
        # 先頭/末尾の静止保持点 (アドレス・フィニッシュ) は位置として正しいので制御点に含める
        if self.points and self.conf.get(self.points[0][2], 0.0) == HOLD_CONF and                 (not good or good[0][2] != self.points[0][2]):
            good.insert(0, self.points[0])
        if self.points and self.conf.get(self.points[-1][2], 0.0) == HOLD_CONF and                 (not good or good[-1][2] != self.points[-1][2]):
            good.append(self.points[-1])
        if len(good) < 2:
            return list(self.points[:: max(1, len(self.points) // max_pts)])
        # 信頼度の高い点から貪欲に採用し (min_gap 以内の近接点は捨てる)、最後にフレーム順に並べる。
        # 先頭・末尾は必ず残す
        ends = [good[0], good[-1]]
        chosen = list(ends)
        for p in sorted(good[1:-1], key=lambda q: -self.conf.get(q[2], 0.0)):
            if len(chosen) >= max_pts:
                break
            if all(abs(p[2] - q[2]) >= min_gap for q in chosen):
                chosen.append(p)
        return sorted(chosen, key=lambda q: q[2])

    def weak_ranges(self, min_conf: float = CONF_OK, min_len: int = 3) -> list:
        """観測できなかった連続区間 [(開始, 終了), ...] (オペレーターの手直し箇所)。"""
        out, run = [], []
        for _, _, f in self.points:
            if self.conf.get(f, 0.0) < min_conf:
                run.append(f)
            else:
                if len(run) >= min_len:
                    out.append((run[0], run[-1]))
                run = []
        if len(run) >= min_len:
            out.append((run[0], run[-1]))
        return out


# ---------------------------------------------------------------- フレーム供給

def _iter_frames(video_path: str, in_frame: int, out_frame: int,
                 interlaced: bool, upper_field_first: bool, size: tuple):
    """(frame_no, bgr_small) を順に yield。interlaced なら 1 フレームから 2 フィールドを作る。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if out_frame < 0 or out_frame >= total:
        out_frame = total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, in_frame)
    aw, ah = size
    try:
        for f in range(in_frame, out_frame + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            if interlaced:
                from field_processor import build_high_frame_rate_frames
                a, b = build_high_frame_rate_frames(bgr, upper_field_first)
                yield float(f), cv2.resize(a, (aw, ah), interpolation=cv2.INTER_AREA)
                yield f + 0.5, cv2.resize(b, (aw, ah), interpolation=cv2.INTER_AREA)
            else:
                if bgr.shape[1] != aw or bgr.shape[0] != ah:
                    bgr = cv2.resize(bgr, (aw, ah), interpolation=cv2.INTER_AREA)
                yield f, bgr
    finally:
        cap.release()


def _probe(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return w, h, n, fps


# ---------------------------------------------------------------- 骨格

def _make_landmarker(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルがありません: {model_path} (setup_assets.py を実行)")
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_tracking_confidence=0.3)
    return mp, vision.PoseLandmarker.create_from_options(opts)


def _fill_missing(arr: np.ndarray) -> np.ndarray:
    """(T, K, 2) の NaN を時間方向に線形補間。"""
    T = arr.shape[0]
    t = np.arange(T)
    out = arr.copy()
    for k in range(arr.shape[1]):
        for c in range(2):
            col = out[:, k, c]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                col[:] = 0
            elif ok.sum() < T:
                col[~ok] = np.interp(t[~ok], t[ok], col[ok])
    return out




def _estimate_body_h(arr: np.ndarray) -> float:
    """身長代替値 (鼻〜足首中点の距離, 解析解像度 px) をアドレス付近から決める。

    クリップ前半 (BODY_H_WINDOW) の値だけを使い、中央値フィルタで単発の外れ値を均したうえで
    90 パーセンタイルを取る (前傾で縮む分の補正)。終盤に選手の顔へズームする素材で、
    拡大後のフレームが身長代替値を押し上げて探索円環・人物マスク・ヘッド径が膨らむのを防ぐ。
    """
    from scipy.signal import medfilt
    T = arr.shape[0]
    dist = np.linalg.norm(arr[:, NOSE] - (arr[:, L_ANKLE] + arr[:, R_ANKLE]) / 2, axis=1)
    dist = np.nan_to_num(dist, nan=float(np.nanmedian(dist)) if np.isfinite(dist).any() else 100.0)
    k = max(3, (T // 20) | 1)
    if k < T:
        dist = medfilt(dist, kernel_size=k)
    head = dist[: max(5, int(T * BODY_H_WINDOW))]
    return max(float(np.percentile(head, 90)), 40.0)


# ---------------------------------------------------------------- 本体

def track_club(video_path, in_frame: int = 0, out_frame: int = -1, *,
               interlaced: bool = False, upper_field_first: bool = True,
               model_path: Optional[Path] = None,
               progress: Optional[Callable[[float, str], None]] = None,
               debug_dir: Optional[Path] = None) -> ClubTrackResult:
    """クラブヘッド軌道を推定する。座標は元解像度のピクセル。"""
    from ai import default_pose_model
    t_start = time.time()
    model_path = Path(model_path) if model_path else default_pose_model()
    W, H, n_total, _fps = _probe(video_path)
    scale = min(1.0, ANALYSIS_MAX_W / max(W, H))
    aw, ah = int(round(W * scale)), int(round(H * scale))
    gap = MOTION_GAP * (2 if interlaced else 1)
    gap_slow = MOTION_GAP_SLOW * (2 if interlaced else 1)

    def report(r, m):
        if progress:
            progress(r, m)

    # ---- パス 1: 骨格推定
    report(0.02, "骨格推定")
    mp, landmarker = _make_landmarker(model_path)
    frame_ids: list = []
    lms: list = []
    t_pose = 0.0
    try:
        for i, (fno, small) in enumerate(_iter_frames(video_path, in_frame, out_frame,
                                                      interlaced, upper_field_first, (aw, ah))):
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / 120))
            t_pose += time.time() - t0
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                lms.append(np.array([(p.x * aw, p.y * ah, p.visibility) for p in lm], dtype=np.float32))
            else:
                lms.append(None)
            frame_ids.append(fno)
            if i % 20 == 0:
                report(0.02 + 0.45 * min(1.0, i / max(1, n_total * (2 if interlaced else 1))),
                       "骨格推定")
    finally:
        landmarker.close()

    T = len(frame_ids)
    if T < 2 * gap_slow + 2:
        raise ValueError("フレームが足りません")

    # ---- 骨格の整形: 欠測補間、身長代替値、手首中点
    arr = np.full((T, 33, 2), np.nan, dtype=np.float32)
    detected = 0
    for i, lm in enumerate(lms):
        if lm is not None:
            arr[i] = lm[:, :2]
            # 速い動きで手首の推定が飛ぶことがある: visibility が低い手首は欠測にして補間する
            for w in (L_WRIST, R_WRIST, L_ELBOW, R_ELBOW):
                if lm[w, 2] < WRIST_MIN_VIS:
                    arr[i, w] = np.nan
            detected += 1
    if detected < 4:
        raise ValueError("骨格を検出できませんでした")
    arr = _fill_missing(arr)
    body_h = _estimate_body_h(arr)
    # 手首の跳び (1 フレームで身長代替値の 1/3 以上) も欠測にして補間し直す
    for w in (L_WRIST, R_WRIST):
        step = np.linalg.norm(np.diff(arr[:, w], axis=0), axis=1)
        bad = np.flatnonzero(step > WRIST_MAX_JUMP * body_h) + 1
        arr[bad, w] = np.nan
    arr = _fill_missing(arr)
    wrist = (arr[:, L_WRIST] + arr[:, R_WRIST]) / 2
    r_min, r_max = R_MIN_RATIO * body_h, R_MAX_RATIO * body_h
    body_w = max(3, int(BODY_WIDTH_RATIO * body_h))
    hand_r = max(4, int(HAND_RATIO * body_h))
    head_d = max(3.0, HEAD_RATIO * body_h)
    head_excl = max(6, int(HEAD_EXCL_RATIO * body_h))
    min_px = max(12.0, MIN_PX_RATIO * head_d ** 2)   # これ未満の動き画素は無視 (フレア等)
    foot_y = np.max(arr[:, [L_HEEL, R_HEEL, L_FOOT, R_FOOT], 1], axis=1)   # 足の最下点 (地面)

    # ---- パス 2: 3 フレーム差分 + 円環内の角度ヒストグラム探索 (動画をもう一度読む)
    report(0.5, "ヘッド探索")
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    raw: dict = {}                 # i -> (x, y, conf)
    trace: dict = {}               # i -> [(score, theta, far, near, n_sel), ...] 診断用
    state = {"theta": None, "omega": 0.0, "r": None, "last": None, "miss": 0, "pending": None}
    buf = collections.deque()       # (i, small) の窓
    t_detect = 0.0
    nb = 360 // ANGLE_BIN
    smooth_k = np.array([1, 2, 3, 2, 1], dtype=np.float32) / 9.0
    # シャフトが半分写っている時の角度ヒストグラム重みの目安 (幅 3px, 重み 0.1+(d/r_max)^2 の積分)
    hist_ref = 0.3 * (r_max - r_min)

    def detect(i, prev, cur, nxt, dbg_tag=""):
        """1 フレーム分のヘッド検出。見つかれば raw[i] に入れて True を返す。"""
        t0 = time.time()
        motion = cv2.min(cv2.absdiff(cur, prev), cv2.absdiff(cur, nxt)).max(axis=2)
        mask = (motion > MOTION_THR).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 人物マスク (骨格の太線 + 手首まわり)、円環、地面より下 (影) を除外
        pts = arr[i]
        body = np.zeros((ah, aw), np.uint8)
        for a, b in BODY_LINES:
            cv2.line(body, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), 255, body_w)
        wx, wy = wrist[i]
        cv2.circle(body, (int(wx), int(wy)), hand_r, 255, -1)
        cv2.circle(body, tuple(pts[NOSE].astype(int)), head_excl, 255, -1)
        dist = np.hypot(xx - wx, yy - wy)
        ground = foot_y[i] + GROUND_MARGIN * body_h
        mask[(body > 0) | (dist < r_min) | (dist > r_max) | (yy > ground)] = 0
        ys, xs = np.nonzero(mask)
        found = False
        head = None
        if len(xs) >= min_px:
            d = dist[ys, xs]
            ang = np.degrees(np.arctan2(ys - wy, xs - wx)) % 360.0
            bins = (ang / ANGLE_BIN).astype(int) % nb
            # シャフト+ヘッド+残像は手首から見て同じ角度に並ぶ → 角度ヒストグラムのピーク
            # 遠い画素ほど重く (手元の腕・服の動きより、伸びたシャフトを優先)
            hist = np.bincount(bins, weights=0.1 + (d / r_max) ** 2, minlength=nb).astype(np.float32)
            hs = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), smooth_k, "valid")
            cand = [b for b in range(nb)
                    if hs[b] >= hs[(b - 1) % nb] and hs[b] >= hs[(b + 1) % nb]
                    and hs[b] >= 0.15 * hs.max() and hs[b] > 0]
            hs_norm = np.minimum(1.0, hs / hist_ref)
            best = None
            cands_info = []
            pend_hit = None
            # 手元から伸びる候補がこのフレームにあるか (あれば離れた塊を減点する)
            has_attached = False
            for b in cand:
                theta = (b + 0.5) * ANGLE_BIN
                dth = np.abs((ang - theta + 180.0) % 360.0 - 180.0)
                sel = dth <= WEDGE_DEG
                if sel.sum() >= max(6, 0.3 * min_px) and float(np.percentile(d[sel], 5)) <= 1.5 * r_min                         and (np.percentile(d[sel], 95) - np.percentile(d[sel], 5)) >= 0.12 * r_max:
                    has_attached = True
                    break
            for b in cand:
                theta = (b + 0.5) * ANGLE_BIN
                dth = np.abs((ang - theta + 180.0) % 360.0 - 180.0)
                sel = dth <= WEDGE_DEG
                if sel.sum() < max(6, 0.3 * min_px):
                    cands_info.append((-9.0, round(theta, 1), "few", int(sel.sum())))
                    continue
                ds = d[sel]
                far = float(np.percentile(ds, 95))
                near = float(np.percentile(ds, 5))
                if (far - near) < 0.12 * r_max and sel.sum() < head_d ** 2:
                    cands_info.append((-8.0, round(theta, 1), "small", round(far, 1), round(near, 1), int(sel.sum())))
                    continue   # 小さく短い塊 (フレア・服の縁) はヘッドでもシャフトでもない
                if state["theta"] is None and (far - near) < 0.25 * r_max                         and sel.sum() < 2.0 * head_d ** 2:
                    # 追跡開始前: シャフトが見えない小さな動きでは始めない。ただし遠くの塊が
                    # 2 フレーム続けて同じ方向・同じ距離に出るなら (ヘッドだけが動き出した) 開始する
                    weak = far >= 0.4 * r_max and sel.sum() >= 0.5 * head_d ** 2
                    pend = state.get("pending")
                    consistent = (weak and pend is not None and i - pend[0] <= 2
                                  and abs((theta - pend[1] + 180.0) % 360.0 - 180.0) <= 12.0
                                  and abs(far - pend[2]) <= 0.15 * r_max)
                    if not consistent:
                        if weak and (pend_hit is None or far > pend_hit[2]):
                            pend_hit = (i, theta, far)
                        cands_info.append((-7.0, round(theta, 1), "acq", round(far, 1), round(near, 1), int(sel.sum())))
                        continue
                score = hs_norm[b] + 0.4 * far / r_max + 0.3 * min(1.0, (far - near) / (0.4 * r_max))
                # クラブは手に繋がっている: 手元から伸びている候補を優先し、離れた塊 (服・髪・影) は減点
                if near <= 1.5 * r_min:
                    score += 0.6
                elif near > 2.5 * r_min and has_attached:
                    score -= 0.5
                if state["theta"] is not None:
                    th_pred = state["theta"] + state["omega"]
                    dang = abs((theta - th_pred + 180.0) % 360.0 - 180.0)
                    score += 0.8 * np.exp(-(dang / ANGLE_GATE) ** 2) - (0.4 if dang > 2 * ANGLE_GATE else 0.0)
                    if state["r"] is not None:
                        score -= 0.4 * min(1.0, abs(far - state["r"]) / r_max)
                cands_info.append((round(float(score), 2), round(theta, 1), round(far, 1), round(near, 1), int(sel.sum())))
                if best is None or score > best[0]:
                    best = (score, theta, far, sel)
            state["pending"] = pend_hit
            trace[i] = sorted(cands_info, reverse=True, key=lambda c: c[0])[:4] + [("npx", int(len(xs)), "state", None if state["theta"] is None else round(state["theta"], 1))]
            if best is not None and best[0] >= 0.5:
                score, theta, far, sel = best
                tipsel = sel & (d >= far - 1.5 * head_d)
                hx, hy = float(xs[tipsel].mean()), float(ys[tipsel].mean())
                head = (hx, hy)
                n_sel = int(sel.sum())
                ext = float(np.percentile(d[sel], 95) - np.percentile(d[sel], 5))
                conf = (1.0 + 0.6 * min(1.0, n_sel / (2.0 * head_d ** 2))
                        + 0.6 * min(1.0, ext / (0.4 * r_max)))
                if state["theta"] is not None:
                    dang = abs((theta - (state["theta"] + state["omega"]) + 180.0) % 360.0 - 180.0)
                    conf += 0.5 if dang <= ANGLE_GATE else -0.5
                raw[i] = (hx, hy, float(conf))
                if state["last"] is not None and i - state["last"] <= 3:
                    state["omega"] = ((theta - state["theta"] + 180.0) % 360.0 - 180.0) / (i - state["last"])
                state.update(theta=theta, r=far, last=i, miss=0)
                found = True
        if debug_dir is not None and (found or dbg_tag == "slow"):
            dbg = cur.copy()
            dbg[mask > 0] = (0, 0, 255)
            cv2.circle(dbg, (int(wx), int(wy)), int(r_min), (255, 255, 0), 1)
            cv2.circle(dbg, (int(wx), int(wy)), int(r_max), (255, 255, 0), 1)
            cv2.line(dbg, (0, int(ground)), (aw, int(ground)), (0, 200, 200), 1)
            if head is not None:
                cv2.circle(dbg, (int(head[0]), int(head[1])), 6, (0, 255, 0), 2)
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(debug_dir) / f"dbg_{i:04d}{dbg_tag}.png"), dbg)
        return found, time.time() - t0

    for i, (fno, small) in enumerate(_iter_frames(video_path, in_frame, out_frame,
                                                  interlaced, upper_field_first, (aw, ah))):
        buf.append((i, small))
        if len(buf) > 2 * gap_slow + 1:
            buf.popleft()
        if len(buf) == 2 * gap_slow + 1:
            ci, cur = buf[gap_slow]
            found, dt = detect(ci, buf[gap_slow - gap][1], cur, buf[gap_slow + gap][1])
            t_detect += dt
            if not found:   # 動きが小さい: 間隔を広げて再試行
                found, dt = detect(ci, buf[0][1], cur, buf[-1][1], dbg_tag="slow")
                t_detect += dt
            if not found and state["theta"] is not None:
                state["miss"] += 1
                if state["miss"] >= MISS_RESET:   # 見失った: 追跡をリセットして再取得に戻る
                    state.update(theta=None, omega=0.0, r=None, miss=0)
                else:
                    state["theta"] = (state["theta"] + state["omega"]) % 360.0
                    state["omega"] *= 0.8
        if i % 20 == 0:
            report(0.5 + 0.45 * i / T, "ヘッド探索")

    # ---- 欠測の扱い: 短い欠測は PCHIP、先頭/末尾の静止区間は最寄りの観測位置を保持
    report(0.97, "軌跡を整形")
    ks = sorted(raw)
    if len(ks) < 2:
        raise ValueError("ヘッドを追跡できませんでした")
    strong = [k for k in ks if raw[k][2] >= CONF_OK]
    anchors = strong if len(strong) >= 4 else ks
    span = list(range(anchors[0], anchors[-1] + 1))
    sx = np.array([raw[k][0] for k in anchors])
    sy = np.array([raw[k][1] for k in anchors])
    try:
        from scipy.interpolate import PchipInterpolator
        xs = PchipInterpolator(anchors, sx)(span)
        ys = PchipInterpolator(anchors, sy)(span)
    except Exception:
        xs = np.interp(span, anchors, sx)
        ys = np.interp(span, anchors, sy)
    conf_i = {}
    for t, k in enumerate(span):
        if k in raw and raw[k][2] >= CONF_OK:
            xs[t], ys[t] = raw[k][0], raw[k][1]
            conf_i[k] = raw[k][2]
        else:
            conf_i[k] = raw[k][2] if k in raw else 0.0
    # 先頭 (アドレス) と末尾 (フィニッシュ) の静止区間: 位置を保持
    head_xy = [(float(xs[t]), float(ys[t])) for t in range(len(span))]
    lead = [(k, head_xy[0]) for k in range(0, span[0])]
    tail = [(k, head_xy[-1]) for k in range(span[-1] + 1, T)]
    seq = lead + list(zip(span, head_xy)) + tail
    for k, _ in lead + tail:
        conf_i[k] = HOLD_CONF

    inv = 1.0 / scale
    points = [(int(round(x * inv)), int(round(y * inv)), frame_ids[k]) for k, (x, y) in seq]
    conf = {frame_ids[k]: conf_i[k] for k, _ in seq}
    wrist_out = {frame_ids[i]: (float(wrist[i][0] * inv), float(wrist[i][1] * inv))
                 for i in range(T)}
    res = ClubTrackResult(points=points, conf=conf, wrist=wrist_out,
                          body_h=body_h * inv, seconds=time.time() - t_start,
                          pose_seconds=t_pose, scale=scale,
                          frames_total=T, frames_detected=len(raw))
    res.detect_seconds = t_detect
    res.raw = {frame_ids[k]: v for k, v in raw.items()}   # 補間前の観測値 (診断用)
    res.trace = {frame_ids[k]: v for k, v in trace.items()}
    # 骨格 33 点 (欠測補間済み、元解像度 px)。縦型書き出しの骨格線・クロップに使う
    res.landmarks = {frame_ids[i]: (arr[i] * inv).tolist() for i in range(T)}
    report(1.0, "完了")
    return res
