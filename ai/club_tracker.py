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
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger("club_tracker")

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
PLAYER_DILATE_RATIO = 0.55  # 選手マスクの膨張半径 (身長代替値比)。この外側の動き (ギャラリー・バッグ・木) は見ない
SHAFT_MIN_LEN_RATIO = 0.15  # シャフト線分の最小長 (r_max 比)
SHAFT_PASS_RATIO = 2.0      # 線分の延長が手首中点からこの距離 (hand_r 比) 以内を通れば「手から伸びる線」
MAX_HEAD_SPEED_BH = 30.0    # ヘッド速度の上限 (身長代替値/秒)。前の観測からこれ以上跳んだ候補は棄却
GAP_MAX_SEC = 0.10          # これより長い欠損は軌道を分割して描かない
RELIABLE_MIN_RATE = 0.35    # 信頼度ゲート: スイング区間の観測率
RELIABLE_MIN_CONT = 0.70    # 信頼度ゲート: 連続する観測の移動量が上限内だった割合
RELIABLE_MIN_OBS = 10       # 信頼度ゲート: 観測フレーム数
RELIABLE_MIN_ATT = 0.10     # 信頼度ゲート: 手元から伸びるシャフトに繋がっていた観測の割合 (130128 は 2%、描けた素材は 12〜47%)
HEAD_NEAR_RATIO = 0.25      # 「頭に乗っている」と見なす、鼻・耳からの距離 (身長代替値比)
DEFAULT_GATE = {            # 信頼度ゲートの既定 (2026-10-22 tune の目視判定で決定、docs/CLUB_TRACKING_EVAL.md §10)。
    "min_observed": 6,                # 実際の値は ai.config.gate_thresholds (config/ai.json, settings.json) で上書き
    "min_rate": RELIABLE_MIN_RATE, "min_continuity": RELIABLE_MIN_CONT,
    "min_hand_connect_ratio": 0.0,    # 手元接続率は使わない (正しい区間でも 0% が多い)
    "min_motion_bh": 0.3,             # 静止した物 (脚・カメラマンの手) を追った区間を止める
    "max_on_head_ratio": 0.3,         # 選手の帽子・頭を追った区間を止める
    "max_dist_std": 0.19,             # 手首距離のばらつき: 観客・背景を追うと大きい
    "max_dist_change": 0.07,          # 手首距離の 1 フレーム変化: 同上
    "min_speed_ratio": 1.0,           # ヘッドが手より遅い区間 (観客) を止める
    "min_wrist_ratio": 0.05,          # 手が止まったままヘッドだけが 2 身長以上動く区間 (観客) を止める
}
WRIST_RATIO_MIN_MOTION = 2.0   # min_wrist_ratio は区間の道のりがこれ以上 (身長比) のときだけ判定する


def wrist_features(pts, wrist: dict, body_h: float) -> dict:
    """観測点と手首中点の関係。クラブは手に固定されているので、手首との距離は滑らかにしか変わらず、
    ヘッドは平均して手より速く動き、手が止まっているのにヘッドだけが長く動くことはない。
      dist_std     : 手首距離 (身長比) の標準偏差。背景や他人を追うと大きい
      dist_change  : 手首距離の 1 フレームあたりの変化 (身長比) の平均
      speed_ratio  : ヘッド速度 / 手首速度 の平均比
      wrist_motion : 手首がたどった道のり (身長比)
    """
    bh = max(float(body_h), 1.0)
    ds, dd, hv, wv = [], [], [], []
    prev = None
    for x, y, f in pts:
        w = wrist.get(f)
        if w is None:
            continue
        d = float(np.hypot(x - w[0], y - w[1])) / bh
        ds.append(d)
        if prev is not None and f - prev[2] <= 3:
            n = f - prev[2]
            dd.append(abs(d - prev[3]) / n)
            hv.append(float(np.hypot(x - prev[0], y - prev[1])) / n / bh)
            wv.append(float(np.hypot(w[0] - prev[4][0], w[1] - prev[4][1])) / n / bh)
        prev = (x, y, f, d, w)
    return {"dist_mean": round(float(np.mean(ds)), 3) if ds else 0.0,
            "dist_std": round(float(np.std(ds)), 3) if ds else 0.0,
            "dist_change": round(float(np.mean(dd)), 4) if dd else 0.0,
            "speed_ratio": round(float(np.mean(hv) / max(float(np.mean(wv)), 1e-4)), 2) if wv else 0.0,
            "wrist_motion": round(float(np.sum(wv)), 3) if wv else 0.0}


def resolve_gate(gate: Optional[dict] = None) -> dict:
    """既定 ← config/ai.json ← settings.json (呼び出し側が渡す gate) の順に閾値を決める。"""
    base = dict(DEFAULT_GATE)
    try:
        from ai.config import gate_thresholds
        base.update({k: v for k, v in gate_thresholds(None).items() if k in base})
    except Exception:
        pass
    if gate:
        base.update({k: v for k, v in gate.items() if k in base})
    return base


def motion_bh(pts, body_h: float) -> float:
    """点列がたどった道のり (身長代替値比)。静止した物を追っていると小さい。"""
    if len(pts) < 2:
        return 0.0
    path = sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pts[:-1], pts[1:]))
    return path / max(float(body_h), 1.0)


def on_head_ratio(pts, landmarks: dict, body_h: float) -> float:
    """観測点が選手の頭 (鼻 0・耳 7/8) から HEAD_NEAR_RATIO×身長以内にあった割合。帽子や顔を追っていると大きい。"""
    if not pts or not landmarks:
        return 0.0
    r = HEAD_NEAR_RATIO * max(float(body_h), 1.0)
    n = 0
    for x, y, f in pts:
        lm = landmarks.get(f)
        if lm is None or len(lm) < 9:
            continue
        if min(np.hypot(x - lm[i][0], y - lm[i][1]) for i in (0, 7, 8)) <= r:
            n += 1
    return n / len(pts)


def gate_reasons(q: dict, gate: dict) -> list:
    """品質 q が閾値 gate を満たさない理由 (空なら通過)。表示用に閾値も入れる。"""
    reasons = []
    if q.get("observed", 0) < gate["min_observed"]:
        reasons.append(f"観測 {q.get('observed', 0)}f < {gate['min_observed']}f")
    if q.get("rate", 0.0) < gate["min_rate"]:
        reasons.append(f"観測率 {q.get('rate', 0.0):.0%} < {gate['min_rate']:.0%}")
    if q.get("continuity", 0.0) < gate["min_continuity"]:
        reasons.append(f"連続性 {q.get('continuity', 0.0):.0%} < {gate['min_continuity']:.0%}")
    if q.get("attached", 0.0) < gate["min_hand_connect_ratio"]:
        reasons.append(f"手元接続 {q.get('attached', 0.0):.0%} < {gate['min_hand_connect_ratio']:.0%}")
    if q.get("motion_bh", 0.0) < gate["min_motion_bh"]:
        reasons.append(f"動き {q.get('motion_bh', 0.0):.2f} < {gate['min_motion_bh']:.2f} 身長")
    if q.get("on_head", 0.0) > gate["max_on_head_ratio"]:
        reasons.append(f"頭上 {q.get('on_head', 0.0):.0%} > {gate['max_on_head_ratio']:.0%}")
    if q.get("dist_std", 0.0) > gate.get("max_dist_std", 9.0):
        reasons.append(f"手首距離ばらつき {q.get('dist_std', 0.0):.2f} > {gate['max_dist_std']:.2f}")
    if q.get("dist_change", 0.0) > gate.get("max_dist_change", 9.0):
        reasons.append(f"手首距離変化 {q.get('dist_change', 0.0):.3f} > {gate['max_dist_change']:.3f}")
    if q.get("speed_ratio", 0.0) < gate.get("min_speed_ratio", 0.0) and q.get("wrist_motion", 0.0) > 0:
        reasons.append(f"速度比 {q.get('speed_ratio', 0.0):.2f} < {gate['min_speed_ratio']:.2f}")
    mot = q.get("motion_bh", 0.0)
    if mot >= WRIST_RATIO_MIN_MOTION and q.get("wrist_motion", 0.0) / max(mot, 1e-6) < gate.get("min_wrist_ratio", 0.0):
        reasons.append(f"手が止まったままヘッドが {mot:.1f} 身長分動いた (手 {q.get('wrist_motion', 0.0):.2f})")
    return reasons
MAX_POSES = 4               # 同時に検出する人数の上限 (キャディ・カメラマンを含めて選手を選ぶため)
TRACK_ASSOC = 1.0           # 腰中点の距離 (胴体長比) がこれ未満なら同じ人物トラック
BODY_H_WINDOW = 0.4         # 身長代替値を決めるのに使うクリップ前半の割合 (アドレス付近。終盤のズームを除外)


@dataclass
class ClubTrackResult:
    points: list                      # [(x, y, frame), ...] 観測できたフレームのみ (補間しない)
    conf: dict                        # {frame: 信頼度}
    wrist: dict = field(default_factory=dict)   # {frame: (x, y)} 手首中点 (元解像度)
    body_h: float = 0.0               # 身長代替値 (元解像度 px)
    seconds: float = 0.0
    pose_seconds: float = 0.0
    scale: float = 1.0                # 解析解像度 / 元解像度
    frames_total: int = 0
    frames_detected: int = 0
    real_fps: float = 30.0
    quality: dict = field(default_factory=dict)   # 観測率・連続性・手元接続率・判定理由
    reliable: bool = False                        # 信頼度ゲート (False なら軌道を描かない)

    def segments(self, max_gap_sec: float = GAP_MAX_SEC, min_pts: int = 4) -> list:
        """欠損 (max_gap_sec より長い) で分割した連続区間ごとの点列。短すぎる区間は捨てる。"""
        max_gap = max(1.0, max_gap_sec * self.real_fps)
        out, cur = [], []
        for p in sorted(self.points, key=lambda q: q[2]):
            if cur and p[2] - cur[-1][2] > max_gap:
                if len(cur) >= min_pts:
                    out.append(cur)
                cur = []
            cur.append(p)
        if len(cur) >= min_pts:
            out.append(cur)
        return out

    def segment_quality(self, max_gap_sec: float = GAP_MAX_SEC, min_pts: int = 4, gate: Optional[dict] = None) -> list:
        """区間ごとの品質。[(points, quality, reliable), ...]。
        区間内の観測率・連続性・手元接続率・動き・頭上率を閾値 (gate、省略時は self.gate か既定) で判定。"""
        gate = gate or getattr(self, "gate", None) or resolve_gate()
        max_step = MAX_HEAD_SPEED_BH * self.body_h / max(self.real_fps, 1.0)
        att_map = getattr(self, "attached", {})
        lms = getattr(self, "landmarks", {}) or {}
        out = []
        for seg in self.segments(max_gap_sec, min_pts):
            fs = [p[2] for p in seg]
            span = fs[-1] - fs[0] + 1
            rate = len(seg) / max(span, 1)
            pairs = [(a, b) for a, b in zip(seg[:-1], seg[1:]) if b[2] - a[2] <= 3]
            cont = (sum(1 for a, b in pairs if np.hypot(b[0] - a[0], b[1] - a[1]) <= max_step * (b[2] - a[2]))
                    / len(pairs)) if pairs else 0.0
            att = sum(1 for p in seg if att_map.get(p[2])) / len(seg)
            q = {"frames": (fs[0], fs[-1]), "observed": len(seg), "span": span, "rate": round(rate, 3),
                 "continuity": round(cont, 3), "attached": round(att, 3),
                 "motion_bh": round(motion_bh(seg, self.body_h), 3),
                 "on_head": round(on_head_ratio(seg, lms, self.body_h), 3),
                 **wrist_features(seg, self.wrist or {}, self.body_h)}
            q["reasons"] = gate_reasons(q, gate)
            out.append((seg, q, not q["reasons"]))
        return out

    def confident_points(self, min_conf: float = CONF_OK, max_pts: int = 18,
                         min_gap: float = 2.0, points: list = None) -> list:
        """確信点だけを、編集タブの制御点として使える点数に間引いて返す (points で区間を指定可)。"""
        src = points if points is not None else self.points
        good = [p for p in src if self.conf.get(p[2], 0.0) >= min_conf]
        if len(good) < 2:
            return list(src[:: max(1, len(src) // max_pts)]) if len(src) >= 2 else list(src)
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
        fs = sorted(p[2] for p in self.points if self.conf.get(p[2], 0.0) >= min_conf)
        out = []
        for a, b in zip(fs[:-1], fs[1:]):
            if b - a - 1 >= min_len:
                out.append((a + 1, b - 1))
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
        running_mode=vision.RunningMode.VIDEO, num_poses=MAX_POSES,
        min_pose_detection_confidence=0.3, min_tracking_confidence=0.3)
    return mp, vision.PoseLandmarker.create_from_options(opts)


def _ball_near_feet(frame_bgr, lm, W, H) -> bool:
    """足元 (ボール側いずれか) に白い小さな球があるか (swing_analyzer.detect_ball_candidates を流用)"""
    try:
        from ai.swing_analyzer import detect_ball_candidates
        ankle = ((lm[L_ANKLE, 0] + lm[R_ANKLE, 0]) / 2, (lm[L_ANKLE, 1] + lm[R_ANKLE, 1]) / 2)
        return any(detect_ball_candidates(frame_bgr, ankle, sign, W, H) for sign in (1, -1))
    except Exception:
        return False


def _select_golfer(per_frame: list, first_frame=None, size=None) -> list:
    """フレームごとの複数人の骨格 [(33, 3) ...] から、選手 1 人分の系列 [(33, 3) or None] を返す。

    腰中点の位置でトラックに束ね (近い人物は同じトラック)、トラックごとに
    「手首中点の移動範囲 ÷ 胴体長」を測り、最も大きい (= スイングした) トラックを選手とする。
    キャディ・カメラマン・ギャラリーは立っているだけなので手首がほとんど動かない。
    上位 2 本が拮抗 (25% 以内) するときは、最初のフレームで足元にボールがある方を優先する。
    """
    tracks: list = []    # {"hip": 最近の腰中点, "torso": 胴体長, "frames": {i: lm}}
    for i, poses in enumerate(per_frame):
        used = set()
        for lm in poses or []:
            hip = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2
            sh = (lm[L_SHOULDER, :2] + lm[R_SHOULDER, :2]) / 2
            torso = float(np.linalg.norm(sh - hip)) + 1e-6
            best, best_d = None, None
            for k, t in enumerate(tracks):
                if k in used:
                    continue
                d = float(np.linalg.norm(t["hip"] - hip)) / max(t["torso"], torso)
                if d < TRACK_ASSOC and (best_d is None or d < best_d):
                    best, best_d = k, d
            if best is None:
                tracks.append({"hip": hip, "torso": torso, "frames": {i: lm}})
                used.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t["hip"] = 0.7 * t["hip"] + 0.3 * hip
                t["torso"] = 0.7 * t["torso"] + 0.3 * torso
                t["frames"][i] = lm
                used.add(best)
    if not tracks:
        return [None] * len(per_frame)
    T = len(per_frame)

    def score(t):
        fr = sorted(t["frames"])
        if len(fr) < max(4, 0.2 * T):
            return -1.0
        w = np.array([(t["frames"][f][L_WRIST, :2] + t["frames"][f][R_WRIST, :2]) / 2 for f in fr])
        rng = float(np.linalg.norm(w.max(axis=0) - w.min(axis=0)))
        return rng / t["torso"]

    scores = [score(t) for t in tracks]
    order = sorted(range(len(tracks)), key=lambda j: -scores[j])
    k = order[0]
    ball_note = ""
    if first_frame is not None and size is not None and len(order) >= 2 and scores[order[1]] > 0 \
            and scores[order[1]] >= 0.75 * scores[order[0]]:
        aw, ah = size
        near = []
        for j in order[:3]:
            if scores[j] <= 0:
                continue
            f0 = min(tracks[j]["frames"])
            near.append((j, _ball_near_feet(first_frame, tracks[j]["frames"][f0], aw, ah)))
        with_ball = [j for j, b in near if b]
        if len(with_ball) == 1:
            k = with_ball[0]
            ball_note = f" (手首移動量が拮抗 → 足元にボールがある {k} を採用)"
    log.info("人物トラック %d 本: 手首移動量/胴長 = %s → %d を選手に%s", len(tracks),
             [round(x, 2) for x in scores], k, ball_note)
    chosen = tracks[k]
    return [chosen["frames"].get(i) for i in range(T)]


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
               model_path: Optional[Path] = None, realtime_factor: Optional[float] = None,
               use_detector: Optional[bool] = None, gate: Optional[dict] = None,
               detector_model: Optional[Path] = None,
               progress: Optional[Callable[[float, str], None]] = None,
               debug_dir: Optional[Path] = None) -> ClubTrackResult:
    """クラブヘッド軌道を推定する。座標は元解像度のピクセル。

    realtime_factor: 素材 1 フレームが実時間の何フレーム分か (HFR 2x = 4)。省略時は fps から推定。
    use_detector:    models/clubhead.onnx (ONNX Runtime CPU) をヘッド候補の提案に使う。None = あれば使う。
                     時間的連続性・欠損・信頼度の判定は検出器の有無によらず同じ。
    gate:            信頼度ゲートの閾値 (ai.config.gate_thresholds)。None = 既定 ← config/ai.json。
    detector_model:  検出器の ONNX の場所。None = models/clubhead.onnx (評価で新旧モデルを比べるときに指定)。
    """
    gate = resolve_gate(gate)
    from ai import default_pose_model
    from ai.config import realtime_factor as _rt
    t_start = time.time()
    detector = None
    res_detector_name = ""
    if use_detector is None or use_detector:
        try:
            from ai.club_detector import ClubDetector
            det = ClubDetector(detector_model)
            if det.available():
                detector = det
                res_detector_name = det.path.stem
            elif use_detector:
                log.warning("models/clubhead.onnx が無いため従来方式で動きます")
        except Exception as e:
            log.warning("検出器を初期化できません (%s)。従来方式で動きます", e)
    model_path = Path(model_path) if model_path else default_pose_model()
    W, H, n_total, _fps = _probe(video_path)
    real_fps = _fps * (realtime_factor if realtime_factor else _rt(_fps, "auto")) * (2 if interlaced else 1)
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
    first_small = None
    try:
        for i, (fno, small) in enumerate(_iter_frames(video_path, in_frame, out_frame,
                                                      interlaced, upper_field_first, (aw, ah))):
            if first_small is None:
                first_small = small.copy()
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / 120))
            t_pose += time.time() - t0
            lms.append([np.array([(p.x * aw, p.y * ah, p.visibility) for p in lm], dtype=np.float32)
                        for lm in res.pose_landmarks] if res.pose_landmarks else None)
            frame_ids.append(fno)
            if i % 20 == 0:
                report(0.02 + 0.45 * min(1.0, i / max(1, n_total * (2 if interlaced else 1))),
                       "骨格推定")
    finally:
        landmarker.close()

    T = len(frame_ids)
    if T < 2 * gap_slow + 2:
        raise ValueError("フレームが足りません")
    lms = _select_golfer(lms, first_small, (aw, ah))   # 複数人 → 手首が最も動く 1 人 (拮抗時はボールの足元)

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

    # ---- パス 2: 動き差分 + シャフト線分 + 角度ヒストグラム (動画をもう一度読む)
    report(0.5, "ヘッド探索")
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    dil_r = max(5, int(PLAYER_DILATE_RATIO * body_h))
    max_step = MAX_HEAD_SPEED_BH * body_h / real_fps     # 1 フレームあたりのヘッド移動量の上限 (解析解像度 px)
    raw: dict = {}                 # i -> (x, y, conf)
    attached: dict = {}            # i -> 手元から伸びる候補だったか
    trace: dict = {}
    state = {"theta": None, "omega": 0.0, "last": None, "miss": 0, "pending": None}
    buf = collections.deque()
    t_detect = 0.0
    nb = 360 // ANGLE_BIN
    smooth_k = np.array([1, 2, 3, 2, 1], dtype=np.float32) / 9.0
    hist_ref = 0.3 * (r_max - r_min)
    shaft_min_len = SHAFT_MIN_LEN_RATIO * r_max

    def player_region(i):
        """探索領域 = 膨張した選手マスク ∩ 手首円環、除外 = 人物の太線・手・頭・地面より下"""
        pts = arr[i]
        lines = np.full((ah, aw), 255, np.uint8)
        for a, b in BODY_LINES:
            cv2.line(lines, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), 0, max(3, body_w // 2))
        # 骨格線からの距離 dil_r 以内 (巨大カーネルの dilate より距離変換が速い)
        region = (cv2.distanceTransform(lines, cv2.DIST_L2, 3) <= dil_r).astype(np.uint8) * 255
        body = np.zeros((ah, aw), np.uint8)
        for a, b in BODY_LINES:
            cv2.line(body, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), 255, body_w)
        wx, wy = wrist[i]
        cv2.circle(body, (int(wx), int(wy)), hand_r, 255, -1)
        cv2.circle(body, tuple(pts[NOSE].astype(int)), head_excl, 255, -1)
        dist = np.hypot(xx - wx, yy - wy)
        ground = foot_y[i] + GROUND_MARGIN * body_h
        ok = (region > 0) & (body == 0) & (dist >= r_min) & (dist <= r_max) & (yy <= ground)
        return ok, dist, (wx, wy), ground

    def shaft_lines(gray, gray_prev, gray_next, ok, wx, wy):
        """手首から伸びる直線 (シャフト) を探し、[(theta, far_end, near_dist, length), ...] を返す。

        前後フレームの同じ位置にもエッジがある線分は静止した背景 (木・バッグ・ギャラリーの輪郭) なので捨てる。
        静止中のクラブ (アドレス・トップ) も落ちるが、その区間は欠損のままで良い。
        """
        x0, x1 = max(0, int(wx - r_max) - 2), min(aw, int(wx + r_max) + 3)
        y0, y1 = max(0, int(wy - r_max) - 2), min(ah, int(wy + r_max) + 3)
        edges = np.zeros_like(gray)
        if x1 <= x0 or y1 <= y0:
            return []
        edges[y0:y1, x0:x1] = cv2.Canny(gray[y0:y1, x0:x1], 60, 160)
        edges[~ok] = 0
        static = np.zeros_like(gray)
        static[y0:y1, x0:x1] = cv2.max(cv2.Canny(gray_prev[y0:y1, x0:x1], 60, 160),
                                       cv2.Canny(gray_next[y0:y1, x0:x1], 60, 160))
        static = cv2.dilate(static, kernel, iterations=2)
        segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18,
                               minLineLength=int(shaft_min_len), maxLineGap=int(max(4, 0.03 * r_max)))
        out = []
        if segs is None:
            return out
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            n = max(8, int(np.hypot(x2 - x1, y2 - y1) / 3))
            sx = np.linspace(x1, x2, n).astype(int)
            sy = np.linspace(y1, y2, n).astype(int)
            if float((static[sy, sx] > 0).mean()) > 0.5:
                continue   # 前後フレームにも同じエッジ → 動いていない線
            d1, d2 = np.hypot(x1 - wx, y1 - wy), np.hypot(x2 - wx, y2 - wy)
            near, far = ((x1, y1), d1) if d1 < d2 else ((x2, y2), d2)
            far_pt = (x2, y2) if d1 < d2 else (x1, y1)
            # 線分の延長が手首中点の近くを通るか (点と直線の距離)
            vx, vy = x2 - x1, y2 - y1
            L = np.hypot(vx, vy) + 1e-6
            perp = abs(vx * (wy - y1) - vy * (wx - x1)) / L
            if perp > SHAFT_PASS_RATIO * hand_r or near[1] > 0.6 * r_max:
                continue
            theta = float(np.degrees(np.arctan2(far_pt[1] - wy, far_pt[0] - wx)) % 360.0)
            out.append((theta, far_pt, float(near[1]), float(L)))
        out.sort(key=lambda t: -t[3])
        return out[:6]

    def detect(i, prev, cur, nxt, dbg_tag=""):
        """1 フレーム分のヘッド検出。見つかれば raw[i] に入れて True を返す。"""
        t0 = time.time()
        ok, dist, (wx, wy), ground = player_region(i)
        motion = cv2.min(cv2.absdiff(cur, prev), cv2.absdiff(cur, nxt)).max(axis=2)
        mask = (motion > MOTION_THR).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask[~ok] = 0
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        lines = shaft_lines(gray, cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY), ok, wx, wy)
        ys, xs = np.nonzero(mask)
        found = False
        head = None
        cands = []     # (score, theta, head(x,y), near, n_sel, kind)
        last_head = None
        if state["last"] is not None and i - state["last"] <= 3:
            last_head = np.array(raw[state["last"]][:2])
            step_cap = max_step * (i - state["last"])

        def continuity(hx, hy):
            if last_head is None:
                return 0.0, True
            d = float(np.hypot(hx - last_head[0], hy - last_head[1]))
            return d, d <= step_cap

        # --- (a) 動き差分の角度ヒストグラム (従来法)
        if len(xs) >= min_px:
            d = dist[ys, xs]
            ang = np.degrees(np.arctan2(ys - wy, xs - wx)) % 360.0
            bins = (ang / ANGLE_BIN).astype(int) % nb
            hist = np.bincount(bins, weights=0.1 + (d / r_max) ** 2, minlength=nb).astype(np.float32)
            hs = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), smooth_k, "valid")
            peaks = [b for b in range(nb)
                     if hs[b] >= hs[(b - 1) % nb] and hs[b] >= hs[(b + 1) % nb]
                     and hs[b] >= 0.15 * hs.max() and hs[b] > 0]
            hs_norm = np.minimum(1.0, hs / hist_ref)
            for b in peaks:
                theta = (b + 0.5) * ANGLE_BIN
                dth = np.abs((ang - theta + 180.0) % 360.0 - 180.0)
                sel = dth <= WEDGE_DEG
                if sel.sum() < max(6, 0.3 * min_px):
                    continue
                ds = d[sel]
                far, near = float(np.percentile(ds, 95)), float(np.percentile(ds, 5))
                if (far - near) < 0.12 * r_max and sel.sum() < head_d ** 2:
                    continue
                tipsel = sel & (d >= far - 1.5 * head_d)
                hx, hy = float(xs[tipsel].mean()), float(ys[tipsel].mean())
                score = hs_norm[b] + 0.4 * far / r_max + 0.3 * min(1.0, (far - near) / (0.4 * r_max))
                is_att = near <= 1.5 * r_min
                if is_att:
                    score += 0.6
                # シャフト線分と向きが合えば加点 (背景の動きとクラブを区別する決め手)
                if any(abs((theta - L[0] + 180.0) % 360.0 - 180.0) <= 8.0 for L in lines):
                    score += 0.8
                    is_att = True
                elif lines and state["theta"] is None:
                    score -= 0.4   # シャフトが別の向きに見えているのに動きだけが違う方向 → 背景
                cands.append((score, theta, (hx, hy), near, int(sel.sum()), "motion", is_att))
        # --- (b) シャフト線分の先端 (静止中や動き差分が出ないとき)
        for theta, far_pt, near, L in lines:
            fx, fy = float(far_pt[0]), float(far_pt[1])
            if L < 0.3 * r_max:
                continue
            score = 0.5 + 0.5 * min(1.0, L / (0.7 * r_max))
            cands.append((score, theta, (fx, fy), near, 0, "shaft", True))
        # --- (c) 学習ベース検出器 (ONNX) の候補。手首中心の切り出しだけを見るので背景の動きに引かれにくい
        if detector is not None:
            try:
                for dx_, dy_, sc in detector.detect(cur, (wx, wy), body_h):
                    dd = float(np.hypot(dx_ - wx, dy_ - wy))
                    iy, ix = int(min(ah - 1, max(0, dy_))), int(min(aw - 1, max(0, dx_)))
                    if dd < r_min or dd > r_max or not ok[iy, ix]:
                        continue
                    theta = float(np.degrees(np.arctan2(dy_ - wy, dx_ - wx)) % 360.0)
                    # is_att=False: 検出器の候補は「手元接続」に数えない。学習データが少ないうちは誤検出もあり、
                    # 接続ありに数えると信頼度ゲートを素通りする (det_1 で 130128 がバッグを追ったまま通過した)。
                    # 追跡開始は motion の遠方塊と同じく 2 フレーム連続で同じ向き・距離に出たときだけ
                    cands.append((0.6 + 1.0 * float(sc), theta, (float(dx_), float(dy_)), 0.0, 0, "det", False))
            except Exception as e:
                log.debug("detector error: %s", e)
        # --- 時間的連続性 (前フレームからの移動量の上限) と向きの予測で選ぶ
        best = None
        infos = []
        pend_hit = None
        for score, theta, (hx, hy), near, n_sel, kind, is_att in cands:
            dmove, ok_move = continuity(hx, hy)
            if not ok_move:
                infos.append((-5.0, round(theta, 1), kind, "jump", round(dmove)))
                continue
            if state["theta"] is not None:
                dang = abs((theta - (state["theta"] + state["omega"]) + 180.0) % 360.0 - 180.0)
                score += 0.8 * np.exp(-(dang / ANGLE_GATE) ** 2) - (0.4 if dang > 2 * ANGLE_GATE else 0.0)
            elif not is_att:
                # 追跡開始前: 手元から伸びていない動きでは原則始めない (ギャラリー・服・影)。
                # ただし遠くの大きな塊が 2 フレーム続けて同じ向き・同じ距離に出るなら (ヘッドだけが動き出した) 開始する
                far_d = float(np.hypot(hx - wx, hy - wy))
                weak = (kind == "motion" and far_d >= 0.4 * r_max and n_sel >= 2.0 * head_d ** 2) or kind == "det"
                pend = state.get("pending")
                consistent = (weak and pend is not None and i - pend[0] <= 2
                              and abs((theta - pend[1] + 180.0) % 360.0 - 180.0) <= 12.0
                              and abs(far_d - pend[2]) <= 0.15 * r_max)
                if not consistent:
                    if weak and (pend_hit is None or far_d > pend_hit[2]):
                        pend_hit = (i, theta, far_d)
                    infos.append((-7.0, round(theta, 1), kind, "acq"))
                    continue
            infos.append((round(float(score), 2), round(theta, 1), kind, round(near, 1), n_sel))
            if best is None or score > best[0]:
                best = (score, theta, (hx, hy), near, n_sel, kind, is_att)
        state["pending"] = pend_hit
        trace[i] = sorted(infos, reverse=True, key=lambda c: c[0])[:4] + [("npx", int(len(xs)), "lines", len(lines))]
        if best is not None and best[0] >= 0.5:
            score, theta, (hx, hy), near, n_sel, kind, is_att = best
            head = (hx, hy)
            conf = 1.0 + 0.6 * min(1.0, n_sel / (2.0 * head_d ** 2)) + (0.6 if is_att else 0.0)
            if state["theta"] is not None:
                dang = abs((theta - (state["theta"] + state["omega"]) + 180.0) % 360.0 - 180.0)
                conf += 0.5 if dang <= ANGLE_GATE else -0.5
            raw[i] = (hx, hy, float(conf))
            attached[i] = bool(is_att)
            if state["last"] is not None and i - state["last"] <= 3:
                state["omega"] = ((theta - state["theta"] + 180.0) % 360.0 - 180.0) / (i - state["last"])
            state.update(theta=theta, last=i, miss=0)
            found = True
        if debug_dir is not None and (found or dbg_tag == "slow"):
            dbg = cur.copy()
            dbg[~ok] = (dbg[~ok] * 0.35).astype(np.uint8)
            dbg[mask > 0] = (0, 0, 255)
            for theta, far_pt, near, L in lines:
                cv2.line(dbg, (int(wx), int(wy)), (int(far_pt[0]), int(far_pt[1])), (255, 0, 255), 1)
            cv2.circle(dbg, (int(wx), int(wy)), int(r_min), (255, 255, 0), 1)
            cv2.circle(dbg, (int(wx), int(wy)), int(r_max), (255, 255, 0), 1)
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
            if not found:
                found, dt = detect(ci, buf[0][1], cur, buf[-1][1], dbg_tag="slow")
                t_detect += dt
            if not found and state["theta"] is not None:
                state["miss"] += 1
                if state["miss"] >= MISS_RESET:
                    state.update(theta=None, omega=0.0, miss=0)
                else:
                    state["theta"] = (state["theta"] + state["omega"]) % 360.0
                    state["omega"] *= 0.8
        if i % 20 == 0:
            report(0.5 + 0.45 * i / T, "ヘッド探索")

    # ---- 欠損は補間しない。信頼度を算出する
    report(0.97, "信頼度を算出")
    ks = sorted(raw)
    inv = 1.0 / scale
    points = [(int(round(raw[k][0] * inv)), int(round(raw[k][1] * inv)), frame_ids[k]) for k in ks]
    conf = {frame_ids[k]: raw[k][2] for k in ks}
    wrist_out = {frame_ids[i]: (float(wrist[i][0] * inv), float(wrist[i][1] * inv)) for i in range(T)}
    quality = {"observed": len(ks), "frames": T}
    if len(ks) >= 2:
        span = ks[-1] - ks[0] + 1
        rate = len(ks) / span
        pairs = [(a, b) for a, b in zip(ks[:-1], ks[1:]) if b - a <= 3]
        cont = (sum(1 for a, b in pairs
                    if np.hypot(raw[b][0] - raw[a][0], raw[b][1] - raw[a][1]) <= max_step * (b - a)) / len(pairs)
                if pairs else 0.0)
        att = sum(1 for k in ks if attached.get(k)) / len(ks)
        pts_a = [(raw[k][0], raw[k][1], k) for k in ks]
        quality.update({"span": span, "rate": round(rate, 3), "continuity": round(cont, 3),
                        "attached": round(att, 3),
                        "motion_bh": round(motion_bh(pts_a, body_h), 3),
                        "on_head": round(on_head_ratio(pts_a, {k: arr[k] for k in ks}, body_h), 3),
                        **wrist_features(pts_a, {k: (wrist[k][0], wrist[k][1]) for k in ks}, body_h)})
        reasons = gate_reasons(quality, gate)
        quality["reasons"] = reasons
        reliable = not reasons
    else:
        quality.update({"span": 0, "rate": 0.0, "continuity": 0.0, "attached": 0.0,
                        "reasons": ["ヘッドを観測できませんでした"]})
        reliable = False
    res = ClubTrackResult(points=points, conf=conf, wrist=wrist_out,
                          body_h=body_h * inv, seconds=time.time() - t_start,
                          pose_seconds=t_pose, scale=scale,
                          frames_total=T, frames_detected=len(ks), real_fps=real_fps,
                          quality=quality, reliable=reliable)
    res.detect_seconds = t_detect
    res.detector_used = detector is not None
    res.detector_name = res_detector_name
    quality["detector"] = detector is not None
    quality["detector_model"] = res_detector_name
    quality["gate"] = dict(gate)
    res.gate = dict(gate)
    res.attached = {frame_ids[k]: bool(v) for k, v in attached.items()}
    res.landmarks = {frame_ids[i]: (arr[i] * inv).tolist() for i in range(T)}
    seg_q = res.segment_quality(gate=gate)
    quality["segments"] = [{**q, "reliable": ok} for _, q, ok in seg_q]
    if seg_q and any(ok for _, _, ok in seg_q):
        res.reliable = True          # 信頼できる区間が 1 つでもあれば、その区間だけ描く
        quality["reasons"] = [] if reliable else quality.get("reasons", [])
    res.quality = quality
    res.raw = {frame_ids[k]: v for k, v in raw.items()}
    res.trace = {frame_ids[k]: v for k, v in trace.items()}
    log.info("軌道の信頼度: %s → %s", quality, "OK" if reliable else "不足")
    report(1.0, "完了")
    return res
