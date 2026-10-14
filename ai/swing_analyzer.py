"""ai.swing_analyzer — 骨格 (MediaPipe 33 点) によるスイング解析。swingcut/analyze.py の移植 + 正面対応。

入力は club_tracker が残した骨格 (analysis/{clip_id}/club_track.json の landmarks、元解像度 px、
欠測補間済み)。swingcut と違い pose.json も 30fps 正規化動画も使わず、任意の fps で動く
(フレーム数で決めていた窓は fps/30 で比例させる)。

出力 (analyze() の戻り値):
    phases      : {address, top, impact, finish}  (入力と同じ相対フレーム番号)
    impact      : 手法ごとの結果 (ball_roi / wrist) と採用手法・警告  ※打球音は Phase 1 では使わない
    camera      : {"angle": dtl / face_on / unknown, "ratio": 肩幅/胴長, "confidence": 0〜1, "source": auto / 設定}
    metrics     : 撮影方向別の指標 (身長比 % / 度)。unknown のときは tempo のみ
    comments    : config/rules.yaml のルールで生成したコメント
    tracks      : 平滑化済み骨格 {landmark_index: (T, 2) px}  (描画用)
    frames      : tracks の行に対応する相対フレーム番号
    warnings    : 文字列のリスト

撮影方向の自動判定: アドレス時の「両肩の見かけの幅 ÷ 胴体の長さ (肩中点〜腰中点)」。
DTL では肩が重なって幅が小さく (比 < 0.35)、正面では広い (比 > 0.65)。間は「斜め」で unknown。
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ai import resource_root

log = logging.getLogger("swing_analyzer")

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
USED = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
        L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]

DTL_MAX_RATIO = 0.35      # 肩幅/胴長 がこれ未満なら DTL
FACE_MIN_RATIO = 0.65     # これ以上なら正面


# ---------------------------------------------------------------- 前処理

def build_tracks(landmarks: dict, W: int, H: int):
    """{frame: [(x, y) * 33]} → (frames, {li: (T, 2) 正規化座標})。欠測は線形補間。"""
    frames = sorted(landmarks)
    T = len(frames)
    tracks = {}
    for li in USED:
        arr = np.full((T, 2), np.nan)
        for i, f in enumerate(frames):
            lm = landmarks[f]
            if lm is not None and len(lm) > li:
                arr[i] = (lm[li][0] / W, lm[li][1] / H)
        for c in range(2):
            col = arr[:, c]
            nans = np.isnan(col)
            if nans.all():
                col[:] = 0.5
            elif nans.any():
                col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
        tracks[li] = arr
    return frames, tracks


def smooth(tracks: dict, scale: float) -> dict:
    """手首・肘は強め (median + SavGol)、他は軽めの SavGol。窓は fps/30 に比例。"""
    from scipy.signal import medfilt, savgol_filter
    T = len(next(iter(tracks.values())))

    def odd(n):
        n = int(round(n))
        return n if n % 2 else n + 1

    out = {}
    for li, arr in tracks.items():
        sm = arr.copy()
        heavy = li in (L_WRIST, R_WRIST, L_ELBOW, R_ELBOW)
        for c in (0, 1):
            col = arr[:, c]
            if heavy:
                k = min(odd(5 * scale), T if T % 2 else T - 1)
                if k >= 3:
                    col = medfilt(col, kernel_size=k)
                win = min(odd(9 * scale), T if T % 2 else T - 1)
            else:
                win = min(odd(7 * scale), T if T % 2 else T - 1)
            if win >= 5:
                col = savgol_filter(col, window_length=win, polyorder=2)
            sm[:, c] = col
        out[li] = sm
    return out


# ---------------------------------------------------------------- 局面 (手首)

def detect_phases_wrist(tracks, raw, fps, scale):
    """両手首中点の y と速度から address / top / impact 候補 / finish を求める (swingcut 準拠)。"""
    from scipy.signal import find_peaks
    wx = (tracks[L_WRIST][:, 0] + tracks[R_WRIST][:, 0]) / 2
    wy = (tracks[L_WRIST][:, 1] + tracks[R_WRIST][:, 1]) / 2
    T = len(wy)
    speed = np.hypot(np.gradient(wx) * fps, np.gradient(wy) * fps)
    peak = float(speed.max())
    quiet_th = max(0.02, 0.05 * peak)
    move_th = max(0.05, 0.12 * peak)
    k3 = max(3, int(round(3 * scale)))
    onset = next((i for i in range(T - k3) if speed[i:i + k3].min() > move_th), None)
    if onset is None:
        raise ValueError("動き出しを検出できませんでした")
    address = onset
    for i in range(onset - 1, -1, -1):
        if speed[i] < quiet_th:
            address = i
            break
    y_range = float(wy.max() - wy.min())
    peaks, _ = find_peaks(-wy, prominence=0.15 * y_range, distance=max(5, int(round(5 * scale))))
    peaks = [p for p in peaks if p > address + 3 * scale]
    if not peaks:
        raise ValueError("トップを検出できませんでした")
    top = int(peaks[0])
    # インパクト候補: トップ後、生データの手首中点がアドレス時の位置に最も近づく点 (±4 フレーム相当)
    wxr = (raw[L_WRIST][:, 0] + raw[R_WRIST][:, 0]) / 2
    wyr = (raw[L_WRIST][:, 1] + raw[R_WRIST][:, 1]) / 2
    dist = np.hypot(wxr - wxr[address], wyr - wyr[address])
    candidate = top + 1 + int(np.argmin(dist[top + 1:]))
    w = max(4, int(round(4 * scale)))
    w0, w1 = max(top + 1, candidate - w), min(T - 1, candidate + w)
    quiet_th2 = max(0.03, 0.10 * peak)
    k5 = max(5, int(round(5 * scale)))
    finish = T - 1
    for i in range(candidate + k5, T - k5 + 1):
        if speed[i:i + k5].max() < quiet_th2:
            finish = i
            break
    return {"address": int(address), "top": top, "candidate": int(candidate),
            "window": (int(w0), int(w1)), "finish": int(finish), "speed": speed}


# ---------------------------------------------------------------- インパクト (ボール ROI)

def _read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def detect_ball_candidates(frame, ankle, ball_sign, W, H):
    """足元付近・ボール側の帯領域で、明るく小さい円形ブロブを列挙する (swingcut 準拠)。"""
    ax, ay = ankle
    y0, y1 = max(0, int(ay - 0.06 * H)), min(H, int(ay + 0.10 * H))
    if ball_sign > 0:
        x0, x1 = min(W - 2, int(ax + 0.08 * W)), min(W, int(ax + 0.65 * W))
    else:
        x0, x1 = max(0, int(ax - 0.65 * W)), max(2, int(ax - 0.08 * W))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return []
    thr = max(140.0, float(np.percentile(band, 99.5)))
    bw = np.where(band >= thr, 255, 0).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    px_scale = (W * H) / (720 * 1280)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (15 * px_scale <= area <= 800 * px_scale):
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if area / (np.pi * r * r + 1e-6) < 0.1:
            continue
        cx, cy = int(x0 + x), int(y0 + y)
        patch = frame[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].reshape(-1, 3)
        mean_bgr = patch.mean(axis=0)
        if mean_bgr.max() < 140 or (mean_bgr.max() - mean_bgr.min()) > 40:
            continue
        cands.append((float(x0 + x), float(y0 + y), float(r)))
    return cands


def detect_impact_ball(video_path, in_frame, raw, ph, W, H, ball_sign):
    """ボール ROI の画素差分でインパクトを確定する。戻り値 (frame or None, info, ball)。"""
    address, top, candidate = ph["address"], ph["top"], ph["candidate"]
    w0, w1 = ph["window"]
    T = len(raw[NOSE])
    cap = cv2.VideoCapture(str(video_path))
    try:
        fa = _read_frame(cap, in_frame + address)
        if fa is None:
            return None, {"available": False, "reason": "フレームを読めません"}, None
        ankle = ((raw[L_ANKLE][address, 0] + raw[R_ANKLE][address, 0]) / 2 * W,
                 (raw[L_ANKLE][address, 1] + raw[R_ANKLE][address, 1]) / 2 * H)
        cands = detect_ball_candidates(fa, ankle, ball_sign, W, H)
        if not cands:
            ft = _read_frame(cap, in_frame + top)
            if ft is not None:
                cands = detect_ball_candidates(ft, ankle, ball_sign, W, H)
        late = min(T - 1, candidate + int(T * 0.15) + 8)
        gray_a = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
        fl = _read_frame(cap, in_frame + late)
        if fl is None:
            return None, {"available": False, "reason": "フレームを読めません"}, None
        gray_l = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
        ball, gone = None, 0.0
        for x, y, r in cands:   # インパクト後に消えているブロブだけがボール
            k = int(max(10, 2 * r))
            bx0, bx1, by0, by1 = max(0, int(x) - k), min(W, int(x) + k), max(0, int(y) - k), min(H, int(y) + k)
            d = float(cv2.absdiff(gray_a[by0:by1, bx0:bx1], gray_l[by0:by1, bx0:bx1]).mean())
            if d >= 8 and d > gone:
                ball, gone = (x, y, r), d
        if ball is None:
            return None, {"available": False, "reason": f"ボール未検出 (候補 {len(cands)})"}, None
        bx, by, br = ball
        k = int(max(16, 3 * br))
        x0, x1, y0, y1 = max(0, int(bx) - k), min(W, int(bx) + k), max(0, int(by) - k), min(H, int(by) + k)
        base_f = _read_frame(cap, in_frame + w0)
        base = cv2.cvtColor(base_f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        diffs = {}
        for f in range(w0, w1 + 1):
            fr = _read_frame(cap, in_frame + f)
            if fr is None:
                continue
            diffs[f] = float(cv2.absdiff(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), base).mean())
        dmax = max(diffs.values()) if diffs else 0.0
        if dmax < 10:
            return None, {"available": False, "reason": f"ROI 変化が小さい (max {dmax:.1f})"}, ball
        thr = max(10.0, 0.5 * dmax)
        frame = next(f for f in sorted(diffs) if diffs[f] >= thr)
        return frame, {"available": True, "frame": int(frame), "confidence": round(min(1.0, dmax / 40), 2),
                       "detail": f"ROI 差分 max={dmax:.1f}"}, ball
    finally:
        cap.release()


# ---------------------------------------------------------------- 撮影方向

def estimate_camera_angle(tracks, address, W, H):
    sh_l, sh_r = tracks[L_SHOULDER][address] * (W, H), tracks[R_SHOULDER][address] * (W, H)
    hip = (tracks[L_HIP][address] + tracks[R_HIP][address]) / 2 * (W, H)
    sh_mid = (sh_l + sh_r) / 2
    shoulder_w = float(np.linalg.norm(sh_l - sh_r))
    torso = float(np.linalg.norm(sh_mid - hip)) + 1e-6
    ratio = shoulder_w / torso
    if ratio < DTL_MAX_RATIO:
        angle, conf = "dtl", min(1.0, (DTL_MAX_RATIO - ratio) / DTL_MAX_RATIO + 0.5)
    elif ratio > FACE_MIN_RATIO:
        angle, conf = "face_on", min(1.0, (ratio - FACE_MIN_RATIO) / 0.35 + 0.5)
    else:
        angle, conf = "unknown", 0.0
    return {"angle": angle, "ratio": round(ratio, 3), "confidence": round(conf, 2), "source": "auto"}


# ---------------------------------------------------------------- 指標

def _px(tracks, li, f, W, H):
    return np.array([tracks[li][f, 0] * W, tracks[li][f, 1] * H])


def _mid(tracks, a, b, f, W, H):
    return (_px(tracks, a, f, W, H) + _px(tracks, b, f, W, H)) / 2


def spine_angle_deg(tracks, f, W, H, ball_sign):
    """肩中点-腰中点の線が垂直線となす角 (DTL)。ボール側への傾きを正。"""
    sh, hip = _mid(tracks, L_SHOULDER, R_SHOULDER, f, W, H), _mid(tracks, L_HIP, R_HIP, f, W, H)
    return float(np.degrees(np.arctan2((sh[0] - hip[0]) * ball_sign, hip[1] - sh[1])))


def axis_tilt_deg(tracks, f, W, H, target_sign):
    """肩中点-腰中点の左右の傾き (正面)。ターゲットと逆側 (後ろ) に傾くと正。"""
    sh, hip = _mid(tracks, L_SHOULDER, R_SHOULDER, f, W, H), _mid(tracks, L_HIP, R_HIP, f, W, H)
    return float(np.degrees(np.arctan2((hip[0] - sh[0]) * target_sign, hip[1] - sh[1])))


def compute_metrics(tracks, phases, W, H, real_fps, angle, target_sign=1):
    a, t, i = phases["address"], phases["top"], phases["impact"]
    body_h = float(np.linalg.norm(_px(tracks, NOSE, a, W, H) - _mid(tracks, L_ANKLE, R_ANKLE, a, W, H))) + 1e-6
    back_s, down_s = (t - a) / real_fps, (i - t) / real_fps
    m = {"tempo": {"backswing_s": round(back_s, 3), "downswing_s": round(down_s, 3),
                   "ratio": round(back_s / down_s, 2) if down_s > 0 else None},
         "body_h_px": round(body_h, 1)}
    if angle == "dtl":
        sh, hip = _mid(tracks, L_SHOULDER, R_SHOULDER, a, W, H), _mid(tracks, L_HIP, R_HIP, a, W, H)
        ball_sign = 1 if sh[0] >= hip[0] else -1
        spine = {k: spine_angle_deg(tracks, f, W, H, ball_sign) for k, f in (("address", a), ("top", t), ("impact", i))}
        spine["top_minus_address"] = spine["top"] - spine["address"]
        spine["impact_minus_address"] = spine["impact"] - spine["address"]
        nose_a, nose_i = _px(tracks, NOSE, a, W, H), _px(tracks, NOSE, i, W, H)
        hip_i = _mid(tracks, L_HIP, R_HIP, i, W, H)
        m.update({
            "ball_sign": ball_sign,
            "spine_angle_deg": {k: round(v, 1) for k, v in spine.items()},
            "head_movement_pct": {"vertical_pct": round(float((nose_i[1] - nose_a[1]) / body_h * 100), 1),
                                  "forward_pct": round(float((nose_i[0] - nose_a[0]) * ball_sign / body_h * 100), 1)},
            "early_extension_pct": round(float((hip_i[0] - hip[0]) * ball_sign / body_h * 100), 1),
        })
    elif angle == "face_on":
        nose_a, nose_t, nose_i = (_px(tracks, NOSE, f, W, H) for f in (a, t, i))
        hip_a, hip_i = _mid(tracks, L_HIP, R_HIP, a, W, H), _mid(tracks, L_HIP, R_HIP, i, W, H)
        m.update({
            "target_sign": target_sign,
            # 頭の左右: トップでターゲットと逆側 (後ろ) に流れた量 = スウェー、インパクトでターゲット側に動いた量
            "head_sway_pct": {"top_back_pct": round(float((nose_a[0] - nose_t[0]) * target_sign / body_h * 100), 1),
                              "impact_target_pct": round(float((nose_i[0] - nose_a[0]) * target_sign / body_h * 100), 1)},
            "hip_shift_pct": round(float((hip_i[0] - hip_a[0]) * target_sign / body_h * 100), 1),
            "axis_tilt_deg": {"address": round(axis_tilt_deg(tracks, a, W, H, target_sign), 1),
                              "impact": round(axis_tilt_deg(tracks, i, W, H, target_sign), 1)},
        })
    return m


# ---------------------------------------------------------------- コメント (rules.yaml)

def rules_path() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "config" / "rules.yaml"   # exe 同階層 (編集可能)
    return resource_root() / "config" / "rules.yaml"


def metric_values(metrics: dict, angle: str) -> dict:
    vals = {"tempo_ratio": metrics["tempo"]["ratio"]}
    if angle == "dtl" and "spine_angle_deg" in metrics:
        sp, hd = metrics["spine_angle_deg"], metrics["head_movement_pct"]
        vals.update({"spine_angle_address": sp["address"], "spine_angle_top": sp["top"],
                     "spine_angle_impact": sp["impact"], "spine_angle_loss": -sp["impact_minus_address"],
                     "hip_forward_pct": metrics["early_extension_pct"],
                     "head_up_pct": -hd["vertical_pct"], "head_down_pct": hd["vertical_pct"],
                     "head_forward_pct": hd["forward_pct"], "head_back_pct": -hd["forward_pct"]})
    elif angle == "face_on" and "head_sway_pct" in metrics:
        hs = metrics["head_sway_pct"]
        vals.update({"head_sway_top_pct": hs["top_back_pct"], "head_target_impact_pct": hs["impact_target_pct"],
                     "hip_shift_pct": metrics["hip_shift_pct"], "axis_tilt_impact_deg": metrics["axis_tilt_deg"]["impact"]})
    return vals


def generate_comments(metrics: dict, angle: str) -> list:
    try:
        import yaml
    except ImportError:
        log.warning("PyYAML が無いのでコメントを生成しません")
        return []
    path = rules_path()
    if not path.exists():
        return []
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    vals = metric_values(metrics, angle)
    fired = []
    for rule in cfg.get("rules", []):
        if rule.get("angle", "dtl") != angle:
            continue
        try:
            ok = all(_eval(c, vals) for c in rule.get("all", []))
        except KeyError:
            continue
        if ok:
            fired.append((rule.get("priority", 99), rule["comment"].format(**vals)))
    fired.sort(key=lambda t: t[0])
    return [c for _, c in fired[: cfg.get("max_comments", 3)]]


def _eval(cond, vals):
    v = vals[cond["metric"]]
    if v is None:
        return False
    op, ref = cond["op"], cond["value"]
    if op == "between":
        return ref[0] <= v <= ref[1]
    if op == ">=":
        return v >= ref
    if op == ">":
        return v > ref
    if op == "<=":
        return v <= ref
    if op == "<":
        return v < ref
    raise ValueError(f"不明な演算子: {op}")


# ---------------------------------------------------------------- 本体

def analyze(landmarks: dict, W: int, H: int, fps: float, real_fps: float, *,
            video_path=None, in_frame: int = 0, camera_angle: str = "auto",
            dense_points=None) -> dict:
    """骨格から局面・撮影方向・指標・コメントを計算する。landmarks のフレーム番号は相対値。"""
    warnings = []
    frames, raw = build_tracks(landmarks, W, H)
    if len(frames) < 10:
        raise ValueError("骨格フレームが足りません")
    scale = max(1.0, real_fps / 30.0)
    tracks = smooth(raw, scale)
    ph = detect_phases_wrist(tracks, raw, real_fps, scale)

    # ボール側の符号 (DTL): アドレスで肩中点が腰中点より出ている向き
    sh = (tracks[L_SHOULDER][ph["address"]] + tracks[R_SHOULDER][ph["address"]]) / 2
    hip = (tracks[L_HIP][ph["address"]] + tracks[R_HIP][ph["address"]]) / 2
    ball_sign = 1 if sh[0] >= hip[0] else -1

    methods = {"wrist": {"available": True, "frame": ph["candidate"], "confidence": 0.4,
                         "detail": "アドレス時の手の位置への最接近"}}
    ball = None
    if video_path is not None:
        try:
            bf, info, ball = detect_impact_ball(video_path, in_frame, raw, ph, W, H, ball_sign)
            methods["ball_roi"] = info
        except Exception as e:
            methods["ball_roi"] = {"available": False, "reason": f"エラー: {e}"}
    chosen = "ball_roi" if methods.get("ball_roi", {}).get("available") else "wrist"
    impact = int(methods[chosen]["frame"])
    avail = {n: m["frame"] for n, m in methods.items() if m.get("available")}
    if len(avail) >= 2 and max(avail.values()) - min(avail.values()) >= 2:
        warnings.append("インパクト手法間で 2 フレーム以上のずれ: " + ", ".join(f"{n}={f}" for n, f in avail.items()))
    phases_idx = {"address": ph["address"], "top": ph["top"], "impact": impact, "finish": max(ph["finish"], impact + 1)}

    # 撮影方向
    cam = estimate_camera_angle(tracks, ph["address"], W, H)
    if camera_angle in ("dtl", "face_on"):
        cam = {**cam, "angle": camera_angle, "confidence": 1.0, "source": "設定"}
    log.info("camera angle: %s (肩幅/胴長=%.2f, 信頼度 %.2f, %s)", cam["angle"], cam["ratio"], cam["confidence"], cam["source"])
    if cam["angle"] == "unknown":
        warnings.append(f"撮影方向を判定できません (肩幅/胴長={cam['ratio']:.2f}、斜め)。局面とテンポのみ出力")

    # 正面のターゲット方向: 軌道があれば、インパクト後のヘッドの進行方向。無ければ右打ち想定 (+x)
    target_sign = 1
    if dense_points:
        pts = sorted(dense_points, key=lambda p: p[2])
        fi = frames[phases_idx["impact"]]
        after = [p for p in pts if p[2] >= fi]
        before = [p for p in pts if p[2] < fi]
        if after and before:
            target_sign = 1 if after[min(len(after) - 1, 5)][0] >= before[-1][0] else -1

    metrics = compute_metrics(tracks, phases_idx, W, H, real_fps, cam["angle"], target_sign)
    comments = generate_comments(metrics, cam["angle"]) if cam["angle"] != "unknown" else []

    to_frame = lambda i: frames[int(np.clip(i, 0, len(frames) - 1))]  # noqa: E731
    return {
        "phases": {k: to_frame(v) for k, v in phases_idx.items()},
        "phase_idx": phases_idx,
        "impact": {"chosen": chosen, "methods": methods, "ball": ball, "window": ph["window"]},
        "camera": cam,
        "metrics": metrics,
        "comments": comments,
        "tracks": {li: arr * (W, H) for li, arr in tracks.items()},
        "frames": frames,
        "warnings": warnings,
    }
