"""ai.trajectory_prior — 観測が無い (信頼度が低い) 区間のヘッド位置を推定して埋める。

考え方
  1. 物理制約: ヘッドは手首からクラブ長 L (アドレス時の手首〜ヘッド距離) 以内にあり、投影で短くは見えるが伸びない。
     シャフト角度 (手首→ヘッドの向き) の変化率には上限がある。
  2. テンプレート (prior): datasets/clubhead の承認済み軌道を撮影方向ごとに正規化して貯めたもの。
     各サンプル = (手首のアドレスからの相対位置 / 身長代替値, 手首の速度の向き, ヘッドの手首からのオフセット / 身長代替値)。
     推定したいフレームの手首状態に近いサンプル k 個の平均を「そのときのヘッドのオフセット」の事前推定にする
     (時間を局面で揃える代わりに、手首の位置と動きで揃える。テンポが違っても使える)。
  3. 観測があるフレームは観測を優先。観測の無いフレームは prior のオフセットを使い、両端の観測との差 (残差) を
     線形に混ぜてつなぎ目を滑らかにする。prior が無ければ両端の観測の間を極座標 (向き・長さ) で補間する。
  4. 最後に長さの上限/下限と角度変化率の上限を掛ける。

保存形式: models/trajectory_prior_<angle>.json
  {"angle", "built", "n_clips", "n_points", "club_len" (身長比), "samples": [[wrx, wry, dirx, diry, hx, hy], ...]}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("trajectory_prior")

MIN_PRIOR_POINTS = 150         # これ未満のサンプルしか無い撮影方向はテンプレート無し扱い (物理制約のみ)
KNN_K = 8
VEL_WEIGHT = 0.7               # KNN 特徴での速度方向の重み (位置は身長比 ≈ 0〜1.5、向きは単位ベクトル)。holdout のマスク評価で 0.35 より良い
MAX_DEG_PER_SEC = 1500.0       # シャフト角度の変化率の上限 (プロのダウンスイングで ~2000°/s。推定では控えめに)
SHORT_GAP_SEC = 0.15           # これより短い欠損は両端の観測の補間だけで埋める (prior は長い欠損にだけ効く)
LONG_GAP_SEC = 0.50            # これ以上の欠損は prior だけ。間は欠損の長さで両者を線形に混ぜる
LEN_MIN_RATIO = 0.25           # 手首〜ヘッド距離の下限 (クラブ長比。投影で短く見える分)
LEN_MAX_RATIO = 1.10           # 上限
EST_STEP_SEC = 0.05            # 推定点を制御点として置く間隔 (実時間)
RESIDUAL_TAU_SEC = 0.15        # 片側しか観測が無いときの残差の減衰時定数


def prior_path(angle: str) -> Path:
    from ai import resource_root
    return resource_root() / "models" / f"trajectory_prior_{angle}.json"


def load_prior(angle: str) -> Optional[dict]:
    p = prior_path(angle)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if len(d.get("samples", [])) < MIN_PRIOR_POINTS:
            return None
        d["_arr"] = np.asarray(d["samples"], dtype=np.float32)
        return d
    except Exception as e:  # pragma: no cover
        log.warning("prior を読めません (%s)", e)
        return None


# ---------------------------------------------------------------- テンプレート作成

def _wrist_from_crop(img_path: Path, crop, crop_size: int, pose):
    """切り出し画像に MediaPipe を掛けて手首中点を元解像度で返す (無ければ None)。pose は vision.PoseLandmarker (IMAGE)。"""
    import cv2
    import mediapipe as mp
    im = cv2.imread(str(img_path))
    if im is None:
        return None
    H, W = im.shape[:2]
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    best = None
    for lm in res.pose_landmarks or []:
        wx, wy = (lm[15].x + lm[16].x) / 2 * W, (lm[15].y + lm[16].y) / 2 * H
        d = abs(wx - W / 2) + abs(wy - H / 2)         # 切り出しは手首中心なので中央に近い人物
        if best is None or d < best[0]:
            best = (d, wx, wy)
    if best is None:
        return None
    x0, y0, side = crop
    s = side / crop_size
    return (x0 + best[1] * s, y0 + best[2] * s)


def build_priors(dataset_root, out_dir=None, min_points: int = MIN_PRIOR_POINTS, write_wrist: bool = True) -> dict:
    """datasets/clubhead から撮影方向ごとのテンプレートを作って models/ に保存する。
    レコードに wrist が無ければ切り出し画像から MediaPipe で求め (labels.json に書き戻す)、次回からはそれを使う。"""
    from ai import resource_root
    out_dir = Path(out_dir) if out_dir else resource_root() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = None
    per_angle: dict = {}
    for lab in sorted(Path(dataset_root).glob("*/labels.json")):
        meta = json.loads(lab.read_text(encoding="utf-8"))
        angle = str(meta.get("camera") or "unknown")
        if angle not in ("dtl", "face_on") or meta.get("excluded"):
            continue
        bh = float(meta.get("body_h") or 0)
        if bh <= 0:
            continue
        real_fps = float(meta.get("real_fps") or 30.0)
        recs = [r for r in meta["points"] if r["source"] in ("manual", "auto_confirmed") and r.get("in_crop", True)]
        recs.sort(key=lambda r: r["frame"])
        changed = False
        for r in recs:
            if r.get("wrist") is None and (lab.parent / r["image"]).exists():
                if pose is None:
                    pose = _make_pose()
                    if pose is None:
                        break
                w = _wrist_from_crop(lab.parent / r["image"], r["crop"], r.get("crop_size", 320), pose)
                r["wrist"] = [round(w[0], 1), round(w[1], 1)] if w else None
                r["wrist_source"] = "mediapipe_crop" if w else "none"
                changed = True
        if changed and write_wrist:
            lab.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        recs = [r for r in recs if r.get("wrist")]
        if len(recs) < 4:
            continue
        w_addr = np.median(np.array([r["wrist"] for r in recs[:3]], dtype=np.float64), axis=0)
        samples = []
        club = []
        for i, r in enumerate(recs):
            w = np.array(r["wrist"], dtype=np.float64)
            p = np.array([r["x"], r["y"]], dtype=np.float64)
            prev_ = recs[max(0, i - 1)]
            next_ = recs[min(len(recs) - 1, i + 1)]
            dt = max(1, next_["frame"] - prev_["frame"])
            v = (np.array(next_["wrist"]) - np.array(prev_["wrist"])) / dt
            n = float(np.linalg.norm(v))
            d = v / n if n > 1e-6 else np.zeros(2)
            off = (p - w) / bh
            if np.linalg.norm(off) > 2.5:          # 手首から 2.5 身長以上離れた点はラベル誤り
                continue
            samples.append([*((w - w_addr) / bh), *d, *off])
            if i < 3:
                club.append(float(np.linalg.norm(off)))
        if samples:
            a = per_angle.setdefault(angle, {"samples": [], "club": [], "clips": 0})
            a["samples"] += samples
            a["club"] += club
            a["clips"] += 1
    if pose is not None:
        pose.close()
    summary = {}
    for angle, a in per_angle.items():
        d = {"angle": angle, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "n_clips": a["clips"], "n_points": len(a["samples"]),
             "club_len": round(float(np.median(a["club"])), 3) if a["club"] else 0.8,
             "enough": len(a["samples"]) >= min_points, "samples": [[round(float(x), 4) for x in s] for s in a["samples"]]}
        (out_dir / f"trajectory_prior_{angle}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        summary[angle] = {k: d[k] for k in ("n_clips", "n_points", "club_len", "enough")}
    return summary


def _make_pose():
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from ai import default_pose_model
        mp_path = default_pose_model()
        if not Path(mp_path).exists():
            return None
        opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(mp_path)),
                                            running_mode=vision.RunningMode.IMAGE, num_poses=3, min_pose_detection_confidence=0.3)
        return vision.PoseLandmarker.create_from_options(opts)
    except Exception as e:  # pragma: no cover
        log.warning("MediaPipe を使えません (%s)", e)
        return None


# ---------------------------------------------------------------- 推定

def _knn_offset(prior: dict, wrist_rel, vel_dir, k: int = KNN_K):
    arr = prior["_arr"]
    q = np.array([wrist_rel[0], wrist_rel[1], vel_dir[0] * VEL_WEIGHT, vel_dir[1] * VEL_WEIGHT], dtype=np.float32)
    feat = np.concatenate([arr[:, 0:2], arr[:, 2:4] * VEL_WEIGHT], axis=1)
    d = np.linalg.norm(feat - q, axis=1)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 0.05)
    off = (arr[idx, 4:6] * w[:, None]).sum(0) / w.sum()
    spread = float(np.sqrt(((arr[idx, 4:6] - off) ** 2).sum(1).mean()))
    return off, spread, float(d[idx].mean())


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fill_trajectory(observed: dict, wrist: dict, body_h: float, real_fps: float, frames, *,
                    prior: Optional[dict] = None, club_len: Optional[float] = None) -> list:
    """観測 {frame: (x, y)} と手首 {frame: (x, y)} から frames の各フレームのヘッド位置を返す。
    戻り値: [(x, y, frame, source), ...]  source = "observed" | "estimated"。手首が無いフレームは飛ばす。
    club_len: 手首〜ヘッド距離 (px)。None なら観測の先頭か prior から決める。"""
    frames = [f for f in frames if f in wrist]
    if not frames:
        return []
    bh = max(float(body_h), 1.0)
    obs_f = sorted(f for f in observed if f in wrist)
    # クラブ長 (px)
    if club_len is None:
        if obs_f:
            first = obs_f[:5]
            club_len = float(np.median([np.hypot(observed[f][0] - wrist[f][0], observed[f][1] - wrist[f][1]) for f in first]))
        elif prior:
            club_len = float(prior.get("club_len", 0.8)) * bh
        else:
            return [(observed[f][0], observed[f][1], f, "observed") for f in obs_f]
    club_len = max(club_len, 0.2 * bh)
    # 手首のアドレス位置 (最初の数フレームの中央値) と速度方向
    first_frames = frames[:max(1, int(round(0.2 * real_fps)))]
    w_addr = np.median(np.array([wrist[f] for f in first_frames], dtype=np.float64), axis=0)

    def vel_dir(f):
        fa, fb = f, f
        for d in range(1, 4):
            if f - d in wrist:
                fa = f - d
                break
        for d in range(1, 4):
            if f + d in wrist:
                fb = f + d
                break
        if fb == fa:
            return np.zeros(2)
        v = (np.array(wrist[fb]) - np.array(wrist[fa])) / (fb - fa)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.zeros(2)

    # prior によるオフセット (身長比) を全フレームで
    e_prior = {}
    if prior is not None:
        for f in frames:
            wr = (np.array(wrist[f]) - w_addr) / bh
            off, spread, _ = _knn_offset(prior, wr, vel_dir(f))
            e_prior[f] = off * bh          # px
    obs_off = {f: np.array(observed[f]) - np.array(wrist[f]) for f in obs_f}

    out = []
    max_dtheta = np.radians(MAX_DEG_PER_SEC) / max(real_fps, 1.0)
    tau = RESIDUAL_TAU_SEC * real_fps
    prev_off = None
    for f in frames:
        w = np.array(wrist[f], dtype=np.float64)
        if f in obs_off:
            off = obs_off[f]
            src = "observed"
        else:
            # 両側の観測
            fa = max((g for g in obs_f if g < f), default=None)
            fb = min((g for g in obs_f if g > f), default=None)
            off_phys = None
            if fa is not None and fb is not None:
                # 物理補間: 向きと長さを別々に線形補間
                oa, ob = obs_off[fa], obs_off[fb]
                ta, tb = np.arctan2(oa[1], oa[0]), np.arctan2(ob[1], ob[0])
                t = (f - fa) / max(fb - fa, 1)
                th = ta + _wrap(tb - ta) * t
                r = np.linalg.norm(oa) * (1 - t) + np.linalg.norm(ob) * t
                off_phys = np.array([np.cos(th), np.sin(th)]) * r
            off_prior = None
            if prior is not None and f in e_prior:
                base = e_prior[f]
                ra = obs_off[fa] - e_prior[fa] if fa is not None and fa in e_prior else None
                rb = obs_off[fb] - e_prior[fb] if fb is not None and fb in e_prior else None
                if ra is not None and rb is not None:
                    t = (f - fa) / max(fb - fa, 1)
                    off_prior = base + (1 - t) * ra + t * rb
                elif ra is not None:
                    off_prior = base + ra * np.exp(-(f - fa) / tau)
                elif rb is not None:
                    off_prior = base + rb * np.exp(-(fb - f) / tau)
                else:
                    off_prior = base
            if off_phys is not None and off_prior is not None:
                # 欠損の長さで混ぜる: 短い欠損は両端補間、長い欠損はテンプレート
                gap_sec = (fb - fa) / max(real_fps, 1.0)
                wp = min(1.0, max(0.0, (gap_sec - SHORT_GAP_SEC) / max(LONG_GAP_SEC - SHORT_GAP_SEC, 1e-6)))
                off = (1 - wp) * off_phys + wp * off_prior
            elif off_prior is not None:
                off = off_prior
            elif off_phys is not None:
                off = off_phys
            elif fa is not None or fb is not None:
                off = obs_off[fa if fa is not None else fb].copy()   # 片側のみ: 直近の観測のオフセットを保持
            else:
                continue
            src = "estimated"
            # 長さの制約
            r = float(np.linalg.norm(off))
            lo, hi = LEN_MIN_RATIO * club_len, LEN_MAX_RATIO * club_len
            if r > 1e-6 and (r < lo or r > hi):
                off = off / r * min(max(r, lo), hi)
            # 角度変化率の制約 (直前の出力に対して)
            if prev_off is not None:
                th0 = np.arctan2(prev_off[1], prev_off[0])
                th1 = np.arctan2(off[1], off[0])
                d = _wrap(th1 - th0)
                if abs(d) > max_dtheta:
                    th1 = th0 + np.sign(d) * max_dtheta
                    off = np.array([np.cos(th1), np.sin(th1)]) * float(np.linalg.norm(off))
        prev_off = off
        head = w + off
        out.append((float(head[0]), float(head[1]), int(f), src))
    return out


def control_points(filled: list, real_fps: float, observed_ctrl: Optional[list] = None, step_sec: float = EST_STEP_SEC) -> list:
    """塗りつぶした軌道から編集用の制御点を作る。観測区間は observed_ctrl (confident_points など) をそのまま使い、
    推定区間は step_sec おきに置く。戻り値 [(x, y, frame, source), ...] フレーム順。"""
    step = max(1, int(round(step_sec * real_fps)))
    obs = {int(p[2]): p for p in (observed_ctrl or []) if len(p) >= 3}
    out = []
    last_est = None
    seq = sorted(filled, key=lambda p: p[2])
    for i, (x, y, f, src) in enumerate(seq):
        if src == "observed":
            if f in obs:
                out.append((int(obs[f][0]), int(obs[f][1]), f, "observed"))
            continue
        nxt_obs = i + 1 < len(seq) and seq[i + 1][3] == "observed"
        prv_obs = i > 0 and seq[i - 1][3] == "observed"
        if last_est is None or f - last_est >= step or nxt_obs or prv_obs:
            out.append((int(round(x)), int(round(y)), f, "estimated"))
            last_est = f
    # 観測区間の制御点が無い (observed_ctrl 未指定) 場合は観測点を step おきに
    if not obs:
        last = None
        extra = []
        for x, y, f, src in seq:
            if src == "observed" and (last is None or f - last >= step):
                extra.append((int(round(x)), int(round(y)), f, "observed"))
                last = f
        out = sorted(out + extra, key=lambda p: p[2])
    return out


def swing_frame_range(landmarks: dict, W: int, H: int, real_fps: float):
    """骨格からアドレス〜フィニッシュのフレーム範囲 (相対フレーム) を返す。失敗したら None。"""
    try:
        from ai.swing_analyzer import build_tracks, smooth, detect_phases_wrist
        frames, raw = build_tracks(landmarks, W, H)
        if len(frames) < 10:
            return None
        scale = max(1.0, real_fps / 30.0)
        tracks = smooth(raw, scale)
        ph = detect_phases_wrist(tracks, raw, real_fps, scale)
        return int(frames[ph["address"]]), int(frames[max(ph["finish"], ph["top"] + 1)])
    except Exception as e:
        log.debug("swing range failed: %s", e)
        return None


def estimate_camera_from_landmarks(landmarks: dict, W: int, H: int, real_fps: float) -> str:
    try:
        from ai.swing_analyzer import build_tracks, smooth, detect_phases_wrist, estimate_camera_angle
        frames, raw = build_tracks(landmarks, W, H)
        if len(frames) < 10:
            return "unknown"
        scale = max(1.0, real_fps / 30.0)
        tracks = smooth(raw, scale)
        ph = detect_phases_wrist(tracks, raw, real_fps, scale)
        return estimate_camera_angle(tracks, ph["address"], W, H, real_fps).get("angle", "unknown")
    except Exception:
        return "unknown"
