"""ai.club_detector — ONNX Runtime (CPU) によるクラブヘッド検出 (候補の提案)。

モデル: models/clubhead.onnx (tools/train_clubhead.py が出力するヒートマップ回帰ネット)。
入力  : 手首中点を中心にした正方形の切り出し (一辺 = CROP_RATIO × 身長代替値) を INPUT×INPUT に縮小、RGB、0〜1
出力  : (1, 1, INPUT/4, INPUT/4) のヒートマップ。ピーク (score 0〜1) をヘッド候補として返す。

モデルが無い / onnxruntime が無い場合は available() が False を返し、club_tracker は従来方式で動く。
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("club_detector")
INPUT = 256
STRIDE = 4
CROP_RATIO = 2.4
PEAK_MIN = 0.25           # これ未満のピークは候補にしない
MAX_PEAKS = 3


def model_path() -> Path:
    from ai import resource_root
    return resource_root() / "models" / "clubhead.onnx"


class ClubDetector:
    def __init__(self, path=None):
        self.path = Path(path) if path else model_path()
        self.sess = None
        self.input_name = None
        if not self.path.exists():
            return
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            log.info("clubhead.onnx を読み込み: %s", self.path)
        except Exception as e:
            log.warning("clubhead.onnx を読めません (%s)。従来方式で動きます", e)
            self.sess = None

    def available(self) -> bool:
        return self.sess is not None

    @staticmethod
    def crop_box(center, body_h, W, H):
        side = int(round(CROP_RATIO * body_h))
        side = max(32, min(side, W, H))   # 縦位置素材でも正方形がフレーム内に収まるように
        x0 = int(round(center[0] - side / 2))
        y0 = int(round(center[1] - side / 2))
        x0 = max(0, min(x0, W - side)) if W >= side else 0
        y0 = max(0, min(y0, H - side)) if H >= side else 0
        return x0, y0, side

    def detect(self, frame_bgr, wrist_center, body_h):
        """[(x, y, score), ...] を frame_bgr の座標で返す (score 降順、最大 MAX_PEAKS)。"""
        if self.sess is None:
            return []
        H, W = frame_bgr.shape[:2]
        x0, y0, side = self.crop_box(wrist_center, body_h, W, H)
        crop = frame_bgr[y0:y0 + side, x0:x0 + side]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return []
        inp = cv2.resize(crop, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]
        hm = self.sess.run(None, {self.input_name: inp})[0][0, 0]
        return self.peaks(hm, x0, y0, side)

    @staticmethod
    def peaks(hm, x0, y0, side):
        hm = np.asarray(hm, dtype=np.float32)
        h, w = hm.shape
        # 3x3 の極大だけを取る
        dil = cv2.dilate(hm, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero((hm >= dil - 1e-6) & (hm >= PEAK_MIN))
        cands = sorted(((float(hm[y, x]), x, y) for y, x in zip(ys, xs)), reverse=True)[:MAX_PEAKS]
        out = []
        scale = side / (w * 1.0)
        for s, x, y in cands:
            # 近傍の重みで小数位置に
            x_lo, x_hi, y_lo, y_hi = max(0, x - 1), min(w, x + 2), max(0, y - 1), min(h, y + 2)
            patch = hm[y_lo:y_hi, x_lo:x_hi]
            gy, gx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            wsum = float(patch.sum()) + 1e-6
            fx, fy = float((patch * gx).sum() / wsum), float((patch * gy).sum() / wsum)
            out.append((x0 + (fx + 0.5) * scale, y0 + (fy + 0.5) * scale, s))
        return out
