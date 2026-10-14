"""label_club — クラブヘッド位置の正解ラベル付けツール。

動画をフレーム送りし、主要フレーム(アドレス / テイクバック途中 / トップ /
ダウンスイング途中 / インパクト / フォロー)でクラブヘッドの中心を左クリックして記録する。
ラベルは tests/club_labels.json に保存され、tools/eval_club.py が精度評価に使う。

使い方:
    python tools/label_club.py <動画> [他の動画...] [--frames 14,33,52,58,63,70]

    --frames を付けると、そのフレームに順番にジャンプする(省略時は手動で移動)。

キー操作:
    d / →      : 1 フレーム進む          a / ←      : 1 フレーム戻る
    w / ↑      : 10 フレーム進む         s / ↓      : 10 フレーム戻る
    n          : --frames の次のフレームへ
    1〜6       : 局面タグ (1=address 2=takeaway 3=top 4=downswing 5=impact 6=follow)
    左クリック : 現在フレームのヘッド位置を記録 (クリック位置が 2 倍拡大で表示される)
    z          : 拡大表示 (最後のクリック位置を中心に 3 倍) の ON/OFF
    u          : 現在フレームのラベルを削除
    Enter / Space : 保存して次の動画へ       q / Esc : 保存せず終了

ラベル形式 (tests/club_labels.json):
    {
      "sample_30fps.mp4": {
        "video": "C:/.../sample_30fps.mp4",
        "width": 720, "height": 1280, "fps": 30.0,
        "frames": {"14": {"x": 410, "y": 1130, "phase": "address"}, ...}
      }
    }
座標は元解像度のピクセル。フレーム番号は動画のフレーム index (0 始まり)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "tests" / "club_labels.json"
PHASES = {ord("1"): "address", ord("2"): "takeaway", ord("3"): "top",
          ord("4"): "downswing", ord("5"): "impact", ord("6"): "follow"}
KEY_STEP = {ord("d"): 1, 2555904: 1, ord("a"): -1, 2424832: -1,
            ord("w"): 10, 2490368: 10, ord("s"): -10, 2621440: -10}
KEY_SAVE = (13, 32)
KEY_QUIT = (ord("q"), 27)
MAX_DISPLAY_H = 900


def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}


def save_labels(labels: dict) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                           encoding="utf-8")


class Labeler:
    def __init__(self, video: Path, entry: dict, jump_frames: list[int]):
        self.video = video
        self.entry = entry
        self.frames: dict = entry.setdefault("frames", {})
        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        entry.update({"video": str(video), "width": self.W, "height": self.H,
                      "fps": self.cap.get(cv2.CAP_PROP_FPS)})
        self.jump = jump_frames
        self.jump_idx = 0
        self.cur = jump_frames[0] if jump_frames else 0
        self.phase = "address"
        self.zoom = False
        self.zoom_center = (self.W // 2, self.H // 2)
        self.scale = min(1.0, MAX_DISPLAY_H / self.H)
        self.win = "label_club  (click=head  1-6=phase  Enter=save  q=quit)"
        self.frame = None

    # ---- 座標変換 (表示 → 元画像)
    def to_image(self, dx: int, dy: int) -> tuple[int, int]:
        if self.zoom:
            x0, y0, zs = self._zoom_rect()
            return int(x0 + dx / zs), int(y0 + dy / zs)
        return int(dx / self.scale), int(dy / self.scale)

    def _zoom_rect(self):
        zs = 3.0 * self.scale
        vw, vh = int(self.W * self.scale / zs), int(self.H * self.scale / zs)
        cx, cy = self.zoom_center
        x0 = max(0, min(self.W - vw, cx - vw // 2))
        y0 = max(0, min(self.H - vh, cy - vh // 2))
        return x0, y0, zs

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.to_image(x, y)
            self.frames[str(self.cur)] = {"x": ix, "y": iy, "phase": self.phase}
            self.zoom_center = (ix, iy)
            print(f"  f{self.cur}: ({ix}, {iy}) {self.phase}")
            self.draw()

    def read(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur)
        ok, fr = self.cap.read()
        self.frame = fr if ok else None

    def draw(self):
        if self.frame is None:
            return
        img = self.frame.copy()
        lab = self.frames.get(str(self.cur))
        for f, l in self.frames.items():  # 他フレームのラベルは薄く
            if f != str(self.cur):
                cv2.circle(img, (l["x"], l["y"]), 4, (120, 120, 120), 1, cv2.LINE_AA)
        if lab:
            cv2.drawMarker(img, (lab["x"], lab["y"]), (0, 0, 255),
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            cv2.circle(img, (lab["x"], lab["y"]), 10, (0, 255, 255), 1, cv2.LINE_AA)
        if self.zoom:
            x0, y0, zs = self._zoom_rect()
            vw, vh = int(self.W * self.scale / zs), int(self.H * self.scale / zs)
            img = cv2.resize(img[y0:y0 + vh, x0:x0 + vw], None, fx=zs, fy=zs,
                             interpolation=cv2.INTER_CUBIC)
        elif self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        txt = (f"{self.video.name}  f{self.cur}/{self.total - 1}  phase={self.phase}"
               f"  label={'(%d,%d)' % (lab['x'], lab['y']) if lab else '-'}"
               f"  {'ZOOM' if self.zoom else ''}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.win, img)

    def run(self) -> bool:
        """True=保存, False=破棄"""
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self.on_mouse)
        try:
            while True:
                self.cur = max(0, min(self.total - 1, self.cur))
                self.read()
                self.draw()
                k = cv2.waitKeyEx(0)
                if k in KEY_STEP:
                    self.cur += KEY_STEP[k]
                elif k in PHASES:
                    self.phase = PHASES[k]
                    if str(self.cur) in self.frames:
                        self.frames[str(self.cur)]["phase"] = self.phase
                elif k == ord("n") and self.jump:
                    self.jump_idx = (self.jump_idx + 1) % len(self.jump)
                    self.cur = self.jump[self.jump_idx]
                elif k == ord("z"):
                    self.zoom = not self.zoom
                elif k == ord("u"):
                    self.frames.pop(str(self.cur), None)
                elif k in KEY_SAVE:
                    return True
                elif k in KEY_QUIT or k == -1:
                    return False
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main() -> int:
    args = sys.argv[1:]
    jump: list[int] = []
    if "--frames" in args:
        i = args.index("--frames")
        jump = [int(x) for x in args[i + 1].split(",")]
        del args[i:i + 2]
    videos = [Path(a) for a in args if Path(a).exists()]
    if not videos:
        print(__doc__)
        return 1
    labels = load_labels()
    for v in videos:
        print(f"=== {v.name} ===")
        entry = labels.setdefault(v.name, {})
        if Labeler(v, entry, jump).run():
            save_labels(labels)
            print(f"  保存: {len(entry['frames'])} 点 -> {LABELS_PATH}")
        else:
            print("  破棄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
