"""
クリップ管理モジュール

録画ファイルのリスト管理、In/Out点設定、トリム保存、軌道データ紐付けを行う。
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class ClipData:
    """1クリップのデータ"""
    id: str                          # ユニークID (タイムスタンプベース)
    source_path: str                 # 元動画ファイルパス
    name: str = ""                   # 表示名
    in_frame: int = 0                # In点 (フレーム番号)
    out_frame: int = -1              # Out点 (-1 = 最終フレーム)
    total_frames: int = 0
    fps: float = 29.97
    width: int = 1920
    height: int = 1080
    duration_sec: float = 0.0
    exported_path: str = ""          # トリム済み出力パス
    trajectory_path: str = ""        # 軌道データJSONパス
    has_trajectory: bool = False
    created_at: str = ""

    def get_out_frame(self):
        """実際のOut点を返す"""
        if self.out_frame < 0:
            return self.total_frames - 1
        return min(self.out_frame, self.total_frames - 1)

    def get_duration_frames(self):
        return self.get_out_frame() - self.in_frame + 1

    def to_dict(self):
        return {
            "id": self.id,
            "source_path": self.source_path,
            "name": self.name,
            "in_frame": self.in_frame,
            "out_frame": self.out_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_sec": self.duration_sec,
            "exported_path": self.exported_path,
            "trajectory_path": self.trajectory_path,
            "has_trajectory": self.has_trajectory,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryData:
    """1スイングの軌道データ"""
    points: list = field(default_factory=list)   # [(x, y, frame_no), ...]
    color_start_hex: str = "#FFFF00"
    color_end_hex: str = "#FF0000"
    thickness: int = 3
    end_frame: int = -1  # この時刻以降は軌跡を描画しない (-1=最後まで描画)
    blur: int = 0        # エッジぼかし量 (0=なし, 最大20)
    fade_frames: int = 0  # フェードイン/アウトのフレーム数 (0=なし)
    alpha: float = 0.85  # 線の不透明度 (0.0=完全透明, 1.0=不透明)


class ClipManager:
    """クリップの管理、保存、読み込み"""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.clips_dir = self.project_dir / "clips"
        self.export_dir = self.project_dir / "exports"
        self.data_file = self.project_dir / "clips.json"
        self.clips: list[ClipData] = []

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self._load()

    def _load(self):
        """JSONからクリップリストを読み込み"""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.clips = [ClipData.from_dict(d) for d in data.get("clips", [])]
                print(f"[ClipManager] {len(self.clips)} クリップを読み込み")
            except Exception as e:
                print(f"[ClipManager] 読み込みエラー: {e}")
                self.clips = []

    def save(self):
        """JSONにクリップリストを保存"""
        data = {"clips": [c.to_dict() for c in self.clips]}
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_clip(self, video_path: str, name: str = "") -> ClipData:
        """動画ファイルからクリップを追加"""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        clip_id = f"clip_{int(time.time() * 1000)}"
        clip = ClipData(
            id=clip_id,
            source_path=str(path.resolve()),
            name=name or path.stem,
            total_frames=total,
            fps=fps or 29.97,
            width=w,
            height=h,
            duration_sec=total / max(fps, 1),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.clips.append(clip)
        self.save()
        print(f"[ClipManager] クリップ追加: {clip.name} ({total} frames, {clip.duration_sec:.1f}s)")
        return clip

    def remove_clip(self, clip_id: str):
        """クリップを削除"""
        self.clips = [c for c in self.clips if c.id != clip_id]
        self.save()

    def get_clip(self, clip_id: str) -> Optional[ClipData]:
        for c in self.clips:
            if c.id == clip_id:
                return c
        return None

    def set_in_out(self, clip_id: str, in_frame: int, out_frame: int):
        """In/Out点を設定"""
        clip = self.get_clip(clip_id)
        if clip:
            clip.in_frame = max(0, in_frame)
            clip.out_frame = min(out_frame, clip.total_frames - 1)
            self.save()

    def export_trimmed(self, clip_id: str) -> Optional[str]:
        """In/Out点でトリムした動画を書き出し"""
        clip = self.get_clip(clip_id)
        if not clip:
            return None

        source = Path(clip.source_path)
        if not source.exists():
            print(f"[ClipManager] ソースが見つかりません: {source}")
            return None

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return None

        in_f = clip.in_frame
        out_f = clip.get_out_frame()

        out_name = f"{clip.name}_trim_{in_f}_{out_f}.mp4"
        out_path = self.export_dir / out_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, clip.fps,
                                  (clip.width, clip.height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, in_f)
        for i in range(in_f, out_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.release()

        clip.exported_path = str(out_path.resolve())
        self.save()
        print(f"[ClipManager] トリム出力: {out_path}")
        return str(out_path)

    def save_trajectory(self, clip_id: str, swings: list):
        """軌道データをJSON保存
        swings: [TrajectoryData, ...]
        """
        clip = self.get_clip(clip_id)
        if not clip:
            return

        traj_path = self.project_dir / f"{clip.id}_trajectory.json"
        data = []
        for swing in swings:
            data.append({
                "points": swing.points,
                "color_start_hex": swing.color_start_hex,
                "color_end_hex": swing.color_end_hex,
                "thickness": swing.thickness,
                "end_frame": getattr(swing, "end_frame", -1),
                "blur": getattr(swing, "blur", 0),
                "fade_frames": getattr(swing, "fade_frames", 0),
                "alpha": getattr(swing, "alpha", 0.85),
            })

        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        clip.trajectory_path = str(traj_path)
        clip.has_trajectory = True
        self.save()

    def load_trajectory(self, clip_id: str) -> list:
        """軌道データを読み込み"""
        clip = self.get_clip(clip_id)
        if not clip or not clip.trajectory_path:
            return []

        traj_path = Path(clip.trajectory_path)
        if not traj_path.exists():
            return []

        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # dict形式 (旧補間モード) と list形式の両方に対応
            if isinstance(data, dict):
                swing_list = data.get("swings", [])
            else:
                swing_list = data

            swings = []
            for d in swing_list:
                td = TrajectoryData(
                    points=[tuple(p) for p in d["points"]],
                    color_start_hex=d.get("color_start_hex", "#FFFF00"),
                    color_end_hex=d.get("color_end_hex", "#FF0000"),
                    thickness=d.get("thickness", 3),
                    end_frame=d.get("end_frame", -1),
                    blur=d.get("blur", 0),
                    fade_frames=d.get("fade_frames", 0),
                    alpha=d.get("alpha", 0.85),
                )
                swings.append(td)
            return swings
        except Exception as e:
            print(f"[ClipManager] 軌道読み込みエラー: {e}")
            return []
