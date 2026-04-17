"""
送出エンジン

クリップリストの再生、DeckLink出力への送出、トランジション制御を行う。

スレッド設計:
  - 再生中: _play_loop スレッドが独自cap (reader_fn) を排他使用
  - 停止中: メインスレッドが self._cap を排他使用 (cue/seek)
  - stop() / cue() / play() は全て非ブロッキング (GUIをフリーズさせない)
  - 世代カウンタ (_gen) で古いスレッドが新操作に干渉しないよう保護
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from clip_manager import ClipData, TrajectoryData
from trajectory import render_trajectory_on_frame, TimedSpline

log = logging.getLogger("playout")


class PlayoutItem:
    """送出リストの1アイテム"""

    def __init__(self, clip: ClipData, swings: list = None):
        self.clip = clip
        self.swings = swings or []


class PlayoutEngine:
    """送出エンジン"""

    def __init__(self, output_device=None):
        self.output_device = output_device
        self.playlist: list[PlayoutItem] = []
        self.current_index = 0
        self.speed = 1.0  # 再生速度 (1.0=通常, 0.5=1/2, 0.25=1/4, 0.125=1/8)

        self._playing = False
        self._paused = False
        self._thread = None
        self._gen = 0             # 世代カウンタ (古いスレッドの干渉防止)
        self._current_frame_no = 0
        self._cap = None
        self._lock = threading.Lock()
        self._preview_frame = None

        # コールバック
        self.on_frame_update = None     # (frame, frame_no, total) 呼ばれる
        self.on_clip_changed = None     # (index, clip) 呼ばれる
        self.on_playback_ended = None   # () 呼ばれる

    @property
    def is_playing(self):
        return self._playing and not self._paused

    @property
    def is_paused(self):
        return self._paused

    @property
    def current_frame_no(self):
        return self._current_frame_no

    @property
    def preview_frame(self):
        with self._lock:
            return self._preview_frame.copy() if self._preview_frame is not None else None

    def add_item(self, clip: ClipData, swings: list = None):
        """送出リストにアイテム追加"""
        self.playlist.append(PlayoutItem(clip, swings))

    def remove_item(self, index: int):
        if 0 <= index < len(self.playlist):
            self.playlist.pop(index)

    def move_item(self, from_idx: int, to_idx: int):
        if 0 <= from_idx < len(self.playlist) and 0 <= to_idx < len(self.playlist):
            item = self.playlist.pop(from_idx)
            self.playlist.insert(to_idx, item)

    # ----- 再生スレッド管理 -----

    def _wait_thread(self):
        """前回の再生スレッドを停止通知してデタッチ (GUIをブロックしない)

        世代カウンタを進めることで、古いスレッドが状態を書き換えるのを防ぐ。
        古いスレッド自身のクリーンアップ (reader join 等) はバックグラウンドで完了する。
        """
        self._gen += 1
        self._playing = False
        old = self._thread
        self._thread = None
        if old and old.is_alive():
            log.debug("_wait_thread: detaching old thread (non-blocking)")
            def _bg_join():
                old.join(timeout=3.0)
                if old.is_alive():
                    log.warning("_wait_thread: old thread still alive after 3.0s")
            threading.Thread(target=_bg_join, daemon=True).start()

    # ----- 公開API -----

    def cue(self, index: int):
        """指定インデックスのクリップをキュー (停止してからキュー)"""
        if not (0 <= index < len(self.playlist)):
            return
        log.debug(f"cue({index})")
        self._playing = False
        self._paused = False
        self._wait_thread()
        self.current_index = index
        self._open_cap(index)
        self._read_preview_at_current()

    def get_cued_total_frames(self):
        """キュー中クリップの総フレーム数"""
        if 0 <= self.current_index < len(self.playlist):
            return self.playlist[self.current_index].clip.get_duration_frames()
        return 0

    def seek_to(self, frame_offset):
        """キュー中クリップ内の相対フレーム位置にシーク (停止中 or 一時停止中)"""
        if self._playing and not self._paused:
            log.debug(f"seek_to: blocked (playing={self._playing}, paused={self._paused})")
            return
        if not (0 <= self.current_index < len(self.playlist)):
            return
        clip = self.playlist[self.current_index].clip
        abs_frame = clip.in_frame + frame_offset
        abs_frame = max(clip.in_frame, min(abs_frame, clip.get_out_frame()))
        log.debug(f"seek_to: offset={frame_offset} -> abs={abs_frame} (in={clip.in_frame}, out={clip.get_out_frame()})")
        self._current_frame_no = abs_frame
        self._read_preview_at_current()

    def play(self):
        """再生開始"""
        if not self.playlist:
            return

        # current_indexが範囲外なら再生不可
        if self.current_index >= len(self.playlist):
            log.debug(f"play: index {self.current_index} out of range, ignoring")
            return

        if self._paused:
            self._paused = False
            if self._thread and self._thread.is_alive():
                log.debug("play: resume from pause (thread alive)")
                return
            log.debug("play: thread dead after pause, restarting")

        # 前回のスレッドが残っていれば待つ
        self._wait_thread()

        # クリップ末尾に達している場合は再生しない (CUEで巻き戻す)
        clip = self.playlist[self.current_index].clip
        out_frame = clip.get_out_frame()
        if self._current_frame_no >= out_frame:
            log.debug(f"play: at end ({self._current_frame_no}>={out_frame}), ignoring (use CUE to rewind)")
            return

        log.debug(f"play: starting from index={self.current_index} frame={self._current_frame_no} gen={self._gen}")
        self._playing = True
        self._paused = False
        gen = self._gen
        self._thread = threading.Thread(target=self._play_loop, args=(gen,), daemon=True)
        self._thread.start()

    def pause(self):
        """一時停止"""
        self._paused = True

    def stop(self):
        """停止 (非ブロッキング — GUIフリーズ防止)"""
        log.debug("stop")
        self._playing = False
        self._paused = False
        # スレッドのjoinはしない。スレッドは_playing=Falseで自然終了する。
        # capのクローズもスレッド側に任せる。

    def next_clip(self):
        """次のクリップへ"""
        if self.current_index < len(self.playlist) - 1:
            self._paused = False
            self._wait_thread()
            self.current_index += 1
            self._open_cap(self.current_index)
            self._read_preview_at_current()

    def prev_clip(self):
        """前のクリップへ"""
        if self.current_index > 0:
            self._paused = False
            self._wait_thread()
            self.current_index -= 1
            self._open_cap(self.current_index)
            self._read_preview_at_current()

    # ----- 内部: cap管理 -----

    def _open_cap(self, index):
        """クリップのVideoCaptureを開く (メインスレッド用)"""
        self._close_cap()
        item = self.playlist[index]
        clip = item.clip
        path = clip.exported_path if clip.exported_path else clip.source_path

        log.debug(f"_open_cap: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            log.error(f"_open_cap: 開けません: {path}")
            self._cap = None
            return

        self._current_frame_no = clip.in_frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, clip.in_frame)

    def _close_cap(self):
        if self._cap:
            log.debug("_close_cap")
            self._cap.release()
            self._cap = None

    def _read_preview_at_current(self):
        """現在フレーム位置のプレビューを読み込み (メインスレッド用)
        DeckLink出力にもフレームを送出する。"""
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_no)
        ret, frame = self._cap.read()
        if ret:
            item = self.playlist[self.current_index]
            if item.swings:
                adjusted = self._current_frame_no - item.clip.in_frame
                render_trajectory_on_frame(frame, item.swings, adjusted)
            # DeckLink出力 (一時停止中/フレーム送り時もモニター出力)
            if self.output_device:
                clip = item.clip
                spd = max(self.speed, 0.01)
                tu = max(round(60000.0 / max(clip.fps, 1) / spd), 1001)
                self.output_device.send_frame(frame, frame_duration_tu=tu)
            with self._lock:
                self._preview_frame = frame
            if self.on_frame_update:
                clip = item.clip
                total = clip.get_duration_frames()
                offset = self._current_frame_no - clip.in_frame
                self.on_frame_update(frame, offset, total)
        # 読み取りで1フレーム進むので位置を戻す
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_no)

    # ----- フレーム先読みスレッド -----

    @staticmethod
    def _reader_fn(path, start_frame, out_frame, swings, clip_in_frame,
                   stop_event, q):
        """フレーム先読みスレッド: cap.read + 軌道描画をメインループから分離

        独自の VideoCapture を開くため、メインスレッドと排他制御不要。
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log.error(f"_reader_fn: cannot open {path}")
            try:
                q.put(None)
            except Exception:
                pass
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        pos = start_frame
        frames_read = 0
        while not stop_event.is_set() and pos <= out_frame:
            ret, frame = cap.read()
            if not ret:
                log.debug(f"_reader_fn: read failed at pos={pos}")
                break
            if swings:
                render_trajectory_on_frame(frame, swings, pos - clip_in_frame)
            try:
                q.put((frame, pos), timeout=0.2)
            except queue.Full:
                if stop_event.is_set():
                    break
                continue
            pos += 1
            frames_read += 1
        cap.release()
        log.debug(f"_reader_fn: done, read {frames_read} frames ({start_frame}..{pos-1}), stopped={stop_event.is_set()}")
        # 停止要求されていない場合のみsentinelを送信
        # (停止された古いリーダーのsentinelが新リーダーのキューを汚染するのを防ぐ)
        if not stop_event.is_set():
            try:
                q.put(None)  # sentinel: 読み取り完了
            except Exception:
                pass

    # ----- 再生ループ (別スレッド) -----

    def _play_loop(self, gen):
        """再生ループ — フレーム先読み + DeckLinkプリスケジュールで安定出力

        gen: この再生が属する世代。_gen != gen になったら即座に終了する。
        """
        log.debug(f"_play_loop: start (gen={gen})")

        # Windows: タイマー分解能を1msに設定 (デフォルト15.625ms → time.sleep精度が大幅改善)
        _timer_period_set = False
        try:
            import ctypes as _ctypes
            _ctypes.windll.winmm.timeBeginPeriod(1)
            _timer_period_set = True
        except Exception:
            pass

        reader_thread = None
        reader_stop = threading.Event()

        try:
            while self._playing and self._gen == gen and self.current_index < len(self.playlist):
                item = self.playlist[self.current_index]
                clip = item.clip
                path = clip.exported_path if clip.exported_path else clip.source_path

                out_frame = clip.get_out_frame()
                base_frame_duration = 1.0 / max(clip.fps, 1)

                if self.on_clip_changed:
                    self.on_clip_changed(self.current_index, clip)

                # --- フレーム先読みスレッド起動 ---
                QUEUE_SIZE = 8
                PRE_SCHEDULE = 4  # DeckLinkパイプラインを埋める初期フレーム数
                frame_q = queue.Queue(maxsize=QUEUE_SIZE)

                def start_reader(start_frame):
                    nonlocal reader_stop
                    # 新しいリーダーには新しいイベントを使用
                    # (古いリーダーのイベントはset済みのまま → sentinelを送信しない)
                    reader_stop = threading.Event()
                    while not frame_q.empty():
                        try:
                            frame_q.get_nowait()
                        except queue.Empty:
                            break
                    t = threading.Thread(
                        target=self._reader_fn,
                        args=(path, start_frame, out_frame,
                              item.swings, clip.in_frame,
                              reader_stop, frame_q),
                        daemon=True,
                    )
                    t.start()
                    return t

                reader_thread = start_reader(self._current_frame_no)

                frames_sent = 0
                was_paused = False
                gui_last_update = 0.0
                GUI_INTERVAL = 1.0 / 15  # GUIプレビューは最大15fps (SDI出力は別)

                while self._playing and self._gen == gen:
                    # --- 一時停止 ---
                    if self._paused:
                        was_paused = True
                        time.sleep(0.05)
                        continue

                    # --- 一時停止から復帰: リーダーを再起動 ---
                    if was_paused:
                        log.debug(f"_play_loop: resume from pause, restarting reader at frame={self._current_frame_no} "
                                  f"(speed={self.speed}, fps={clip.fps}, out={out_frame}, output_dev={bool(self.output_device)})")
                        reader_stop.set()
                        if reader_thread:
                            reader_thread.join(timeout=1.0)
                        reader_thread = start_reader(self._current_frame_no)
                        was_paused = False
                        frames_sent = 0

                    # --- フレーム取得 ---
                    try:
                        entry = frame_q.get(timeout=0.2)
                    except queue.Empty:
                        if not self._playing or self._gen != gen:
                            break
                        continue
                    if entry is None:  # sentinel: 読み取り完了
                        log.debug(f"_play_loop: sentinel received, frames_sent={frames_sent}")
                        break

                    frame, frame_no = entry
                    t0 = time.time()

                    self._current_frame_no = frame_no

                    # フレーム表示間隔の計算 (DeckLink + スリープ共通)
                    spd = max(self.speed, 0.01)
                    tu = max(round(60000.0 / max(clip.fps, 1) / spd), 1001)

                    # DeckLink出力
                    if self.output_device:
                        self.output_device.send_frame(frame, frame_duration_tu=tu)

                    # プレビュー保存
                    with self._lock:
                        self._preview_frame = frame

                    # GUI更新 (間引き: 30fps上限)
                    now = time.time()
                    if self.on_frame_update and (now - gui_last_update) >= GUI_INTERVAL:
                        total = clip.get_duration_frames()
                        self.on_frame_update(
                            frame, frame_no - clip.in_frame, total)
                        gui_last_update = now

                    self._current_frame_no = frame_no + 1
                    frames_sent += 1

                    # プリスケジュール: 最初の数フレームはスリープなしで
                    # DeckLinkのスケジュールバッファを満たす
                    if frames_sent <= PRE_SCHEDULE and self.output_device:
                        continue

                    # フレーム間タイミング制御
                    # DeckLink出力時: スケジュール間隔 (tu/TIME_SCALE) に合わせる
                    # → HFRクリップでも自動的にスローモーション再生
                    # → フレームプール枯渇を防止 (読み込みが表示を追い越さない)
                    if self.output_device:
                        frame_duration = tu / 60000.0
                    else:
                        frame_duration = base_frame_duration / spd
                    elapsed = time.time() - t0
                    sleep_time = frame_duration - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # --- リーダースレッド停止 ---
                reader_stop.set()
                if reader_thread and reader_thread.is_alive():
                    reader_thread.join(timeout=1.0)
                    reader_thread = None

                # 世代が変わっていたら (新しい cue/play が来た)、状態を触らず終了
                if self._gen != gen:
                    break

                # 最終フレームのGUI更新 (間引きで未送信の場合)
                if self.on_frame_update:
                    total = clip.get_duration_frames()
                    offset = min(
                        max(self._current_frame_no - 1 - clip.in_frame, 0),
                        total - 1)
                    pf = self.preview_frame
                    if pf is not None:
                        self.on_frame_update(pf, offset, total)

                # クリップ末尾で停止 (自動進行しない — 放送用途)
                self._current_frame_no = clip.get_out_frame()
                break

        except Exception as e:
            log.exception(f"_play_loop: exception: {e}")
        finally:
            reader_stop.set()
            if reader_thread and reader_thread.is_alive():
                reader_thread.join(timeout=1.0)
            if _timer_period_set:
                try:
                    import ctypes as _ctypes
                    _ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
            log.debug("_play_loop: end")

        # 自分の世代がまだ有効な場合のみ状態を更新
        # (新しい cue/play が来ていたら上書きしない)
        if self._gen == gen:
            self._playing = False
            self._paused = False
            if self.on_playback_ended:
                self.on_playback_ended()

    # ----- 永続化 -----
    def save_playlist(self, path):
        """送出リストをJSONに保存"""
        data = []
        for item in self.playlist:
            data.append({
                "clip": item.clip.to_dict(),
                "swings": [
                    {"points": s.points,
                     "color_start_hex": s.color_start_hex,
                     "color_end_hex": s.color_end_hex,
                     "thickness": s.thickness,
                     "end_frame": getattr(s, "end_frame", -1)}
                    for s in item.swings
                ],
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_playlist(self, path):
        """送出リストをJSONから読み込み"""
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                clip = ClipData.from_dict(entry["clip"])
                # ソースファイルが存在しなければスキップ
                if not Path(clip.source_path).exists():
                    log.warning(f"ファイルなしスキップ: {clip.source_path}")
                    continue
                swings = []
                for s in entry.get("swings", []):
                    swings.append(TrajectoryData(
                        points=[tuple(pt) for pt in s.get("points", [])],
                        color_start_hex=s.get("color_start_hex", "#FFFF00"),
                        color_end_hex=s.get("color_end_hex", "#FF0000"),
                        thickness=s.get("thickness", 3),
                        end_frame=s.get("end_frame", -1),
                    ))
                self.playlist.append(PlayoutItem(clip, swings))
            log.info(f"{len(self.playlist)}件 読み込み")
        except Exception as e:
            log.error(f"読み込みエラー: {e}")

    def scan_directory(self, directory):
        """ディレクトリ内のmp4ファイルをスキャンし、未登録のものを追加。
        実ファイルが存在しないエントリは除去する。"""
        d = Path(directory)
        if not d.is_dir():
            return 0
        # 実ファイルが存在しないエントリを除去
        before = len(self.playlist)
        self.playlist = [item for item in self.playlist
                         if Path(item.clip.source_path).exists()]
        removed = before - len(self.playlist)
        if removed:
            log.info(f"ファイルなし {removed} 件除去")
        # 既存のパスセット
        existing = set()
        for item in self.playlist:
            existing.add(str(Path(item.clip.source_path).resolve()))
        added = 0
        for mp4 in sorted(d.glob("*.mp4")):
            if str(mp4.resolve()) in existing:
                continue
            # mp4からメタデータを取得
            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                log.warning(f"スキャン: 開けません: {mp4.name}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if total <= 0:
                log.warning(f"スキャン: 0フレーム: {mp4.name}")
                continue
            clip = ClipData(
                id=f"scan_{mp4.stem}",
                source_path=str(mp4),
                name=mp4.stem,
                total_frames=total,
                fps=fps,
                width=w,
                height=h,
                duration_sec=total / fps,
            )
            self.playlist.append(PlayoutItem(clip, []))
            added += 1
        if added:
            log.info(f"ディレクトリスキャンで {added} 件追加")
        return added
