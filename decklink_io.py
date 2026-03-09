"""
DeckLink入出力モジュール

Blackmagic DeckLink SDKのCOMインターフェースをPythonから利用する。
comtypes.client.GetModuleでDeckLinkAPI64.dllのtype libraryを読み込み、
低レベルvtable呼び出しでデバイス列挙、comtypes高レベルAPIで入出力制御。

rvm-decklink-app (C#) の DeckLinkWrapper.cs を参考に実装。

フォールバック: DeckLinkが利用不可の場合はOpenCVのカメラ入力/ダミー出力に切り替え。
"""

import ctypes
import ctypes.wintypes
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from field_processor import CaptureMode, make_processor, CAPTURE_MODE_EFFECTIVE_FPS

# =============================================================================
# DeckLink SDK COM type library の読み込み
# =============================================================================
DECKLINK_AVAILABLE = False
_dl_mod = None  # comtypes generated module

DECKLINK_TLB_PATH = (
    "C:/Program Files/Blackmagic Design/Blackmagic Desktop Video/DeckLinkAPI64.dll"
)

try:
    import comtypes
    import comtypes.client

    _dl_mod = comtypes.client.GetModule(DECKLINK_TLB_PATH)
    DECKLINK_AVAILABLE = True
    print("[DeckLink] SDK type library loaded OK")
except ImportError:
    print("[DeckLink] comtypes not installed. DeckLink disabled.")
except OSError:
    print(f"[DeckLink] SDK not found at {DECKLINK_TLB_PATH}. DeckLink disabled.")
except Exception as e:
    print(f"[DeckLink] SDK load error: {e}")


# =============================================================================
# COM ポインタ ヘルパー
# =============================================================================
def _vtable_call(com_ptr_void, vtable_index, argtypes, *args):
    """COMオブジェクトのvtableメソッドを直接呼び出す"""
    if isinstance(com_ptr_void, int):
        com_ptr_void = ctypes.c_void_p(com_ptr_void)
    punk = ctypes.cast(com_ptr_void, ctypes.POINTER(ctypes.c_void_p))
    vtable_ptr = punk[0]
    vtable = ctypes.cast(vtable_ptr, ctypes.POINTER(ctypes.c_void_p))

    fn_type = ctypes.CFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
    fn = ctypes.cast(vtable[vtable_index], fn_type)

    this = com_ptr_void if isinstance(com_ptr_void, ctypes.c_void_p) else ctypes.c_void_p(com_ptr_void)
    return fn(this, *args)


def _wrap_com_ptr(raw_ptr, interface):
    """Raw COM ポインタ (int) → comtypes POINTER(Interface) に変換

    comtypes の POINTER(Interface)() は内部に void* を保持する。
    ctypes.byref でその void* のアドレスを取得し、そこに raw_ptr を書き込む。
    これにより comtypes の高レベル API (メソッド呼び出し) が使えるようになる。
    """
    p = ctypes.POINTER(interface)()
    # p の内部ポインタ領域に raw_ptr を書き込み
    ctypes.cast(
        ctypes.byref(p), ctypes.POINTER(ctypes.c_void_p)
    )[0] = raw_ptr
    return p


def _qi_raw(com_ptr_void, iid):
    """QueryInterface を vtable[0] 経由で呼び出し、raw pointer (int) を返す"""
    if isinstance(com_ptr_void, int):
        com_ptr_void = ctypes.c_void_p(com_ptr_void)
    pOut = ctypes.c_void_p(None)
    hr = _vtable_call(
        com_ptr_void, 0,
        [ctypes.POINTER(comtypes.GUID), ctypes.POINTER(ctypes.c_void_p)],
        ctypes.byref(iid),
        ctypes.byref(pOut),
    )
    if hr == 0 and pOut.value:
        return pOut.value
    return None


def _qi(com_ptr_void, interface):
    """QueryInterface を呼び出し、comtypes POINTER(Interface) を返す"""
    raw = _qi_raw(com_ptr_void, interface._iid_)
    if raw:
        return _wrap_com_ptr(raw, interface)
    return None


# =============================================================================
# デバイス情報
# =============================================================================
@dataclass
class DeckLinkDevice:
    index: int
    name: str
    model_name: str = ""
    supports_input: bool = False
    supports_output: bool = False
    _com_obj: object = field(default=None, repr=False)  # comtypes POINTER(IDeckLink)


# =============================================================================
# DeckLink デバイス列挙
# =============================================================================
def enumerate_decklink_devices():
    """利用可能なDeckLinkデバイスを列挙 (rvm-decklink-app準拠)"""
    devices = []

    if not DECKLINK_AVAILABLE:
        return devices

    try:
        # CDeckLinkIterator を生成
        iterator = comtypes.CoCreateInstance(
            _dl_mod.CDeckLinkIterator._reg_clsid_,
            interface=_dl_mod.IDeckLinkIterator,
        )
        iter_ptr = ctypes.cast(iterator, ctypes.c_void_p)

        idx = 0
        while True:
            # IDeckLinkIterator::Next (vtable[3])
            # HRESULT Next([out] IDeckLink** deckLinkInstance)
            ppDevice = ctypes.c_void_p(None)
            hr = _vtable_call(
                iter_ptr, 3,
                [ctypes.POINTER(ctypes.c_void_p)],
                ctypes.byref(ppDevice),
            )

            # S_OK=0 means device found, S_FALSE=1 means no more
            if hr != 0 or ppDevice.value is None:
                break

            raw_ptr = ppDevice.value

            # raw pointer → comtypes IDeckLink
            device_com = _wrap_com_ptr(raw_ptr, _dl_mod.IDeckLink)

            # デバイス名取得
            try:
                display_name = device_com.GetDisplayName()
            except Exception:
                display_name = f"DeckLink Device {idx}"

            try:
                model_name = device_com.GetModelName()
            except Exception:
                model_name = ""

            # QueryInterface で入出力サポートを確認
            has_input = _qi_raw(raw_ptr, _dl_mod.IDeckLinkInput._iid_) is not None
            has_output = _qi_raw(raw_ptr, _dl_mod.IDeckLinkOutput._iid_) is not None

            device = DeckLinkDevice(
                index=idx,
                name=display_name,
                model_name=model_name,
                supports_input=has_input,
                supports_output=has_output,
                _com_obj=device_com,
            )
            devices.append(device)
            print(f"[DeckLink] Device {idx}: {display_name} (model: {model_name}, "
                  f"input={has_input}, output={has_output})")
            idx += 1

    except Exception as e:
        print(f"[DeckLink] デバイス列挙エラー: {e}")

    return devices


# =============================================================================
# DeckLink 入力キャプチャ (COM)
# =============================================================================
class DeckLinkCaptureDevice:
    """DeckLink SDKを使用した入力キャプチャ (rvm-decklink-app準拠)

    IDeckLinkInputCallback を実装し、フレーム到着を受信する。
    UYVY → BGR 変換を行い、OpenCVフレームとして提供する。
    """

    def __init__(self, device_com, width=1920, height=1080, fps=29.97,
                 capture_mode=None):
        """
        device_com:    comtypes POINTER(IDeckLink) デバイスオブジェクト
        capture_mode:  CaptureMode.Normal または CaptureMode.HighFrameRate2x
        """
        self.width = width
        self.height = height
        self.fps = fps
        self._device_com = device_com
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._callback_ref = None
        self._user_callback = None
        self._interlaced = True  # 1080i: デインタレース有効
        self._upper_field_first = True  # 上フィールドが先 (field dominance = 5)

        # フレーム処理モード (通常 / HFR 2x) — Strategy パターン
        self._capture_mode = capture_mode if capture_mode is not None else CaptureMode.Normal
        self._processor = make_processor(self._capture_mode)
        self._input_frame_no = 0  # DeckLink 入力フレーム通番 (ログ用)

        # IDeckLinkInput を QueryInterface で取得
        self._input = _qi(
            ctypes.cast(device_com, ctypes.c_void_p).value,
            _dl_mod.IDeckLinkInput,
        )
        if not self._input:
            raise RuntimeError("IDeckLinkInput not supported on this device")

        # IDeckLinkConfiguration を QueryInterface で取得
        self._config = _qi(
            ctypes.cast(device_com, ctypes.c_void_p).value,
            _dl_mod.IDeckLinkConfiguration,
        )

        print(f"[DeckLink Input] Initialized")

    def start(self, frame_callback=None):
        """キャプチャ開始"""
        self._user_callback = frame_callback

        # SDI入力コネクションを設定 (rvm-decklink-appと同じ)
        if self._config:
            try:
                # bmdDeckLinkConfigVideoInputConnection = 0x7669636E ('vicn')
                # bmdVideoConnectionSDI = 1
                self._config.SetInt(0x7669636E, 1)
                print("[DeckLink Input] SDI input connection set")
            except Exception as e:
                print(f"[DeckLink Input] SDI config warning: {e}")

        # コールバック設定
        self._callback_ref = DeckLinkInputCallbackImpl(self)
        self._input.SetCallback(self._callback_ref)

        # 映像入力を有効化 (1080i59.94, YUV 8bit, フォーマット自動検出)
        # bmdModeHD1080i5994 = 0x48693539
        # bmdFormat8BitYUV = 0x32767579
        # bmdVideoInputEnableFormatDetection = 0x40
        self._input.EnableVideoInput(0x48693539, 0x32767579, 0x40)

        # ストリーム開始
        self._input.StartStreams()
        self._running = True
        print("[DeckLink Input] Capture started (1080i59.94 YUV422 + format detection)")

    @property
    def capture_mode(self):
        return self._capture_mode

    @capture_mode.setter
    def capture_mode(self, mode: CaptureMode):
        """モードを動的に切り替える (キャプチャ中でも可)"""
        self._capture_mode = mode
        self._processor = make_processor(mode)
        print(f"[DeckLink] キャプチャモード変更: {mode.value} "
              f"(実効fps={self.effective_fps:.2f})")

    @property
    def effective_fps(self):
        """実効FPS: 通常モード=bob (fps*2) / 倍速モード=119.88fps"""
        if self._capture_mode == CaptureMode.HighFrameRate2x:
            # Sony HFR 2x: 1入力フレーム → 2独立フレーム → 約119.88fps
            return 119.88
        return self.fps * 2 if self._interlaced else self.fps

    def _deliver_frame(self, bgr):
        """フレームを配信 (プレビュー更新 + コールバック)"""
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))
        with self._lock:
            self._frame = bgr
        if self._user_callback:
            self._user_callback(bgr)

    def _on_frame_arrived(self, frame_data_copy, width, height, row_bytes):
        """フレーム到着コールバック (DeckLinkInputCallbackImpl から呼ばれる)

        frame_data_copy: コピー済みバイト列 (bytes)。コールバック外でも有効。
        インタレース時: フィールド分離で2フレーム配信 → 59.94fps
        プログレッシブ時: そのまま1フレーム配信
        """
        try:
            # UYVY → BGR 変換
            uyvy = np.frombuffer(frame_data_copy, dtype=np.uint8).reshape(height, row_bytes)

            # row_bytes が width*2 より大きい場合はパディングを除去
            expected_row = width * 2
            if row_bytes > expected_row:
                uyvy = uyvy[:, :expected_row]

            uyvy_img = uyvy.reshape(height, width, 2)
            bgr = cv2.cvtColor(uyvy_img, cv2.COLOR_YUV2BGR_UYVY)

            t_proc = time.time()

            if height >= 720 and self._interlaced:
                # 処理モードに応じてフィールド分離
                #   通常モード: bob デインターレース → 59.94fps 相当
                #   倍速モード: フィールド独立展開 (Sony HFR 2x) → 119.88fps 相当
                self._processor.process(
                    bgr, self._upper_field_first,
                    self._deliver_frame, self._input_frame_no,
                )
            else:
                # プログレッシブ: そのまま配信
                self._deliver_frame(bgr)

            proc_ms = (time.time() - t_proc) * 1000
            import logging as _logging
            _log = _logging.getLogger("decklink_capture")
            _log.debug(
                "frame#%d mode=%s proc=%.1fms",
                self._input_frame_no, self._capture_mode.value, proc_ms,
            )
            self._input_frame_no += 1

        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_frame(self):
        """最新フレームを取得"""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """キャプチャ停止"""
        self._running = False
        try:
            if self._input:
                self._input.StopStreams()
                self._input.DisableVideoInput()
                self._input.SetCallback(None)
                print("[DeckLink Input] Capture stopped")
        except Exception as e:
            print(f"[DeckLink Input] Stop error: {e}")


# =============================================================================
# IDeckLinkInputCallback 実装
# =============================================================================
if DECKLINK_AVAILABLE:
    class DeckLinkInputCallbackImpl(comtypes.COMObject):
        """IDeckLinkInputCallback の Python 実装 (rvm-decklink-app準拠)

        C# の DeckLinkWrapper.VideoInputFrameArrived と同じパターン:
        1. IDeckLinkVideoBuffer に QueryInterface
        2. StartAccess(bmdBufferAccessRead) でバッファロック
        3. GetBytes() でポインタ取得
        4. データをコピー (コールバック内でのみ有効)
        5. EndAccess() でバッファ解放
        """
        _com_interfaces_ = [_dl_mod.IDeckLinkInputCallback]

        def __init__(self, capture_device):
            super().__init__()
            self._capture = capture_device
            self._frame_count = 0
            self._no_signal_count = 0

        def VideoInputFormatChanged(self, notificationEvents, newDisplayMode, detectedSignalFlags):
            """入力フォーマット変更通知"""
            try:
                if newDisplayMode:
                    h = newDisplayMode.GetHeight()
                    field_dom = newDisplayMode.GetFieldDominance()
                    # field_dom: 0=unknown, 4=progressive, 5=upperFirst, 6=lowerFirst
                    is_interlaced = field_dom in (5, 6)
                    self._capture._interlaced = is_interlaced
                    if is_interlaced:
                        self._capture._upper_field_first = (field_dom == 5)
                    mode = 'interlaced' if is_interlaced else 'progressive'
                    eff_fps = self._capture.effective_fps
                    print(f"[DeckLink] Format changed: {newDisplayMode.GetWidth()}x{h} "
                          f"{mode} (effective {eff_fps:.2f}fps)")
            except Exception as e:
                print(f"[DeckLink] Format change handler: {e}")
            return 0  # S_OK

        def VideoInputFrameArrived(self, videoFrame, audioPacket):
            """フレーム到着コールバック (rvm-decklink-app準拠)"""
            if videoFrame is None:
                return 0  # S_OK

            try:
                width = videoFrame.GetWidth()
                height = videoFrame.GetHeight()
                row_bytes = videoFrame.GetRowBytes()

                # フレームフラグを確認 (bmdFrameHasNoInputSource = 0x80000000)
                flags = videoFrame.GetFlags()
                if flags & 0x80000000:
                    self._no_signal_count += 1
                    return 0  # 入力信号なし

                # --- rvm-decklink-app 準拠: IDeckLinkVideoBuffer で GetBytes ---
                # videoFrame → IDeckLinkVideoBuffer に QueryInterface
                raw_frame = ctypes.cast(videoFrame, ctypes.c_void_p).value
                buffer = _qi(raw_frame, _dl_mod.IDeckLinkVideoBuffer)
                if buffer is None:
                    print("[DeckLink] QI for IDeckLinkVideoBuffer failed")
                    return 0

                # bmdBufferAccessRead = 1
                buffer.StartAccess(1)
                try:
                    frame_bytes_ptr = buffer.GetBytes()
                    if frame_bytes_ptr:
                        # フレームデータを即座にコピー (C# の Marshal.Copy と同等)
                        buf_size = row_bytes * height
                        frame_copy = (ctypes.c_uint8 * buf_size)()
                        ctypes.memmove(frame_copy, frame_bytes_ptr, buf_size)
                finally:
                    buffer.EndAccess(1)

                if frame_bytes_ptr:
                    self._frame_count += 1
                    if self._frame_count <= 3 or self._frame_count % 300 == 0:
                        print(f"[DeckLink] Frame #{self._frame_count}: "
                              f"{width}x{height} rowBytes={row_bytes}")

                    self._capture._on_frame_arrived(
                        bytes(frame_copy), width, height, row_bytes
                    )

            except Exception as e:
                import traceback
                print(f"[DeckLink] Frame callback error: {e}")
                traceback.print_exc()

            return 0  # S_OK


# =============================================================================
# DeckLink 出力 (COM)
# =============================================================================
class DeckLinkOutputDevice:
    """DeckLink SDKを使用した出力送出 (rvm-decklink-app準拠)

    スケジュール出力を使用し、BGRAフレームをDeckLinkに送出する。
    """

    # タイミング定数 (rvm-decklink-app準拠)
    TIME_SCALE = 60000
    FRAME_DURATION_5994 = 1001   # 59.94fps
    FRAME_DURATION_2997 = 2002   # 29.97fps
    PREROLL_FRAMES = 4
    FRAME_POOL_SIZE = 8

    def __init__(self, device_com, width=1920, height=1080, fps=29.97):
        """
        device_com: comtypes POINTER(IDeckLink) デバイスオブジェクト
        """
        self.width = width
        self.height = height
        self.fps = fps
        self._device_com = device_com
        self._running = False
        self._frame_pool = []
        self._current_frame_index = 0
        self._scheduled_count = 0
        self._frame_duration = self.FRAME_DURATION_2997
        self._lock = threading.Lock()

        # IDeckLinkOutput を QueryInterface で取得
        self._output = _qi(
            ctypes.cast(device_com, ctypes.c_void_p).value,
            _dl_mod.IDeckLinkOutput,
        )
        if not self._output:
            raise RuntimeError("IDeckLinkOutput not supported on this device")

        if fps > 50:
            self._frame_duration = self.FRAME_DURATION_5994

        print(f"[DeckLink Output] Initialized")

    def start(self):
        """出力開始"""
        # 映像出力を有効化
        # bmdModeHD1080i5994 = 0x48693539, bmdVideoOutputFlagDefault = 0
        self._output.EnableVideoOutput(0x48693539, 0)

        # フレームプールを作成
        row_bytes = self.width * 4  # BGRA
        for i in range(self.FRAME_POOL_SIZE):
            # bmdFormat8BitBGRA = 0x42475241, bmdFrameFlagDefault = 0
            frame = self._output.CreateVideoFrame(
                self.width, self.height, row_bytes, 0x42475241, 0
            )
            self._frame_pool.append(frame)
        print(f"[DeckLink Output] Created {self.FRAME_POOL_SIZE} frame pool")

        # プリロール (黒フレームを送出)
        for i in range(self.PREROLL_FRAMES):
            self._schedule_next_frame(None)

        # 再生開始
        self._output.StartScheduledPlayback(0, self.TIME_SCALE, 1.0)
        self._running = True
        print("[DeckLink Output] Playback started")

    def _schedule_next_frame(self, frame_com):
        """フレームをスケジュール"""
        if frame_com is None and self._frame_pool:
            frame_com = self._frame_pool[self._current_frame_index]
            self._current_frame_index = (self._current_frame_index + 1) % self.FRAME_POOL_SIZE

        if frame_com is None:
            return

        display_time = self._scheduled_count * self._frame_duration
        self._output.ScheduleVideoFrame(
            frame_com, display_time, self._frame_duration, self.TIME_SCALE
        )
        self._scheduled_count += 1

    def send_frame(self, bgr_frame):
        """OpenCV BGRフレームをDeckLink出力に送出"""
        if not self._running or not self._frame_pool:
            return

        try:
            # BGR → BGRA 変換
            bgra = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
            if bgra.shape[1] != self.width or bgra.shape[0] != self.height:
                bgra = cv2.resize(bgra, (self.width, self.height))

            frame_data = bgra.tobytes()

            # フレームプールの次のフレームにコピー
            output_frame = self._frame_pool[self._current_frame_index]
            self._current_frame_index = (self._current_frame_index + 1) % self.FRAME_POOL_SIZE

            # IDeckLinkVideoBuffer に QI して GetBytes (rvm-decklink-app準拠)
            # comtypes は IDeckLinkMutableVideoFrame の継承チェーンで
            # GetBytes を解決できないため、明示的に QI が必要
            raw_frame = ctypes.cast(output_frame, ctypes.c_void_p).value
            buffer = _qi(raw_frame, _dl_mod.IDeckLinkVideoBuffer)
            if buffer:
                # bmdBufferAccessWrite = 2
                buffer.StartAccess(2)
                try:
                    dest_ptr = buffer.GetBytes()
                    if dest_ptr:
                        ctypes.memmove(dest_ptr, frame_data, len(frame_data))
                finally:
                    buffer.EndAccess(2)

            # スケジュール
            self._schedule_next_frame(output_frame)

        except Exception as e:
            if not hasattr(self, '_send_error_logged'):
                print(f"[DeckLink Output] send_frame error: {e}")
                self._send_error_logged = True

    def stop(self):
        """出力停止"""
        self._running = False
        try:
            if self._output:
                self._output.StopScheduledPlayback(0, 0)
                self._output.DisableVideoOutput()
                print("[DeckLink Output] Playback stopped")
        except Exception as e:
            print(f"[DeckLink Output] Stop error: {e}")
        self._frame_pool.clear()


# =============================================================================
# フォールバック: OpenCVカメラ入力
# =============================================================================
class OpenCVCaptureDevice:
    """DeckLinkが利用できない場合のフォールバックキャプチャ"""

    def __init__(self, device_index=0, width=1920, height=1080, fps=29.97,
                 test_pattern_only=False):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._running = False
        self._thread = None
        self._frame = None
        self._lock = threading.Lock()
        self._callback = None
        self._test_pattern_only = test_pattern_only

    def start(self, frame_callback=None):
        """キャプチャ開始"""
        self._callback = frame_callback

        if not self._test_pattern_only:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"[Capture] カメラ {self.device_index} を開けません。テストパターンを使用します。")
                self.cap = None
            if self.cap:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            print("[Capture] テストパターンモードで起動")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """キャプチャ停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self):
        """最新フレームを取得"""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self):
        """キャプチャスレッド"""
        frame_duration = 1.0 / self.fps
        while self._running:
            t0 = time.time()

            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                else:
                    frame = self._generate_test_pattern()
            else:
                frame = self._generate_test_pattern()

            with self._lock:
                self._frame = frame

            if self._callback:
                self._callback(frame)

            elapsed = time.time() - t0
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_test_pattern(self):
        """SMPTEカラーバー風のテストパターン"""
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        colors = [
            (192, 192, 192),  # White
            (0, 192, 192),    # Yellow
            (192, 192, 0),    # Cyan
            (0, 192, 0),      # Green
            (192, 0, 192),    # Magenta
            (0, 0, 192),      # Red
            (192, 0, 0),      # Blue
        ]

        bar_w = w // len(colors)
        for i, color in enumerate(colors):
            x1 = i * bar_w
            x2 = (i + 1) * bar_w if i < len(colors) - 1 else w
            frame[:, x1:x2] = color

        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"TEST PATTERN  {ts}", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        return frame


# =============================================================================
# フォールバック: ダミー出力
# =============================================================================
class DummyOutputDevice:
    """DeckLinkが利用できない場合のダミー出力"""

    def __init__(self):
        self._running = False

    def start(self, width=1920, height=1080, fps=29.97):
        self._running = True
        print(f"[Output] ダミー出力開始 ({width}x{height} @ {fps}fps)")

    def send_frame(self, frame):
        pass

    def stop(self):
        self._running = False
        print("[Output] ダミー出力停止")


# =============================================================================
# 統合ラッパー: DeckLink入力 (自動フォールバック付き)
# =============================================================================
class DeckLinkInput:
    """DeckLink SDKを使用した入力キャプチャ (自動フォールバック)

    DeckLinkが起動成功しても一定時間フレームが届かない場合、
    自動的にOpenCVフォールバックに切り替える。
    """

    FRAME_TIMEOUT_SEC = 3.0  # フレーム未到着タイムアウト

    def __init__(self, device_index=0, width=1920, height=1080, fps=29.97,
                 capture_mode=None):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._capture_mode = capture_mode if capture_mode is not None else CaptureMode.Normal
        self._decklink = None
        self._fallback = None
        self._frame_callback = None
        self._decklink_start_time = None
        self._decklink_got_frame = False
        self._fallback_activated = False

    def start(self, frame_callback=None):
        """入力開始。DeckLinkが使えなければフォールバック"""
        self._frame_callback = frame_callback
        devices = enumerate_decklink_devices()
        input_devices = [d for d in devices if d.supports_input]

        if input_devices and self.device_index < len(input_devices):
            dev = input_devices[self.device_index]
            print(f"[DeckLink] 入力デバイス: {dev.name}")
            try:
                self._decklink = DeckLinkCaptureDevice(
                    dev._com_obj, self.width, self.height, self.fps,
                    capture_mode=self._capture_mode,
                )
                self._decklink.start(frame_callback)
                self._decklink_start_time = time.time()
                self._decklink_got_frame = False
                self._fallback_activated = False
                # フレーム監視スレッドを開始
                monitor = threading.Thread(target=self._monitor_frames, daemon=True)
                monitor.start()
                return
            except Exception as e:
                print(f"[DeckLink] 入力開始エラー: {e}. フォールバックモードに切り替え。")
                self._decklink = None

        if not input_devices:
            print("[DeckLink] 入力デバイスが見つかりません。フォールバックモードで起動。")
        self._start_fallback(frame_callback)

    def _monitor_frames(self):
        """DeckLinkフレーム到着を監視し、タイムアウトでフォールバック起動"""
        time.sleep(self.FRAME_TIMEOUT_SEC)
        if self._decklink and not self._fallback_activated:
            frame = self._decklink.get_frame()
            if frame is not None:
                self._decklink_got_frame = True
                print("[DeckLink] フレーム受信確認OK")
            else:
                print(f"[DeckLink] {self.FRAME_TIMEOUT_SEC}秒間フレーム未到着。フォールバック起動。")
                self._start_fallback(self._frame_callback)

    def _start_fallback(self, frame_callback):
        self._fallback_activated = True
        self._fallback = OpenCVCaptureDevice(
            self.device_index, self.width, self.height, self.fps,
            test_pattern_only=True,
        )
        self._fallback.start(frame_callback)

    @property
    def capture_mode(self):
        return self._capture_mode

    @capture_mode.setter
    def capture_mode(self, mode: CaptureMode):
        """モードを動的に切り替える (キャプチャ中でも可)"""
        self._capture_mode = mode
        if self._decklink:
            self._decklink.capture_mode = mode

    @property
    def effective_fps(self):
        """実効FPS"""
        if self._decklink:
            return self._decklink.effective_fps
        return CAPTURE_MODE_EFFECTIVE_FPS.get(self._capture_mode, self.fps)

    def get_frame(self):
        # DeckLinkからフレームが取得できればそれを使う
        if self._decklink:
            frame = self._decklink.get_frame()
            if frame is not None:
                self._decklink_got_frame = True
                return frame
        # DeckLinkフレームがなければフォールバック
        if self._fallback:
            return self._fallback.get_frame()
        return None

    def stop(self):
        if self._decklink:
            self._decklink.stop()
        if self._fallback:
            self._fallback.stop()


# =============================================================================
# 統合ラッパー: DeckLink出力 (自動フォールバック付き)
# =============================================================================
class DeckLinkOutput:
    """DeckLink SDKを使用した出力送出 (自動フォールバック)"""

    def __init__(self, device_index=0, width=1920, height=1080, fps=29.97):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._decklink = None
        self._fallback = None

    def start(self):
        """出力開始"""
        devices = enumerate_decklink_devices()
        output_devices = [d for d in devices if d.supports_output]

        if output_devices and self.device_index < len(output_devices):
            dev = output_devices[self.device_index]
            print(f"[DeckLink] 出力デバイス: {dev.name}")
            try:
                self._decklink = DeckLinkOutputDevice(
                    dev._com_obj, self.width, self.height, self.fps
                )
                self._decklink.start()
                return
            except Exception as e:
                print(f"[DeckLink] 出力開始エラー: {e}. ダミー出力に切り替え。")
                self._decklink = None

        if not output_devices:
            print("[DeckLink] 出力デバイスが見つかりません。ダミー出力モード。")
        self._start_fallback()

    def _start_fallback(self):
        self._fallback = DummyOutputDevice()
        self._fallback.start(self.width, self.height, self.fps)

    def send_frame(self, frame):
        if self._decklink:
            self._decklink.send_frame(frame)
        elif self._fallback:
            self._fallback.send_frame(frame)

    def stop(self):
        if self._decklink:
            self._decklink.stop()
        if self._fallback:
            self._fallback.stop()
