"""
ShuttlePRO v2 HID ドライバ

Contour Design ShuttlePRO v2 をPythonから読み取り、
ジョグ・シャトル・ボタンイベントをコールバックで通知する。

自動フォーマット検出:
  初回レポートのバイトパターンを分析し、バイト位置を自動判定。
  シャトル値の範囲検証 (-7~+7) によりフォーマットを確認する。

既知フォーマット:
  A) [shuttle, jog, btn_lo, btn_hi, ...]   (Linux/標準)
  B) [btn_lo, btn_hi, pad, shuttle, jog]    (一部ドライバ)
  C) [reportID, shuttle, jog, btn_lo, btn_hi] (report ID付き)
"""

import logging
import threading
import time

log = logging.getLogger("shuttle")

VENDOR_ID = 0x0B33
PRODUCT_ID_PRO_V2 = 0x0030

# 既知フォーマット定義
FORMATS = {
    "A":   {"shuttle": 0, "jog": 1, "btn_start": 2, "btn_len": 2},
    "B":   {"shuttle": 3, "jog": 4, "btn_start": 0, "btn_len": 2},
    "C":   {"shuttle": 1, "jog": 2, "btn_start": 3, "btn_len": 2},
}


class ShuttlePRO:
    """ShuttlePRO v2 HID ドライバ (自動フォーマット検出)"""

    def __init__(self, format_name=None):
        self._dev = None
        self._thread = None
        self._running = False

        # バイト位置 (デフォルト: Format A)
        self._idx_shuttle = 0
        self._idx_jog = 1
        self._idx_btn_start = 2
        self._idx_btn_len = 2
        self._format_name = "A"
        self._format_locked = format_name is not None

        if format_name and format_name in FORMATS:
            self._apply_format(format_name)

        # 前回値
        self._prev_jog = None
        self._prev_shuttle = 0
        self._prev_buttons = 0
        self._prev_data = None
        self._report_count = 0
        self._initialized = False

        # フォーマット検証
        self._shuttle_out_of_range_count = 0

        # ボタンデバウンス (ゴーストフィルタ)
        self._btn_last_change = {}  # {btn_no: (time, pressed)}
        self._btn_debounce_ms = 80  # 80ms以内のトグルは無視

        # コールバック
        self.on_jog = None       # (delta: int)
        self.on_shuttle = None   # (position: int)
        self.on_button = None    # (button_no: int, pressed: bool)

        self.connected = False

    def _apply_format(self, name):
        """フォーマットを適用"""
        fmt = FORMATS[name]
        self._idx_shuttle = fmt["shuttle"]
        self._idx_jog = fmt["jog"]
        self._idx_btn_start = fmt["btn_start"]
        self._idx_btn_len = fmt["btn_len"]
        self._format_name = name
        log.info(f"Format {name}: shuttle=[{self._idx_shuttle}] "
                 f"jog=[{self._idx_jog}] "
                 f"btn=[{self._idx_btn_start}:{self._idx_btn_start + self._idx_btn_len}]")

    def start(self):
        """読み取りスレッド開始"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """読み取りスレッド停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _open(self):
        try:
            import hid
        except ImportError:
            log.warning("hidapi 未インストール: pip install hidapi")
            return False
        try:
            self._dev = hid.device()
            self._dev.open(VENDOR_ID, PRODUCT_ID_PRO_V2)
            self._dev.set_nonblocking(0)
            self.connected = True
            # 再接続時はフォーマット検出をリセット
            self._initialized = False
            self._report_count = 0
            self._shuttle_out_of_range_count = 0
            log.info("ShuttlePRO v2 接続")
            return True
        except Exception as e:
            log.debug(f"ShuttlePRO v2 未検出: {e}")
            self._dev = None
            self.connected = False
            return False

    def _close(self):
        if self._dev:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None
            self.connected = False

    def _read_loop(self):
        while self._running:
            if not self._dev:
                if not self._open():
                    time.sleep(5.0)
                    continue
            try:
                data = self._dev.read(64, timeout_ms=200)
                if not data:
                    continue
                self._process(list(data))
            except Exception as e:
                log.warning(f"ShuttlePRO 読み取りエラー: {e}")
                self._close()
                time.sleep(1.0)
        self._close()

    def _process(self, data):
        """レポート処理"""
        self._report_count += 1

        # 最初の30レポートを詳細ログ
        if self._report_count <= 30:
            hex_str = " ".join(f"{b:02X}" for b in data)
            changes = ""
            if self._prev_data and len(self._prev_data) == len(data):
                ch = []
                for i in range(len(data)):
                    if data[i] != self._prev_data[i]:
                        ch.append(f"[{i}]:{self._prev_data[i]:02X}->{data[i]:02X}")
                if ch:
                    changes = "  " + " ".join(ch)
            log.info(f"HID#{self._report_count}: {hex_str}{changes}")

        # 同一データなら処理スキップ
        if self._prev_data and data == self._prev_data:
            return

        if not self._initialized:
            self._detect_and_init(data)
        else:
            self._parse(data)

        self._prev_data = list(data)

    def _detect_and_init(self, data):
        """初回レポートからフォーマットを自動検出し初期化"""
        n = len(data)
        log.info(f"レポート長: {n} bytes, data: {[f'0x{b:02X}' for b in data]}")

        if not self._format_locked and n >= 4:
            # ヒューリスティック: 初期状態ではシャトル=0, ボタン=0
            # ジョグカウンタだけが非ゼロの可能性が高い
            non_zero = [(i, data[i]) for i in range(n) if data[i] != 0]

            if len(non_zero) == 1:
                jog_idx = non_zero[0][0]
                jog_val = non_zero[0][1]
                log.info(f"非ゼロバイト: byte[{jog_idx}]=0x{jog_val:02X} → ジョグと推定")

                if jog_idx == 1 and n >= 4:
                    self._apply_format("A")
                elif jog_idx == 4 and n >= 5:
                    self._apply_format("B")
                elif jog_idx == 2 and n >= 5:
                    self._apply_format("C")
                else:
                    log.info(f"ジョグ位置[{jog_idx}]が既知パターンに一致せず、Format Aを使用")
                    self._apply_format("A")
            elif len(non_zero) == 0:
                # 全バイトゼロ: ジョグが偶然0の位置にある
                log.info("全バイトゼロ → Format A をデフォルト使用")
                self._apply_format("A")
            else:
                # 複数の非ゼロバイト: ボタンが押されている等
                log.info(f"複数非ゼロバイト({len(non_zero)}個) → Format A をデフォルト使用")
                self._apply_format("A")

        # 初期値設定
        if self._idx_shuttle < n:
            self._prev_shuttle = self._to_signed(data[self._idx_shuttle])
        if self._idx_jog < n:
            self._prev_jog = data[self._idx_jog]
        self._prev_buttons = self._read_buttons(data)

        self._initialized = True
        log.info(f"初期値: shuttle={self._prev_shuttle}, jog={self._prev_jog}, "
                 f"buttons=0x{self._prev_buttons:04X}")

    def _to_signed(self, val):
        """unsigned byte → signed (-128~127)"""
        return val if val < 128 else val - 256

    def _read_buttons(self, data):
        """ボタンバイトをビットマスクとして読み取り"""
        result = 0
        for i in range(self._idx_btn_len):
            idx = self._idx_btn_start + i
            if idx < len(data):
                result |= data[idx] << (i * 8)
        return result

    def _parse(self, data):
        """レポートパース＋イベント発火"""
        n = len(data)
        if self._idx_shuttle >= n or self._idx_jog >= n:
            return

        shuttle = self._to_signed(data[self._idx_shuttle])
        jog_raw = data[self._idx_jog]
        buttons = self._read_buttons(data)

        # --- フォーマット検証: シャトル値は -7~+7 ---
        if not self._format_locked and self._report_count <= 100:
            if shuttle < -7 or shuttle > 7:
                self._shuttle_out_of_range_count += 1
                if self._shuttle_out_of_range_count >= 3:
                    log.warning(f"シャトル値が範囲外({shuttle})、フォーマット変更を試行")
                    self._try_alternate_format(data)
                # フォーマット不正の疑いがあるのでイベント発火しない
                return

        # --- ジョグ差分計算 (ラッピングカウンタ) ---
        jog_delta = 0
        if self._prev_jog is not None:
            diff = jog_raw - self._prev_jog
            if diff > 128:
                diff -= 256
            elif diff < -128:
                diff += 256
            jog_delta = diff
        self._prev_jog = jog_raw

        # --- ボタン変化検出 ---
        btn_changed = buttons ^ self._prev_buttons

        # --- イベント発火 ---
        if jog_delta != 0 and self.on_jog:
            try:
                self.on_jog(jog_delta)
            except Exception as e:
                log.error(f"on_jog error: {e}")

        if shuttle != self._prev_shuttle and self.on_shuttle:
            try:
                self.on_shuttle(shuttle)
            except Exception as e:
                log.error(f"on_shuttle error: {e}")
        self._prev_shuttle = shuttle

        if btn_changed and self.on_button:
            now = time.time()
            for i in range(15):
                mask = 1 << i
                if btn_changed & mask:
                    pressed = bool(buttons & mask)
                    btn_no = i + 1
                    # デバウンス: 短時間の高速トグルはゴーストとして無視
                    last = self._btn_last_change.get(btn_no)
                    if last:
                        dt = (now - last[0]) * 1000
                        if dt < self._btn_debounce_ms and last[1] != pressed:
                            # ゴースト検出 — イベントを抑制
                            self._btn_last_change[btn_no] = (now, pressed)
                            continue
                    self._btn_last_change[btn_no] = (now, pressed)
                    try:
                        self.on_button(btn_no, pressed)
                    except Exception as e:
                        log.error(f"on_button error: {e}")
        self._prev_buttons = buttons

    def _try_alternate_format(self, data):
        """現在のフォーマットが不正な場合、別フォーマットを試行"""
        current = self._format_name
        alternatives = [name for name in FORMATS if name != current]

        for alt in alternatives:
            fmt = FORMATS[alt]
            shuttle_idx = fmt["shuttle"]
            if shuttle_idx < len(data):
                val = self._to_signed(data[shuttle_idx])
                if -7 <= val <= 7:
                    log.info(f"フォーマット{alt}に変更 (shuttle byte[{shuttle_idx}]={val})")
                    self._apply_format(alt)
                    self._shuttle_out_of_range_count = 0
                    # 再初期化
                    self._prev_shuttle = val
                    if fmt["jog"] < len(data):
                        self._prev_jog = data[fmt["jog"]]
                    self._prev_buttons = self._read_buttons(data)
                    return

        log.warning("全フォーマット検証失敗、Format A を維持")
        self._shuttle_out_of_range_count = 0
