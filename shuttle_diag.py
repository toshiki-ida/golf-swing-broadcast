"""ShuttlePRO v2 HID診断GUI

ジョグ・シャトル・ボタンを操作すると、変化するバイトをリアルタイムで表示。
正しいバイトマッピングを特定するためのツール。
"""
import hid
import threading
import time
import tkinter as tk

VID = 0x0B33
PID = 0x0030


class ShuttleDiagApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ShuttlePRO v2 HID 診断")
        self.geometry("900x650")
        self.configure(bg="#1a1a1a")

        self._dev = None
        self._running = True
        self._prev_data = None
        self._report_count = 0

        # ヘッダー
        tk.Label(self, text="ShuttlePRO v2 HID 診断ツール",
                 font=("Consolas", 18, "bold"), bg="#1a1a1a", fg="white").pack(pady=10)

        # 接続状態
        self.status_label = tk.Label(self, text="接続中...",
                                      font=("Consolas", 14), bg="#1a1a1a", fg="#FFAA00")
        self.status_label.pack()

        # バイト表示エリア
        byte_frame = tk.Frame(self, bg="#1a1a1a")
        byte_frame.pack(pady=20)

        tk.Label(byte_frame, text="Byte位置:", font=("Consolas", 12),
                 bg="#1a1a1a", fg="gray").grid(row=0, column=0, padx=5)

        self.byte_labels = []
        self.byte_values = []
        for i in range(8):
            tk.Label(byte_frame, text=f"[{i}]", font=("Consolas", 12, "bold"),
                     bg="#1a1a1a", fg="cyan").grid(row=0, column=i + 1, padx=8)
            val = tk.Label(byte_frame, text="--", font=("Consolas", 20, "bold"),
                           bg="#333", fg="white", width=4)
            val.grid(row=1, column=i + 1, padx=5, pady=5)
            self.byte_values.append(val)

        # 変化検出
        tk.Label(byte_frame, text="HEX:", font=("Consolas", 12),
                 bg="#1a1a1a", fg="gray").grid(row=1, column=0, padx=5)

        self.hex_labels = []
        for i in range(8):
            val = tk.Label(byte_frame, text="--", font=("Consolas", 14),
                           bg="#1a1a1a", fg="#888")
            val.grid(row=2, column=i + 1, padx=5, pady=2)
            self.hex_labels.append(val)

        # 解析結果
        analysis_frame = tk.Frame(self, bg="#2a2a2a", bd=1, relief="solid")
        analysis_frame.pack(fill="x", padx=20, pady=10)

        self.analysis_text = tk.Text(analysis_frame, height=8, font=("Consolas", 13),
                                      bg="#2a2a2a", fg="#00FF00", insertbackground="white",
                                      wrap="word", bd=0)
        self.analysis_text.pack(fill="x", padx=10, pady=10)
        self.analysis_text.insert("1.0",
                                   "操作してください:\n"
                                   "  1. シャトルリング (外側の回転リング) を回す\n"
                                   "  2. ジョグダイヤル (内側のホイール) を回す\n"
                                   "  3. 各ボタンを1つずつ押す\n\n"
                                   "変化するバイトがハイライトされます。")
        self.analysis_text.config(state="disabled")

        # ログ
        log_frame = tk.Frame(self, bg="#1a1a1a")
        log_frame.pack(fill="both", expand=True, padx=20, pady=5)

        tk.Label(log_frame, text="生データログ:", font=("Consolas", 11),
                 bg="#1a1a1a", fg="gray").pack(anchor="w")

        self.log_text = tk.Text(log_frame, height=10, font=("Consolas", 11),
                                 bg="#111", fg="#AAA", insertbackground="white", wrap="none")
        self.log_text.pack(fill="both", expand=True)

        # バイト変化トラッカー
        self._byte_change_count = [0] * 8
        self._byte_ranges = [set() for _ in range(8)]

        # HID読み取りスレッド
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._running = False
        if self._dev:
            try:
                self._dev.close()
            except Exception:
                pass
        self.destroy()

    def _read_loop(self):
        try:
            self._dev = hid.device()
            self._dev.open(VID, PID)
            self._dev.set_nonblocking(0)
            self.after(0, lambda: self.status_label.configure(
                text="接続OK: ShuttlePRO v2", fg="#00FF00"))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(
                text=f"接続エラー: {e}", fg="#FF0000"))
            return

        while self._running:
            try:
                data = self._dev.read(64, timeout_ms=200)
                if data:
                    self._report_count += 1
                    self.after(0, lambda d=list(data), n=self._report_count:
                               self._update_display(d, n))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(
                    text=f"読み取りエラー: {e}", fg="#FF0000"))
                break

    def _update_display(self, data, count):
        n = min(len(data), 8)

        # バイト値表示
        for i in range(8):
            if i < len(data):
                val = data[i]
                self.byte_values[i].configure(text=f"{val:3d}")
                self.hex_labels[i].configure(text=f"0x{val:02X}")

                # 変化検出
                changed = False
                if self._prev_data and i < len(self._prev_data):
                    if data[i] != self._prev_data[i]:
                        changed = True
                        self._byte_change_count[i] += 1
                        self._byte_ranges[i].add(val)

                if changed:
                    self.byte_values[i].configure(bg="#FF4400", fg="white")
                else:
                    self.byte_values[i].configure(bg="#333", fg="white")
            else:
                self.byte_values[i].configure(text="--", bg="#222", fg="#555")
                self.hex_labels[i].configure(text="--")

        # ログ追加
        hex_str = " ".join(f"{b:02X}" for b in data)
        changed_bytes = ""
        if self._prev_data and len(self._prev_data) == len(data):
            changed = [str(i) for i in range(len(data)) if data[i] != self._prev_data[i]]
            if changed:
                changed_bytes = f"  <- byte[{','.join(changed)}]"

        self.log_text.insert("end", f"#{count:4d}: {hex_str}{changed_bytes}\n")
        self.log_text.see("end")

        # 解析更新 (10レポートごと)
        if count % 5 == 0:
            self._update_analysis()

        self._prev_data = list(data)

    def _update_analysis(self):
        lines = ["バイト変化分析:\n"]
        for i in range(min(8, len(self._byte_ranges))):
            cnt = self._byte_change_count[i]
            if cnt > 0:
                rng = sorted(self._byte_ranges[i])
                min_v, max_v = min(rng), max(rng)
                signed_min = min_v if min_v < 128 else min_v - 256
                signed_max = max_v if max_v < 128 else max_v - 256
                lines.append(
                    f"  byte[{i}]: {cnt}回変化, "
                    f"範囲={min_v}-{max_v} (signed: {signed_min}~{signed_max}), "
                    f"値数={len(rng)}"
                )
                # 推定
                if signed_min >= -7 and signed_max <= 7 and len(rng) <= 15:
                    lines[-1] += "  → シャトルリング?"
                elif len(rng) > 15 and cnt > 5:
                    lines[-1] += "  → ジョグダイヤル?"
                elif max_v > 0 and len(rng) <= 5:
                    lines[-1] += "  → ボタン?"

        self.analysis_text.config(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "\n".join(lines))
        self.analysis_text.config(state="disabled")


if __name__ == "__main__":
    app = ShuttleDiagApp()
    app.mainloop()
