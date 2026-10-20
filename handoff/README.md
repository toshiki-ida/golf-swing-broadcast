# handoff/ — 現場 PC 持ち込みセット

作成: feature/ai-analysis (コミットは `VERSION.txt`)。手順は `docs/MIGRATION_CHECKLIST.md`。

## 中身

| パス | 内容 |
|---|---|
| `GolfSwingBroadcast-*.zip` | onedir ビルド一式 (exe + `_internal` + `config/`)。展開して `GolfSwingBroadcast\GolfSwingBroadcast.exe` |
| `GolfSwingBroadcast/` | 上と同じものを展開した状態 (USB から直接起動確認する用) |
| `samples/` | 動作確認用素材 3 本 (切り出し済み、1920x1080)。`rec_0611_131029_trim` = HFR 2x (29.97fps 記録)、`rec_0612_123119_trim` = 59.94p、`rec_0611_164617_trim` = キャディ在席 (複数人) |
| `samples/results/` | 開発 PC での処理結果の見本 (軌道の確認動画、縦型動画、静止画、解析 JSON、selftest 結果) |
| `wheels/` | Python 3.11 / win_amd64 用 wheel 一式 (requirements.txt + pyinstaller)。オフラインで `pip install` できる |
| `requirements.txt` | 上に対応する requirements |

## ソースから動かす場合 (任意)

1. Python 3.11 (64bit) をインストール: https://www.python.org/downloads/windows/ → "Python 3.11.x Windows installer (64-bit)"。
   「Add python.exe to PATH」にチェック。
2. FFmpeg essentials: https://www.gyan.dev/ffmpeg/builds/ → `ffmpeg-release-essentials.zip` を展開し、`bin\ffmpeg.exe` を PATH に入れるか、
   リポジトリ直下に `ffmpeg.exe` / `ffprobe.exe` を置く (`ffmpeg_writer.find_ffmpeg` はスクリプト同階層も探す)。
   (exe 版は `_internal\ffmpeg.exe` を同梱しているので不要)
3. リポジトリ: `git clone -b feature/ai-analysis https://github.com/toshiki-ida/golf-swing-broadcast.git`
4. オフラインインストール:
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install --no-index --find-links <USB>\handoff\wheels -r requirements.txt
   ```
5. モデル・フォントは exe の `_internal\models` / `_internal\assets` からコピーするか、ネットがあれば `python setup_assets.py`。

## Blackmagic Desktop Video

- ダウンロード: https://www.blackmagicdesign.com/support/ → Capture and Playback → Desktop Video (Windows)。
- **必要バージョン: 12.0 以降** (本アプリは SDK 12.0 で追加された `IDeckLinkVideoBuffer` を使う)。
- UltraStudio 4K Mini は Desktop Video 10.9.5 以降で対応。Thunderbolt 3 接続。
- 開発履歴に「動作確認済みの Desktop Video / SDK バージョン」の記録は **無い**。開発 PC (Thunderbolt 非対応) には SDK が入っていない。
  現場で使ったバージョンはアプリ起動ログの `[DeckLink] DeckLinkAPI64.dll version:` / `[DeckLink] API version:` に出るので、
  `docs/MIGRATION_CHECKLIST.md` 末尾に記録する。

## 動作確認コマンド

```
GolfSwingBroadcast\GolfSwingBroadcast.exe --ai-selftest samples\rec_0611_131029_trim.mp4 D:\selftest
type D:\selftest\selftest_result.json      ← "status": "OK" なら軌道生成 → 縦型書き出しが通っている
```
