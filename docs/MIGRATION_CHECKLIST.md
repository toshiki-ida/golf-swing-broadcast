# 現場 PC への移行チェックリスト (UltraStudio 4K Mini / Thunderbolt)

対象: `feature/ai-analysis` の onedir ビルド (`handoff/GolfSwingBroadcast/` または zip)。
持ち込み物の一覧は `handoff/README.md`。所要時間の目安: 1 時間 (Desktop Video のインストールと再起動を含む)。

各ステップは **確認方法** と **うまくいかない時の確認点** を書いた。上から順に進め、失敗したステップで止まって切り分ける。

---

## 0. 持ち込み前 (開発 PC 側)

- [ ] `handoff/` の中身が揃っている (`handoff/README.md` の一覧)。exe の zip、サンプル素材、wheel、ドキュメント。
- [ ] 開発 PC で `GolfSwingBroadcast.exe --ai-selftest handoff\samples\rec_0611_131029_trim.mp4 <出力dir>` が `OK` になることを確認済み。
- [ ] `git push` 済み (現場で `git clone` できる)。

## 1. Desktop Video (DeckLink ドライバ) のインストール

- Blackmagic の Support ページから **Desktop Video 12.x 以降** (Windows) をダウンロードしてインストール → 再起動。
  - 本アプリは SDK 12.0 で追加された `IDeckLinkVideoBuffer` を使うため、**12.0 未満では動かない**。
  - 開発履歴に「動作確認済みの Desktop Video バージョン」の記録は無い。現場で使った版を本書末尾の記録欄に書くこと。
- **確認**: `C:\Program Files\Blackmagic Design\Blackmagic Desktop Video\DeckLinkAPI64.dll` が存在する。
  アプリ起動時のログに `[DeckLink] DeckLinkAPI64.dll version: 12.x.x.x` と `[DeckLink] API version: 12.x` が出る (手順 6 で確認)。
- **うまくいかない時**: インストーラが「再起動が必要」のまま止まる → 再起動後にもう一度インストーラを実行。
  旧版が残っている場合は「プログラムの追加と削除」で Desktop Video を削除してから入れ直す。

## 2. Thunderbolt 接続の許可

- UltraStudio 4K Mini を Thunderbolt 3 ケーブルで接続し、電源を入れる。
- Windows の通知「新しい Thunderbolt デバイス」→ **常に接続を許可**。出ない場合は Thunderbolt コントロールセンター (Intel) を開き、デバイスを「常に接続」にする。
- **確認**: デバイスマネージャーに「Blackmagic Design」配下で UltraStudio が表示され、警告マークが無い。
- **うまくいかない時**:
  - Thunderbolt コントロールセンターに何も出ない → BIOS で Thunderbolt のセキュリティレベルが「No Security」か「User Authorization」になっているか確認。ケーブルが USB-C (非 Thunderbolt) でないか確認 (Thunderbolt の雷マーク)。
  - デバイスマネージャーに「!」→ Desktop Video を入れ直す (手順 1)。

## 3. Desktop Video Setup で入力設定

- 「Blackmagic Desktop Video Setup」を起動 → UltraStudio 4K Mini を選択。
- **Video Input: SDI**、**Video Output: SDI** (アプリも `bmdDeckLinkConfigVideoInputConnection = SDI` を起動時に設定するが、ここで合わせておく)。
- SDI 入力の規格: 1080i59.94 (Sony CCU HFR 2x の場合も信号は 1080i59.94)。
- **確認**: Setup 画面の入力モニターに映像が出る。
- **うまくいかない時**: 「No Input」→ SDI ケーブルが **IN** に刺さっているか、カメラ / CCU 側が 1080i59.94 を出しているか (1080p59.94 や 3G-SDI Level B だと出ないことがある)。

## 4. Media Express で映像確認

- Media Express を起動 → Log and Capture → プレビューに映像と音声レベルが出る。
- **確認**: 映像が止まらず流れ、プロジェクト設定が 1080i59.94 になっている。
- **うまくいかない時**: ここで映らないならアプリでも映らない。手順 2〜3 に戻る。Media Express は確認後 **必ず終了** する (デバイスを掴んだままだとアプリが開けない)。

## 5. ウイルス対策ソフトの例外設定

- 未署名の PyInstaller 製 exe のため、ウイルスバスター / Defender が隔離・遅延させることがある。
- 展開先フォルダ (例 `D:\GolfSwingBroadcast\`) をスキャン除外に追加。zip 展開直後に exe が消えていたら隔離を疑う。
- **確認**: `GolfSwingBroadcast.exe` と `_internal\ffmpeg.exe` が存在する。
- **うまくいかない時**: 隔離ログから復元 → 除外設定 → zip を再展開。

## 6. exe 起動 (DeckLink 検出ログの確認)

- `GolfSwingBroadcast.exe` は `--noconsole` なのでログが見えない。**初回はコマンドプロンプトから** 起動する:
  ```
  cd /d D:\GolfSwingBroadcast
  GolfSwingBroadcast.exe --project D:\golf_project > startup.log 2>&1
  ```
  (ウィンドウを閉じたあと `startup.log` を見る)
- **確認** (ログに以下が出る):
  - `[DeckLink] SDK type library loaded OK`
  - `[DeckLink] DeckLinkAPI64.dll version: ...` / `[DeckLink] API version: ...`
  - `[DeckLink] Device 0: UltraStudio 4K Mini (model: ..., input=True, output=True)`
  - `[DeckLink Input] EnableVideoInput: 1080i59.94 ...`
  - `[DeckLink] Format changed: HD 1080i 59.94 1920x1080 29.970fps interlaced fieldDominance=5 ...`
  - `[DeckLink] Frame #1: 1920x1080 rowBytes=3840`
  - `[DeckLink] フレーム受信確認OK`
- **うまくいかない時**:
  - `SDK not found` → 手順 1。
  - `入力デバイスが見つかりません` → 手順 2 (Thunderbolt 許可) / Media Express が開いたままでないか。
  - `Device 0` は出るが `Frame #1` が出ず `3秒間フレーム未到着。フォールバック起動` → 入力信号なし (手順 3〜4)。アプリはテストパターンで動き続けるので、カメラを繋いでから「入力停止 → 入力開始」。
  - `Format changed: ... progressive` → カメラが 1080p を出している。CCU 側を 1080i59.94 に。
  - 起動自体が落ちる → `startup.log` の末尾の Traceback を控える。`_internal\customtkinter` が無ければ zip 展開が不完全。

## 7. SDI 入力の確認 (収録タブ)

- 収録タブのプレビューに映像が出て、「実効: 59.94 fps」表示。入力モードは **通常 (59.94fps)**。
  Sony CCU の HFR 2x を受ける場合のみ **倍速 HFR 2x** を選ぶ (29.97fps 記録 → 1/2 速スロー)。
- **うまくいかない時**: NO SIGNAL のまま → 手順 6 のログで `Frame #1` が出ているか。出ているのに真っ黒 → `rowBytes` が 3840 でない (フォーマット不一致)。

## 8. 収録

- REC → 5 秒程度 → STOP。`recordings/MM-DD/rec_*.mp4` ができ、クリップタブに現れる。
- **確認**: ログの `[REC DIAG] enqueued=N, written=N, dropped=0`。`dropped` が 0 でなければディスクが遅い (USB HDD は不可、内蔵 SSD に)。
- **うまくいかない時**: `ffmpeg未検出、cv2.VideoWriter にフォールバック` → `_internal\ffmpeg.exe` が無い (手順 5 の隔離)。

## 9. 切り出し

- クリップタブで In/Out を設定 → 「クリップ切り出し」→ `clips/MM-DD/*_trim_*.mp4`。
- **確認**: HFR 2x 収録なら切り出した mp4 の fps が 29.97、通常なら 59.94 (`ffprobe` で確認)。
- **うまくいかない時**: 切り出しボタンが出ない → クリップを選択していない。失敗 → ログの `ffmpeg_extract 失敗`。

## 10. 軌道自動生成 (編集タブ → AI (試作))

- 切り出したクリップを編集タブで開き、「ヘッド軌道を自動生成」。10〜60 秒 (CPU)。
- **確認**: ステータスに `完了 (N秒): 制御点 18 点 → Swing 1 (観測 M/T f)`、軌道がキャンバスに描かれる。
  `要手直し F..` が出た区間は点を手で直す (右ドラッグ移動 / 中クリック削除)。
- **うまくいかない時**:
  - `mediapipe が未インストール` → exe に同梱されていない (ビルドが古い)。
  - `モデルがありません` → `_internal\models\pose_landmarker_heavy.task` が無い。
  - `骨格を検出できませんでした` → 人物が小さすぎる / 画面外。In/Out でスイング 1 回に切ってから再実行。
  - キャディに骨格が付く → 修正 5 (手首が最も動く人を選ぶ) 済みだが、確認動画で別人なら素材名を控える。
  - 処理が止まって見える → バックグラウンドで動いている。ステータスの % を待つ。

## 11. 縦型書き出し (SNS)

- 「縦型動画 (SNS) 書き出し」→ `exports/sns/<クリップ名>_short.mp4` と同名 `.json`、`exports/sns/preview/` に静止画 3 枚。
- **確認**: `.json` の `camera.angle` (dtl / face_on / unknown) と `warnings`。トップ/インパクトの静止画が妥当か。
- **うまくいかない時**:
  - `ffmpeg がエラー終了` → `exports/sns/ffmpeg_sns.log` を見る。NVENC が無い PC では `hw_encode` が自動で libx264 に落ちる。
  - 局面が明らかに違う → `.json` の `trajectory_phases` / `skeleton_phases` を控える (両手法のどちらが外れたか)。
  - フォントが豆腐 → `_internal\assets\fonts\NotoSansJP.ttf` が無い。

## 12. 現場で控えて持ち帰るもの

- `startup.log` (手順 6)、`settings.json`、うまくいかなかった素材 (`recordings/` の該当 mp4) と `exports/sns/*.json`。
- 使った Desktop Video のバージョン: ______________ (本書を更新して commit する)
- UltraStudio 4K Mini のファームウェア更新を求められた場合、その版: ______________

---

### 補足: 実収録素材の評価を現場で回す場合

```
python tools\label_club.py tests\real\<素材> --frames <アドレス,テイクバック,トップ,ダウン,インパクト,フォロー>
python tools\eval_club.py --all --split all --tag site_<日付>
```
(Python 環境は `handoff/README.md` のオフラインインストール手順で作る)
