"""retrain_report — 再学習サイクルの結果を 1 枚の Markdown にまとめる (tools/retrain.ps1 の最後に呼ばれる)。

    python tools/retrain_report.py --inventory build/retrain/<stamp>/inventory.json --train models/<new>.train.json
                                   --result tests/real/results/<stamp>_retrain_..._holdout_*.md --out build/retrain/<stamp>/report.md
                                   [--inventory-txt <txt>]

内容: データの棚卸し (要確認を含む)、学習 (分割・学習曲線・検証精度)、holdout での (a) 従来 / (b) 現行モデル / (c) 新モデル の
クリップごとの観測率・ゲート・一致、推奨の目安、次回に向けた不足データ、採用するときの差し替え手順。
推奨は機械的な目安 (ゲート通過・平均観測率・検証 px) なので、確認動画を見て最終判断する。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import re
import sys
from pathlib import Path

ROW = re.compile(r"^(rec_\S+?)((?:\s\+det\S*)?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(OK|NG)")


def parse_result(path: Path) -> dict:
    """結果 md の追跡率表を {mode: [{clip, rate, gate, match, observed, frames}]} に"""
    modes: dict = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = ROW.match(ln)
        if not m:
            continue
        mode = m.group(2).strip() or "従来"
        modes.setdefault(mode, []).append({"clip": m.group(1).replace("_trim.mp4", "").replace(".mp4", ""),
                                           "frames": int(m.group(3)), "observed": int(m.group(4)),
                                           "rate": int(m.group(6)), "match": m.group(12), "gate": m.group(13)})
    return modes


def arg(args, key, default=None):
    return args[args.index(key) + 1] if key in args else default


def main() -> int:
    args = sys.argv[1:]
    inv = json.loads(Path(arg(args, "--inventory")).read_text(encoding="utf-8")) if arg(args, "--inventory") else {}
    inv_txt = ""
    if arg(args, "--inventory-txt"):
        raw = Path(arg(args, "--inventory-txt")).read_bytes()
        for enc in ("utf-8-sig", "utf-16", "cp932"):
            try:
                inv_txt = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    train_paths = [Path(args[i + 1]) for i, a_ in enumerate(args) if a_ == "--train" and i + 1 < len(args)]
    trains = [json.loads(p.read_text(encoding="utf-8")) for p in train_paths if p.exists()]
    trains.sort(key=lambda t: (t.get("train_fraction", 1.0), t.get("n_strong_train", 0)))
    train = trains[-1] if trains else {}
    result = Path(arg(args, "--result")) if arg(args, "--result") else None
    out = Path(arg(args, "--out", "build/retrain/report.md"))
    modes = parse_result(result) if result and result.exists() else {}
    new_name = Path(train.get("onnx", "")).stem if train else ""

    L = []
    L.append(f"# 再学習サイクル レポート ({new_name or 'no model'})\n")
    # ---- 1. データ
    L.append("## 1. データの棚卸し\n")
    if inv:
        L.append(f"- クリップ {len(inv['clips'])} 本、点 {sum(inv['totals'].values())} 点 "
                 f"(" + ", ".join(f"{k} {v}" for k, v in inv["totals"].items()) + ")")
        L.append(f"- 学習に使える強ラベル (holdout 除く): {inv['n_strong_usable']} 点")
        L.append("- 撮影方向: " + ", ".join(f"{k} {v} 本" for k, v in inv["cameras"].items()))
        L.append("- 画像サイズ: " + ", ".join(f"{k} {v} 本" for k, v in inv["sizes"].items()))
        if inv.get("problems"):
            L.append("- 要確認:")
            L += [f"  - {p}" for p in inv["problems"]]
        else:
            L.append("- 要確認: なし")
    if inv_txt:
        L.append("\n```\n" + inv_txt.strip() + "\n```")
    # ---- 2. 学習
    L.append("\n## 2. 学習\n")
    if train:
        L.append(f"- モデル: `{train.get('onnx')}` ({train.get('onnx_mb')} MB, {train.get('arch')}, {train.get('params')} params)")
        L.append(f"- 学習 {train.get('n_train')} 点 (負例 {train.get('n_negative', 0)}) / 検証 {train.get('n_val')} 点、"
                 f"検証クリップ: {', '.join(train.get('val_clips', []))}")
        cams = train.get("cameras", {})
        if cams:
            L.append("- 撮影方向ごとの学習/検証クリップ数: " + ", ".join(f"{c} {len(v['train'])}/{len(v['val'])}" for c, v in cams.items()))
        L.append(f"- {train.get('epochs')} epochs, {train.get('seconds')} 秒 ({train.get('device')})")
        v = train.get("val") or {}
        L.append(f"- 検証精度 (元解像度 px): 平均 {v.get('mean_px')} / 中央値 {v.get('median_px')} / PCK@15 {v.get('pck15')} / PCK@30 {v.get('pck30')} "
                 f"(検出 {v.get('detected')}/{v.get('n')})")
        t = train.get("train_fit") or {}
        L.append(f"- 学習側の当てはまり: 平均 {t.get('mean_px')} px / PCK@15 {t.get('pck15')}")
        hist = train.get("history") or []
        if hist:
            L.append("\n| epoch | loss | 検証 平均px | 中央値px | PCK@15 | PCK@30 |\n|---|---|---|---|---|---|")
            for h in hist:
                L.append(f"| {h['epoch']} | {h['loss']} | {h.get('mean_px')} | {h.get('median_px')} | {h.get('pck15')} | {h.get('pck30')} |")
    if inv.get("delta"):
        d = inv["delta"]
        L.append("\n### 前回からの増分\n")
        L.append(f"- クリップ +{len(d['clips_added'])} 本、点 +{d['points']}、強ラベル {d.get('n_strong_prev')} → {inv['n_strong_usable']}")
        L.append("- 由来: " + ", ".join(f"{k} +{v}" for k, v in d["sources"].items()))
        L.append("- 撮影方向: " + ", ".join(f"{k} {d['cameras_prev'].get(k, 0)} → {inv['cameras'].get(k, 0)}"
                                         for k in sorted(set(inv["cameras"]) | set(d["cameras_prev"]))))
    if len(trains) > 1:
        L.append("\n## 2b. 学習曲線 (学習クリップの割合を変えて学習、検証クリップは同じ)\n")
        L.append("| 割合 | 学習クリップ | 学習点 | 検証 中央値 px | 検証 PCK@15 | 検証 PCK@30 | 検出率 | モデル |\n|---|---|---|---|---|---|---|---|")
        for t in trains:
            v = t.get("val") or {}
            L.append(f"| {t.get('train_fraction', 1.0):.0%} | {len(t.get('train_clips', []))} | {t.get('n_strong_train')} | "
                     f"{v.get('median_px')} | {v.get('pck15')} | {v.get('pck30')} | {v.get('detected')}/{v.get('n')} | `{Path(t.get('onnx', '')).stem}` |")
    # ---- 3. holdout
    L.append("\n## 3. holdout での比較 (tools/eval_club.py)\n")
    if modes:
        L.append(f"結果ファイル: `{result.name}`\n")
        L.append("| モード | 平均観測率 | ゲート通過 | 一致 (トップ/インパクト) | 不通過 |\n|---|---|---|---|---|")
        summary = {}
        for mode, rows in modes.items():
            rate = sum(r["rate"] for r in rows) / len(rows)
            ok = sum(1 for r in rows if r["gate"] == "OK")
            match = sum(r["match"].count("o") for r in rows)
            ng = [r["clip"][-6:] for r in rows if r["gate"] == "NG"]
            summary[mode] = (rate, ok, match)
            L.append(f"| {mode} | {rate:.0f}% | {ok} / {len(rows)} | {match} | {', '.join(ng) or '-'} |")
        clips = [r["clip"] for r in next(iter(modes.values()))]
        L.append("\n| 素材 | " + " | ".join(f"{m} 率 / ゲート" for m in modes) + " |\n|---|" + "---|" * len(modes))
        for i, c in enumerate(clips):
            cells = []
            for m, rows in modes.items():
                r = rows[i] if i < len(rows) else None
                cells.append(f"{r['rate']}% / {r['gate']}" if r else "-")
            L.append(f"| {c[-11:]} | " + " | ".join(cells) + " |")
        L.append("\n正解ラベル (px 誤差) は結果ファイルの「誤差」節を参照 (ラベルが無い素材は観測率・ゲートのみ)。")
        # ---- 4. 推奨の目安
        L.append("\n## 4. 判断材料 (機械的な目安)\n")
        new_mode = next((m for m in modes if new_name and new_name in m), None)
        cur_mode = "+det" if "+det" in modes else None
        if new_mode and cur_mode:
            nr, nk, nm = summary[new_mode]
            cr, ck, cm = summary[cur_mode]
            br, bk, bm = summary.get("従来", (0, 0, 0))
            verdict = "採用候補" if (nk >= ck and nr >= cr - 2 and nm >= cm) else "見送り (現行のまま)"
            L.append(f"- 新モデル vs 現行: ゲート通過 {ck}→{nk}、平均観測率 {cr:.0f}%→{nr:.0f}%、一致 {cm}→{nm} → **{verdict}**")
            L.append(f"- 従来方式 (検出器なし): ゲート通過 {bk}、平均観測率 {br:.0f}%、一致 {bm}")
            L.append("- 上は目安。確認動画 (`build/club_eval/real/<素材>_det_<モデル名>/club_tracker_overlay.mp4`) で改善・悪化した素材を見て決める。")
    else:
        L.append("(holdout の結果ファイルがありません)")
    # ---- 5. 差し替え手順
    L.append("\n## 5. 採用するときの差し替え手順 (まだ差し替えない)\n")
    L.append("```powershell\n"
             f"Copy-Item models\\clubhead.onnx models\\clubhead_prev.onnx          # 現行を退避\n"
             f"Copy-Item models\\{new_name}.onnx models\\clubhead.onnx            # 新モデルを現行に\n"
             f"Copy-Item models\\{new_name}.train.json models\\clubhead_train.json\n"
             "python tools\\eval_club.py --all --split tune --tag <tag> --detector both   # tune で退行確認\n"
             "powershell -ExecutionPolicy Bypass -File tools\\release.ps1              # exe に同梱して配布\n"
             "```\n"
             "戻すときは `clubhead_prev.onnx` を `clubhead.onnx` に戻す。現場 PC は `_internal\\models\\clubhead.onnx` を差し替えるだけでもよい。")
    # ---- 6. 不足データ
    L.append("\n## 6. 次回に向けて不足しているデータ\n")
    if inv:
        cams = inv["cameras"]
        L.append("- 撮影方向の偏り: " + ", ".join(f"{k} {v} 本" for k, v in cams.items()) +
                 "。少ない方向 (特に後方 DTL と斜め) の実収録を増やす")
        L.append("- 同じティー・同じ背景のクリップが大半なら、別ホール・別の時間帯 (逆光・夕方)・別の服装 (白ウェア) を足す")
        L.append("- 負例 (人が削除した自動点) が少ないと誤検出を学べない。自動生成 → 間違いを Delete → 書き出し の手順で貯める")
        L.append("- HFR 2x (29.97fps 記録) とインターレース素材、ドライバー以外 (アイアン・ウェッジ) の割合")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"レポート: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
