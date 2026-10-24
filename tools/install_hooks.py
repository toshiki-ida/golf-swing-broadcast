"""install_hooks — ローカルの git フックを設置する (main への push をブロック)。

    python tools/install_hooks.py

.git/hooks/pre-push を書き、main / master へ push しようとしたら拒否する。
(リポジトリ外の設定なので clone ごとに実行が必要)
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

HOOK = """#!/bin/sh
# main / master への push をブロックする (feature/* で作業し Pull Request で提出する)
while read local_ref local_sha remote_ref remote_sha; do
  case "$remote_ref" in
    refs/heads/main|refs/heads/master)
      echo "pre-push: $remote_ref への直接 push は禁止です。feature/* ブランチから Pull Request を出してください。" >&2
      exit 1
      ;;
  esac
done
exit 0
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hooks = root / ".git" / "hooks"
    if not hooks.is_dir():
        print(f".git/hooks がありません: {hooks}")
        return 1
    path = hooks / "pre-push"
    path.write_text(HOOK, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"設置しました: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
