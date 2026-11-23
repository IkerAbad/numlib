"""Execute every Python code block in README.md, in order."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def main() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    if not blocks:
        raise SystemExit("no Python code blocks found in README.md")
    namespace: dict[str, Any] = {}
    for index, code in enumerate(blocks, start=1):
        source = f"README.md Python block {index}"
        exec(compile(code, source, "exec"), namespace)
    print(f"executed {len(blocks)} Python blocks from README.md")


if __name__ == "__main__":
    main()
