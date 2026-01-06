"""
生成 instruments/f88.txt（主力连续 *88）

Qlib 的 instruments 文件格式通常为：
INSTRUMENT<TAB>START_DATE<TAB>END_DATE

本脚本从 instruments/all.txt 过滤出 “主力连续” 合约集合，并写入 instruments/f88.txt。

过滤规则（针对本仓库数据的命名习惯）：
- 合约代码以 '88' 结尾
- 且倒数第 3 位不是数字
  - 例如：A88 / AG88 / IF88 / L_F88 ✅
  - 例如：A888 / AG889 ❌（避免误把 888/889 当成 *88）
"""

from __future__ import annotations

import argparse
from pathlib import Path


def is_f88_instrument(inst: str) -> bool:
    """判断是否为主力连续 *88 合约（避免 888/889 等误匹配）。"""
    inst = inst.strip()
    if len(inst) < 3:
        return False
    if not inst.endswith("88"):
        return False
    # inst[-3] 是 '88' 前一位；若是数字则说明可能是 888/188 等，不作为主力连续
    return not inst[-3].isdigit()


def build_f88_file(qlib_dir: Path, overwrite: bool = True) -> tuple[Path, int]:
    instruments_dir = qlib_dir / "instruments"
    src = instruments_dir / "all.txt"
    dst = instruments_dir / "f88.txt"

    if not src.exists():
        raise FileNotFoundError(f"source instruments file not found: {src}")

    if dst.exists() and not overwrite:
        # 不覆盖时，直接返回已有文件
        line_cnt = sum(1 for _ in dst.open("r", encoding="utf-8"))
        return dst, line_cnt

    out_lines: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # 兼容 TAB/SPACE 分隔
            parts = line.split()
            inst = parts[0]
            if is_f88_instrument(inst):
                out_lines.append(line)

    instruments_dir.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        if out_lines:
            f.write("\n".join(out_lines) + "\n")
        else:
            # 保持文件存在但为空（便于排查）
            f.write("")

    return dst, len(out_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qlib_dir",
        type=str,
        required=True,
        help="Qlib bin 根目录（包含 instruments/、features/、calendars/）",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=1,
        help="1=覆盖写入 f88.txt；0=若已存在则跳过",
    )
    args = parser.parse_args()

    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    dst, n = build_f88_file(qlib_dir, overwrite=bool(args.overwrite))
    print(f"[build_instruments_f88] wrote {dst} (n={n})")


if __name__ == "__main__":
    main()


