#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Convert tiletranslation benchmark result text files into a Google Sheets CSV.

Reads the ``*_results.txt`` files produced by run_all_tiletranslation.sh (only
the benchmarks the tiletranslation backend implements), parses the triton
perf_report tables, and writes ``tiletranslation_results.csv`` with one row
per (benchmark, config, variant) data point.

Tables are the pandas ``DataFrame.to_string()`` output of
``triton.testing.perf_report``: a ``<name>-<unit>:`` title line, a header line,
then data rows whose first token is the DataFrame index. Columns are separated
by runs of 2+ spaces, so multi-word names like ``CuTile static persistent
(GB/s)`` stay intact when split on ``\\s{2,}``. Data rows carry one extra
leading token (the index), so a data line must split into ``len(header) + 1``
all-numeric tokens.

Column layout (fixed, so the overhead formulas always reference the right
letters):

    A Benchmark   B Unit   C Param Name   D Param Value   E Variant
    F CuTile      G DropTokens   H TileTranslation   I PyTorch
    J DropTokens vs CuTile       =IF(AND(ISNUMBER(F2),ISNUMBER(G2)),G2/F2-1,"")
    K TileTranslation vs CuTile  =IF(AND(ISNUMBER(F2),ISNUMBER(H2)),H2/F2-1,"")
    L PyTorch vs CuTile          =IF(AND(ISNUMBER(F2),ISNUMBER(I2)),I2/F2-1,"")
    M Source File   N Run Status

Overhead cells are relative to the CuTile column (ratio - 1): 0 = same speed,
0.25 = 25% slower, -0.1 = 10% faster. Format J:N as percent in Google Sheets
(Format > Number > Percent) if you prefer percentages.

Usage:
    python3 bench_results_to_csv.py [OUTPUT_DIR]
"""

import csv
import re
import sys
from pathlib import Path

CSV_NAME = "tiletranslation_results.csv"

# Fixed column letters the overhead formulas depend on.
COL_BASELINE = "F"  # CuTile
OVERHEAD_TARGETS = {
    "DropTokens": "G",
    "TileTranslation": "H",
    "PyTorch": "I",
}
# A table column maps to a CSV backend column when its name (unit suffix
# stripped) starts with one of these prefixes. Longer prefixes first so
# e.g. "TileTranslation static persistent" wins over any shorter overlap.
BACKEND_PREFIXES = ["DropTokens", "TileTranslation", "CuTile", "PyTorch"]

HEADERS = [
    "Benchmark",
    "Unit",
    "Param Name",
    "Param Value",
    "Variant",
    "CuTile",
    "DropTokens",
    "TileTranslation",
    "PyTorch",
    "DropTokens vs CuTile",
    "TileTranslation vs CuTile",
    "PyTorch vs CuTile",
    "Source File",
    "Run Status",
]

OVERHEAD_TEMPLATE = '=IF(AND(ISNUMBER({base}{row}),ISNUMBER({target}{row})),{target}{row}/{base}{row}-1,"")'

# Benchmarks the tiletranslation backend implements, discovered the same way
# run_all_tiletranslation.sh selects them: any bench_*.py declaring a
# ("tiletranslation", ...) backend entry.
TT_BACKEND_RE = re.compile(r'\("tiletranslation"')


def overhead_formula(target_col: str, row: int) -> str:
    return OVERHEAD_TEMPLATE.format(base=COL_BASELINE, target=target_col, row=row)


def tiletranslation_result_files(benchmark_dir: Path) -> list:
    """Result .txt files for the tiletranslation benchmarks, in bench order."""
    files = []
    for bench in sorted(benchmark_dir.rglob("bench_*.py")):
        # Skip this converter (and any helper) whose source mentions the
        # backend string without being a runnable benchmark.
        if bench.name in ("bench_results_to_csv.py", "bench_utils.py"):
            continue
        if not TT_BACKEND_RE.search(bench.read_text(errors="replace")):
            continue
        rel = bench.relative_to(benchmark_dir)
        safe_name = rel.as_posix().replace("/", "_")
        safe_name = re.sub(r"\.py$", "", safe_name)
        files.append(benchmark_dir / f"{safe_name}_results.txt")
    return files


def _split_columns(line: str) -> list:
    """Split a perf_report table line into its columns.

    pandas separates columns with runs of 2+ spaces; single spaces occur
    inside column names ("CuTile static persistent (GB/s)").
    """
    return re.split(r"\s{2,}", line.strip())


def _all_numeric(tokens: list) -> bool:
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def _match_backend(column_name: str) -> tuple:
    """Map a table column name to (csv_column, variant) or (None, None).

    Strips the trailing unit suffix ("(GB/s)", "(TFLOPS)") and matches the
    remainder by prefix. "CuTile multi wave cached (GB/s)" becomes
    ("CuTile", "multi wave cached"); plain "CuTile (GB/s)" becomes
    ("CuTile", "").
    """
    name = re.sub(r"\s*\((?:GB/s|TFLOPS|us|ms)\)\s*$", "", column_name).strip()
    for prefix in BACKEND_PREFIXES:
        if name == prefix:
            return prefix, ""
        if name.startswith(prefix + " "):
            return prefix, name[len(prefix) + 1 :]
    return None, None


def parse_results_file(txt: Path) -> list:
    """Extract (benchmark, unit, param_name, param_value, variant, values) rows."""
    rows = []
    lines = txt.read_text(errors="replace").splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.endswith("-GBps:") and not stripped.endswith("-TFLOPS:"):
            continue
        unit = stripped[:-1].rsplit("-", 1)[-1]
        benchmark = stripped[: -(len(unit) + 2)]

        # Header is the next non-empty line; the x-axis parameter is its first
        # column, the rest are backend columns.
        header = None
        for j in range(i + 1, min(i + 5, len(lines))):
            if lines[j].strip():
                header = _split_columns(lines[j])
                break
        if header is None or len(header) < 2:
            continue
        param_name = header[0]

        # Data rows: DataFrame index + one numeric value per header column.
        # Stop at the first line that does not parse (blank line, next title,
        # or interleaved log output).
        data_rows = []
        for j in range(i + 2, len(lines)):
            if not lines[j].strip():
                break
            tokens = _split_columns(lines[j])
            if len(tokens) != len(header) + 1 or not _all_numeric(tokens[1:]):
                break
            data_rows.append(tokens)

        for tokens in data_rows:
            values = {}  # (backend, variant) -> value
            for column_name, token in zip(header[1:], tokens[2:]):
                backend, variant = _match_backend(column_name)
                if backend is not None:
                    values[(backend, variant)] = float(token)

            # One CSV row per variant so rmsnorm's kernel variants stay apart.
            variants = sorted({v for (_, v) in values})
            for variant in variants or [""]:
                row_values = {
                    backend: values.get((backend, variant))
                    for backend in BACKEND_PREFIXES
                }
                rows.append(
                    {
                        "benchmark": benchmark,
                        "unit": unit,
                        "param_name": param_name,
                        "param_value": tokens[1],
                        "variant": variant,
                        "values": row_values,
                        "source": txt.name,
                        "status": "PASSED",
                    }
                )
    return rows


def parse_results_dir(output_dir: Path) -> list:
    rows = []
    for txt in tiletranslation_result_files(output_dir):
        if not txt.exists():
            continue
        file_rows = parse_results_file(txt)
        status = "FAILED" if txt.read_text(errors="replace").startswith("BENCHMARK FAILED") else "PASSED"
        for row in file_rows:
            row["status"] = status
        rows.extend(file_rows)
    return rows


def write_csv(rows: list, csv_path: Path) -> None:
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADERS)
        for index, row in enumerate(rows, start=2):  # data starts at row 2
            values = row["values"]
            writer.writerow(
                [
                    row["benchmark"],
                    row["unit"],
                    row["param_name"],
                    row["param_value"],
                    row["variant"],
                    values.get("CuTile") if values.get("CuTile") is not None else "",
                    values.get("DropTokens") if values.get("DropTokens") is not None else "",
                    values.get("TileTranslation") if values.get("TileTranslation") is not None else "",
                    values.get("PyTorch") if values.get("PyTorch") is not None else "",
                    overhead_formula(OVERHEAD_TARGETS["DropTokens"], index),
                    overhead_formula(OVERHEAD_TARGETS["TileTranslation"], index),
                    overhead_formula(OVERHEAD_TARGETS["PyTorch"], index),
                    row["source"],
                    row["status"],
                ]
            )


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = output_dir.resolve()

    rows = parse_results_dir(output_dir)
    if not rows:
        print(f"No tiletranslation benchmark tables found in {output_dir}")
        sys.exit(1)

    csv_path = output_dir / CSV_NAME
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} result rows to {csv_path}")


if __name__ == "__main__":
    main()
