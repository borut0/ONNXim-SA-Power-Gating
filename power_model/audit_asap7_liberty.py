#!/usr/bin/env python3
import gzip
import pathlib
import re
import sys


def read_text(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as stream:
        return stream.read()


if len(sys.argv) != 2:
    raise SystemExit("usage: audit_asap7_liberty.py ASAP7_LIB_DIRECTORY")

root = pathlib.Path(sys.argv[1])
files = sorted(set(root.glob("*.lib")) | set(root.glob("*.lib.gz")))
if not files:
    raise SystemExit(f"no Liberty files found in {root}")

for path in files:
    text = read_text(path)
    corner = next((value for value in ("TT", "SS", "FF")
                   if re.search(rf"(?:^|_){value}(?:_|\.)", path.name)), "UNKNOWN")
    voltage = re.findall(r"\bvoltage\s*:\s*([^;]+);", text)
    temperature = re.findall(r"\btemperature\s*:\s*([^;]+);", text)
    time_unit = re.findall(r'\btime_unit\s*:\s*"([^"]+)"', text)
    voltage_unit = re.findall(r'\bvoltage_unit\s*:\s*"([^"]+)"', text)
    print(
        f"LIBERTY_AUDIT file={path.name} corner={corner} "
        f"internal_power={text.count('internal_power')} "
        f"rise_power={text.count('rise_power')} "
        f"fall_power={text.count('fall_power')} "
        f"leakage_power={text.count('leakage_power')} "
        f"cell_leakage_power={text.count('cell_leakage_power')} "
        f"time_unit={time_unit[:1]} voltage_unit={voltage_unit[:1]} "
        f"voltage={voltage[:1]} temperature={temperature[:1]}"
    )
