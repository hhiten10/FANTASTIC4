"""
load_data.py - command-line data loader (guide 4.7).

Loads a CSV in the Annex B schema (e.g. judge test data or the late drop) and runs the validation checks.

    python load_data.py data/late_drop/sensor_readings_2026-10.csv
    python load_data.py some_file.csv --table work_orders
    python load_data.py                          # only run the validation checks

The file is cleaned (column names fixed, times to UTC, bad rows and duplicates dropped)
and stored in data/raw as <table>__<name>.csv, where training and the API pick it up.
"""
import argparse
import json
from pathlib import Path

from src.config import TABLES
from src.ingest import load_all, read_csv, save_new_file
from src.validation import print_checks, validate


def main():
    """Load each given file, then print the validation checks for all loaded data."""
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--table", choices=list(TABLES))
    a = ap.parse_args()
    for f in a.files:
        # Guess the table from the file name (e.g. "sensor_readings_2026-10.csv") unless --table is given.
        table = a.table or next((t for t in TABLES if f.stem.startswith(t)), None)
        if not table:
            raise SystemExit(f"cannot guess the table for {f.name}; pass --table")
        print(f.name, "->", table, json.dumps(save_new_file(read_csv(f), table, f.name)))

    print("\nValidation checks")
    print_checks(validate(load_all()))


if __name__ == "__main__":
    main()
