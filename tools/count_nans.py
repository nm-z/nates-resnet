#!/usr/bin/env python3
import glob
from pathlib import Path
import pandas as pd
import numpy as np

VNA_DIR = Path(__file__).resolve().parents[1] / 'VNA-D4'
TEMP_CSV = Path(__file__).resolve().parents[1] / 'temp_readings-D4.csv'

def main():
    files = sorted(glob.glob(str(VNA_DIR / '*.csv')))
    if not files:
        print(f'No VNA CSV files found in {VNA_DIR}')
        return

    totals: dict[str, int] = {}
    rows_total = 0
    files_count = 0

    for idx, fp in enumerate(files, 1):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f'Failed to read {fp}: {e}')
            continue
        rows_total += len(df)
        files_count += 1
        for col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            nans = int(s.isna().sum())
            totals[col] = totals.get(col, 0) + nans
        if idx % 250 == 0:
            print(f'Progress: processed {idx}/{len(files)} files...')

    print(f'VNA files processed: {files_count}, total rows: {rows_total}')
    print('NaN totals per VNA column:')
    for k in sorted(totals.keys()):
        print(f'{k}: {totals[k]}')
    print(f'Total NaNs across all VNA numeric columns: {sum(totals.values())}')

    # Temperature CSV
    try:
        tdf = pd.read_csv(TEMP_CSV)
        t_nan_temp = int(pd.to_numeric(tdf.get('temp_c'), errors='coerce').isna().sum())
        t_nan_timestamp = int(pd.to_datetime(tdf.get('timestamp'), errors='coerce').isna().sum())
        print(f'NaNs in temp_readings-D4.csv temp_c: {t_nan_temp}')
        print(f'Invalid timestamps in temp_readings-D4.csv: {t_nan_timestamp}')
    except Exception as e:
        print(f'Could not read temperature CSV at {TEMP_CSV}: {e}')

if __name__ == '__main__':
    main()


