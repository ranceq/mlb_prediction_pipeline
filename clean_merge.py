import argparse
import os
import pandas as pd

def clean_retrosheet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'], low_memory=False)
    df['home_team'] = df['home_team'].str.strip().str.upper()
    df['away_team'] = df['away_team'].str.strip().str.upper()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean and merge Retrosheet + Statcast data by year.")
    parser.add_argument('--retrosheet-dir', required=True)
    parser.add_argument('--statcast-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for fname in os.listdir(args.retrosheet_dir):
        if not fname.endswith('.csv') or 'retrosheet_' not in fname:
            continue
        year = fname.split('_')[1].split('.')[0]
        rs = clean_retrosheet(os.path.join(args.retrosheet_dir, fname))
        sc = pd.read_csv(os.path.join(args.statcast_dir, f"statcast_{year}.csv"), low_memory=False)
        merged = rs.merge(
            sc,
            how='left',
            left_on=['date', 'home_team'],
            right_on=['game_date', 'batting_team']
        )
        out_path = os.path.join(args.out_dir, f"merged_{year}.parquet")
        merged.to_parquet(out_path)
        print(f"Wrote {out_path} ({len(merged)} rows)")