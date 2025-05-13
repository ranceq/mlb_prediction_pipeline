import argparse
import os
import pandas as pd

def clean_retrosheet(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df['home_team'] = df['home_team'].str.strip().str.upper()
    df['away_team'] = df['away_team'].str.strip().str.upper()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrosheet-dir', required=True)
    parser.add_argument('--statcast-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for year in os.listdir(args.retrosheet_dir):
        rs_path = os.path.join(args.retrosheet_dir, year, f"retrosheet_{year}.csv")
        sc_path = os.path.join(args.statcast_dir, f"statcast_{year}.csv")
        df_rs = clean_retrosheet(rs_path)
        df_sc = pd.read_csv(sc_path)
        df = df_rs.merge(df_sc, how='left', left_on=['date','home_team'], right_on=['game_date','batting_team'])
        out_path = os.path.join(args.out_dir, f"merged_{year}.parquet")
        df.to_parquet(out_path)
        print(f"Wrote {out_path}")