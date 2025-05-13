import argparse
import os
import pandas as pd

def engineer_features(df_rs, df_elo):
    df = df_rs.merge(df_elo, on='game_id')
    # Example: run_diff_last3
    df['run_diff'] = df.home_score - df.away_score
    df['run_diff_last3'] = df.groupby('home_team')['run_diff'].transform(lambda x: x.rolling(3).mean())
    # Add streak and rest_days similarly...
    return df

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged-dir', required=True)
    parser.add_argument('--elo-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for fname in os.listdir(args.merged_dir):
        year = fname.split('_')[1].split('.')[0]
        df_rs = pd.read_parquet(os.path.join(args.merged_dir, fname))
        df_elo = pd.read_parquet(os.path.join(args.elo_dir, f"elo_{year}.parquet"))
        df_feat = engineer_features(df_rs, df_elo)
        out_path = os.path.join(args.out_dir, f"features_{year}.parquet")
        df_feat.to_parquet(out_path)
        print(f"Wrote {out_path}")