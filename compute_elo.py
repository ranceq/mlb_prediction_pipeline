import argparse
import os
import pandas as pd

def compute_elo(df, K=25, HFA=100):
    teams = pd.concat([df.home_team, df.away_team]).unique()
    ratings = {t:1500 for t in teams}
    rows = []
    for _, r in df.sort_values('date').iterrows():
        h,a = r.home_team, r.away_team
        Rh, Ra = ratings[h], ratings[a]
        Eh = 1/(1+10**((Ra-Rh+HFA)/400))
        Sh = 1 if r.home_score > r.away_score else 0
        ratings[h] += K*(Sh - Eh)
        ratings[a] += K*((1-Sh) - (1-Eh))
        rows.append({'game_id':r.game_id, 'elo_home':Rh, 'elo_away':Ra})
    return pd.DataFrame(rows)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--K', type=int, default=25)
    parser.add_argument('--HFA', type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for fname in os.listdir(args.merged_dir):
        if fname.startswith('merged_') and fname.endswith('.parquet'):
            year = fname.split('_')[1].split('.')[0]
            df = pd.read_parquet(os.path.join(args.merged_dir, fname))
            elo_df = compute_elo(df, args.K, args.HFA)
            out_path = os.path.join(args.out_dir, f"elo_{year}.parquet")
            elo_df.to_parquet(out_path)
            print(f"Wrote {out_path}")