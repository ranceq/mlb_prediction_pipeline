import argparse
import os
from pybaseball import statcast

def fetch_statcast(start, end, out_dir):
    years = range(int(start[:4]), int(end[:4]) + 1)
    for year in years:
        s = f"{year}-01-01"
        e = f"{year}-12-31"
        df = statcast(s, e)
        out_path = os.path.join(out_dir, f"statcast_{year}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} with {len(df)} rows")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fetch_statcast(args.start, args.end, args.out_dir)