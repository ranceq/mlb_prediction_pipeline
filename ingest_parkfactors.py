import argparse
import os
import time
import pandas as pd
import requests
from requests.exceptions import HTTPError

def fetch_park_factors(year: int, retries: int = 3, backoff: int = 5) -> pd.DataFrame:
    """
    Fetch the park factors table from FanGraphs Guts! for a given year,
    with retry logic for transient HTTP errors.
    """
    url = f"https://www.fangraphs.com/guts.aspx?type=pf&year={year}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        )
    }
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            tables = pd.read_html(response.text)
            df = tables[0]
            df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
            if 'pf' in df.columns:
                factor_col = 'pf'
            elif 'park_factor' in df.columns:
                factor_col = 'park_factor'
            else:
                raise ValueError("Could not find park factor column in FanGraphs table.")
            result = df[['park', factor_col]].rename(columns={factor_col: 'park_factor'})
            return result
        except HTTPError as e:
            attempt += 1
            print(f"HTTPError on {year}, attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(backoff)
            else:
                raise
        except Exception as e:
            print(f"Error fetching park factors for {year}: {e}")
            raise


def main(start_year: int, end_year: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for year in range(start_year, end_year + 1):
        print(f"Fetching park factors for {year}...")
        df = fetch_park_factors(year)
        df['year'] = year
        out_path = os.path.join(out_dir, f"park_factors_{year}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Ingest FanGraphs park factors by year with retry logic."
    )
    parser.add_argument('--start-year', type=int, required=True, help="First year to fetch (e.g., 2015)")
    parser.add_argument('--end-year', type=int, required=True, help="Last year to fetch (e.g., 2024)")
    parser.add_argument('--out-dir', required=True, help="Directory to save CSVs")
    args = parser.parse_args()
    main(args.start_year, args.end_year, args.out_dir)
