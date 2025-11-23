# scripts/update_dataset.py
"""
Update local data/historic_data.csv by fetching latest results.
Supports:
 - API-Football (recommended; requires API_KEY environment variable)
 - fallback to football-data.co.uk CSV for E0 (Premier League)
Usage:
    python scripts/update_dataset.py --season 2526 --append
"""
import os
import argparse
import pandas as pd
import requests
from datetime import datetime

DATA_PATH = "data/historic_data.csv"
os.makedirs("data", exist_ok=True)

def read_local():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    else:
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result"])

def save_local(df):
    df.to_csv(DATA_PATH, index=False)
    print("Saved", DATA_PATH)

def fetch_from_api_football(api_key, league=39, season=None):
    # API-Football: https://dashboard.api-football.com/
    headers = {"x-rapidapi-key": api_key}
    base = "https://v3.football.api-sports.io"
    params = {"league": league}
    if season:
        params["season"] = season
    try:
        r = requests.get(base + "/fixtures", headers=headers, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        fixtures = j.get("response", [])
        out = []
        for f in fixtures:
            # ensure only finished matches
            if f.get("fixture", {}).get("status", {}).get("short") not in ("FT","AET","PEN"):
                continue
            home = f["teams"]["home"]["name"]
            away = f["teams"]["away"]["name"]
            goals_home = f["goals"]["home"]
            goals_away = f["goals"]["away"]
            date_raw = f["fixture"]["date"]
            date = pd.to_datetime(date_raw)
            if goals_home is None or goals_away is None:
                continue
            if goals_home > goals_away:
                res = "H"
            elif goals_home == goals_away:
                res = "D"
            else:
                res = "A"
            out.append({"Date": date, "HomeTeam": home, "AwayTeam": away, "HomeGoals": goals_home, "AwayGoals": goals_away, "Result": res})
        return pd.DataFrame(out)
    except Exception as e:
        print("API-Football fetch failed:", e)
        return None

def fetch_from_football_data_csv(season_code):
    """
    Download football-data.co.uk CSV for E0 (Premier League).
    season_code example: 2425 for 2024/25, 2526 for 2025/26
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        from io import StringIO
        tmp = pd.read_csv(StringIO(r.text))
        # football-data columns vary; try to extract Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
        mapping = {}
        if "Date" in tmp.columns and "HomeTeam" in tmp.columns:
            df = tmp.rename(columns={"FTHG":"HomeGoals","FTAG":"AwayGoals","FTR":"Result"})[["Date","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result"]]
            # parse Date - football-data uses dd/mm/yy or dd/mm/yyyy
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            return df
        else:
            print("CSV format unexpected.")
            return None
    except Exception as e:
        print("football-data CSV fetch failed:", e)
        return None

def append_new(existing, fetched):
    if fetched is None or fetched.empty:
        return existing
    existing = existing.copy()
    # normalize datetimes
    if "Date" in existing.columns:
        existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
    fetched["Date"] = pd.to_datetime(fetched["Date"], errors="coerce")
    # remove duplicates by Date+Home+Away
    existing_keys = set(existing.apply(lambda r: (pd.to_datetime(r["Date"]).date(), r["HomeTeam"], r["AwayTeam"]), axis=1).tolist())
    new_rows = []
    for _, r in fetched.iterrows():
        key = (pd.to_datetime(r["Date"]).date(), r["HomeTeam"], r["AwayTeam"])
        if key not in existing_keys:
            new_rows.append(r)
    if new_rows:
        appended = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
        appended = appended.sort_values("Date").reset_index(drop=True)
        return appended
    return existing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", help="API-Football key (optional)", default=None)
    parser.add_argument("--season", help="season code for football-data (e.g. 2526)", default=None)
    args = parser.parse_args()

    local = read_local()
    fetched = None
    if args.api_key:
        print("Trying API-Football...")
        fetched = fetch_from_api_football(args.api_key, season=args.season)
        print("Fetched", len(fetched) if fetched is not None else 0, "rows from API-Football.")
        season = 2526
    if (fetched is None or fetched.empty) and args.season:
        print("Trying football-data CSV fallback...")
        fetched = fetch_from_football_data_csv(args.season)
    if fetched is None:
        print("No data fetched. Exiting.")
    else:
        print("Data fetched successfully.")
        print(f"Fetched {len(fetched)} rows.")
        updated = append_new(local, fetched)
        save_local(updated)
