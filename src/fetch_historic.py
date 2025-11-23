import os
import requests
import pandas as pd
from dotenv import load_dotenv

def fetch_historic_data(season=2024):
    """Fetch completed Premier League fixtures from API-Football"""
    load_dotenv()
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        raise ValueError("❌ Missing API key. Add it in .env as API_FOOTBALL_KEY")

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {
        "league": 39,      # Premier League
        "season": season,
        "status": "FT"     # Full-time (completed games only)
    }

    print(f"📡 Fetching completed matches for {season} season...")
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed ({response.status_code}): {response.text}")

    data = response.json().get("response", [])
    if not data:
        print("⚠️ No data received from API.")
        return None

    matches = []
    for match in data:
        matches.append({
            "Date": match["fixture"]["date"].split("T")[0],
            "HomeTeam": match["teams"]["home"]["name"],
            "AwayTeam": match["teams"]["away"]["name"],
            "HomeGoals": match["goals"]["home"],
            "AwayGoals": match["goals"]["away"],
            "Result": (
                "H" if match["goals"]["home"] > match["goals"]["away"]
                else "A" if match["goals"]["home"] < match["goals"]["away"]
                else "D"
            )
        })

    df = pd.DataFrame(matches)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    os.makedirs("data", exist_ok=True)
    out_path = "data/historic_data.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} matches to {out_path}")
    return df

if __name__ == "__main__":
    fetch_historic_data()
