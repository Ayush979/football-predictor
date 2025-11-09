import pandas as pd
import os

def fetch_historic_data():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",  # 2023-24
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",  # 2024-25
    ]
    
    dfs = []
    for url in urls:
        print(f"📥 Downloading: {url}")
        df = pd.read_csv(url)
        dfs.append(df)
    
    df_all = pd.concat(dfs)
    df_all = df_all[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
    df_all.columns = ["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"]
    
    os.makedirs("data", exist_ok=True)
    df_all.to_csv("data/historic_data.csv", index=False)
    print(f"✅ Saved historic data: {df_all.shape[0]} matches")

if __name__ == "__main__":
    fetch_historic_data()
# To run this script, ensure you have pandas installed:
# pip install pandas