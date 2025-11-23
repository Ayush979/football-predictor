import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def build_team_stats(df):
    """Compute historical averages for each team."""
    home_stats = df.groupby("HomeTeam").agg({
        "HomeGoals": "mean",
        "AwayGoals": "mean"
    }).reset_index().rename(columns={
        "HomeTeam": "Team",
        "HomeGoals": "Home_AvgGoalsFor",
        "AwayGoals": "Home_AvgGoalsAgainst"
    })

    away_stats = df.groupby("AwayTeam").agg({
        "AwayGoals": "mean",
        "HomeGoals": "mean"
    }).reset_index().rename(columns={
        "AwayTeam": "Team",
        "AwayGoals": "Away_AvgGoalsFor",
        "HomeGoals": "Away_AvgGoalsAgainst"
    })

    stats = pd.merge(home_stats, away_stats, on="Team", how="outer").fillna(0)
    return stats


def train_model():
    print("⚙️ Loading data...")
    hist_path = "data/historic_data.csv"
    upcoming_path = "data/upcoming_fixtures.csv"

    if not os.path.exists(hist_path) or not os.path.exists(upcoming_path):
        raise FileNotFoundError("Missing required data files in /data/ folder")

    df = pd.read_csv(hist_path)
    upcoming = pd.read_csv(upcoming_path)

    # Ensure proper date format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    upcoming["Date"] = pd.to_datetime(upcoming["Date"], errors="coerce")

    latest_fixture_date = upcoming["Date"].min()
    df = df[df["Date"] < latest_fixture_date]  # only use past matches

    print(f"📅 Using {len(df)} past matches for training (before {latest_fixture_date.date()})")

    # Encode result as label
    df["ResultCode"] = df["Result"].map({"H": 0, "D": 1, "A": 2})

    # Compute per-team stats
    team_stats = build_team_stats(df)

    df = df.merge(team_stats.add_prefix("Home_"), left_on="HomeTeam", right_on="Home_Team", how="left")
    df = df.merge(team_stats.add_prefix("Away_"), left_on="AwayTeam", right_on="Away_Team", how="left")

    df["GoalDiff"] = df["Home_Home_AvgGoalsFor"] - df["Away_Away_AvgGoalsFor"]
    df["HomeAdvantage"] = 1

    X = df[[
        "Home_Home_AvgGoalsFor", "Home_Home_AvgGoalsAgainst",
        "Away_Away_AvgGoalsFor", "Away_Away_AvgGoalsAgainst",
        "GoalDiff", "HomeAdvantage"
    ]]
    y = df["ResultCode"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("✅ Model training complete!\n")
    print(classification_report(y_test, preds, target_names=["Home Win", "Draw", "Away Win"]))

    with open("models/match_predictor.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n💾 Model saved → models/match_predictor.pkl")


if __name__ == "__main__":
    train_model()
