import pandas as pd
import pickle
import matplotlib.pyplot as plt

def build_team_stats(df):
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

    return pd.merge(home_stats, away_stats, on="Team", how="outer").fillna(0)


def predict_fixtures():
    print("⚙️ Loading model and data...")
    with open("models/match_predictor.pkl", "rb") as f:
        model = pickle.load(f)

    historic = pd.read_csv("data/historic_data.csv")
    upcoming = pd.read_csv("data/upcoming_fixtures.csv")

    historic["Date"] = pd.to_datetime(historic["Date"], errors="coerce", dayfirst=True)
    upcoming["Date"] = pd.to_datetime(upcoming["Date"], errors="coerce")

    team_stats = build_team_stats(historic)

    merged = upcoming.merge(team_stats.add_prefix("Home_"), left_on="HomeTeam", right_on="Home_Team", how="left")
    merged = merged.merge(team_stats.add_prefix("Away_"), left_on="AwayTeam", right_on="Away_Team", how="left")
    merged.fillna(team_stats.mean(numeric_only=True), inplace=True)

    merged["GoalDiff"] = merged["Home_Home_AvgGoalsFor"] - merged["Away_Away_AvgGoalsFor"]
    merged["HomeAdvantage"] = 1

    X_pred = merged[[
        "Home_Home_AvgGoalsFor", "Home_Home_AvgGoalsAgainst",
        "Away_Away_AvgGoalsFor", "Away_Away_AvgGoalsAgainst",
        "GoalDiff", "HomeAdvantage"
    ]]

    y_pred_proba = model.predict_proba(X_pred)
    y_pred = model.predict(X_pred)

    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    merged["Predicted"] = [label_map[i] for i in y_pred]

    proba_df = pd.DataFrame(y_pred_proba, columns=[label_map[i] for i in range(3)])
    merged = pd.concat([merged, proba_df], axis=1)

    merged.to_csv("data/predictions.csv", index=False)
    print("✅ Predictions saved → data/predictions.csv")

    # Plot
    counts = merged["Predicted"].value_counts()
    plt.bar(counts.index, counts.values, color=["#2ecc71", "#f1c40f", "#e74c3c"])
    plt.title("Premier League Predictions Distribution")
    plt.xlabel("Predicted Result")
    plt.ylabel("Count")
    plt.show()

    print("\n📊 Predicted Match Outcomes:\n")
    print(merged[["Date", "HomeTeam", "AwayTeam", "Predicted", "Home Win", "Draw", "Away Win"]])
    return merged


if __name__ == "__main__":
    predict_fixtures()
