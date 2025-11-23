# src/scoreline_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor

def compute_form_features(df, team, n=5, decay=0.8):
    """
    Compute form-based weighted average goals for and against for a team over its last n matches.
    df: historic matches with Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals.
    """
    df = df.sort_values("Date")
    # filter matches where team played
    team_matches = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    weights = np.array([decay ** i for i in range(len(team_matches)-1, -1, -1)])
    # compute goals for and against
    goals_for = []
    goals_against = []
    for _, row in team_matches.iterrows():
        if row.HomeTeam == team:
            goals_for.append(row.HomeGoals)
            goals_against.append(row.AwayGoals)
        else:
            goals_for.append(row.AwayGoals)
            goals_against.append(row.HomeGoals)
    gf = np.dot(goals_for, weights) / weights.sum()
    ga = np.dot(goals_against, weights) / weights.sum()
    return gf, ga

def build_scoreline_dataset(df):
    """
    Build a training dataset with features + targets (home_goals, away_goals).
    """
    rows = []
    for idx, row in df.iterrows():
        home = row.HomeTeam
        away = row.AwayTeam
        gf_home, ga_home = compute_form_features(df[df.Date < row.Date], home)
        gf_away, ga_away = compute_form_features(df[df.Date < row.Date], away)
        rows.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "Form_Home_GF": gf_home,
            "Form_Home_GA": ga_home,
            "Form_Away_GF": gf_away,
            "Form_Away_GA": ga_away,
            "HomeGoals": row.HomeGoals,
            "AwayGoals": row.AwayGoals
        })
    return pd.DataFrame(rows)

def train_goal_models(df):
    ds = build_scoreline_dataset(df)
    features = ["Form_Home_GF", "Form_Home_GA", "Form_Away_GF", "Form_Away_GA"]
    X = ds[features]
    y_home = ds["HomeGoals"]
    y_away = ds["AwayGoals"]

    model_home = PoissonRegressor(alpha=1e-6, max_iter=300).fit(X, y_home)
    model_away = PoissonRegressor(alpha=1e-6, max_iter=300).fit(X, y_away)
    return model_home, model_away

def predict_expected_goals(model_home, model_away, historic_df, home_team, away_team):
    gf_h, ga_h = compute_form_features(historic_df, home_team)
    gf_a, ga_a = compute_form_features(historic_df, away_team)
    Xpred = np.array([[gf_h, ga_h, gf_a, ga_a]])
    lambda_home = model_home.predict(Xpred)[0]
    lambda_away = model_away.predict(Xpred)[0]
    return lambda_home, lambda_away

def scoreline_probabilities(lambda_home, lambda_away, max_goals=5):
    """
    Compute joint probability P(X = i, Y = j) assuming independence (simplest).
    Returns a dict {(i, j): prob}.
    """
    probs = {}
    from math import exp, factorial
    for i in range(0, max_goals+1):
        for j in range(0, max_goals+1):
            p_i = exp(-lambda_home) * lambda_home**i / factorial(i)
            p_j = exp(-lambda_away) * lambda_away**j / factorial(j)
            probs[(i, j)] = p_i * p_j
    return probs
