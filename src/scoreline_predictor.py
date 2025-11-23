# src/scoreline_predictor.py

import numpy as np
import pandas as pd
from scipy.stats import poisson

def compute_strengths(df: pd.DataFrame, team: str, opponent: str,
                      recent_n: int = 8, h2h_n: int = 4,
                      decay: float = 0.7):
    """
    Compute attack and defense strength for a team based on recent form + H2H.

    - df: historic dataframe with columns ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']
    - team: name of the team for which to compute strength
    - opponent: opposing team (for H2H)
    - recent_n: number of most recent matches to use
    - h2h_n: number of recent head-to-head matches
    - decay: weight decay factor for recent form (more recent have higher weight)
    """

    # Filter recent matches of `team` as home or away
    team_matches = df[
        (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    ].sort_values("Date", ascending=False).head(recent_n)

    # Compute weights for decay: most recent weight = 1, then decay^i
    weights = np.array([decay**i for i in range(len(team_matches))])
    weights = weights / weights.sum()

    # Compute average goals scored (attack) and conceded (defense) in recent form
    goals_for = []
    goals_against = []

    for i, row in enumerate(team_matches.itertuples()):
        w = weights[i]
        if row.HomeTeam == team:
            goals_for.append(w * row.HomeGoals)
            goals_against.append(w * row.AwayGoals)
        else:
            goals_for.append(w * row.AwayGoals)
            goals_against.append(w * row.HomeGoals)

    avg_for_recent = sum(goals_for)
    avg_against_recent = sum(goals_against)

    # Now head-to-head
    h2h = df[
        ((df["HomeTeam"] == team) & (df["AwayTeam"] == opponent)) |
        ((df["HomeTeam"] == opponent) & (df["AwayTeam"] == team))
    ].sort_values("Date", ascending=False).head(h2h_n)

    if not h2h.empty:
        # simple unweighted average for H2H
        gf_h2h = []
        ga_h2h = []
        for row in h2h.itertuples():
            if row.HomeTeam == team:
                gf_h2h.append(row.HomeGoals)
                ga_h2h.append(row.AwayGoals)
            else:
                gf_h2h.append(row.AwayGoals)
                ga_h2h.append(row.HomeGoals)
        avg_for_h2h = np.mean(gf_h2h)
        avg_against_h2h = np.mean(ga_h2h)
    else:
        # fallback to recent only
        avg_for_h2h = avg_for_recent
        avg_against_h2h = avg_against_recent

    # Combine recent + H2H strengths (you can give more weight to recent or H2H as needed)
    # Here: 60% recent, 40% H2H
    attack_strength = 0.6 * avg_for_recent + 0.4 * avg_for_h2h
    defense_weakness = 0.6 * avg_against_recent + 0.4 * avg_against_h2h

    return attack_strength, defense_weakness

def predict_score_distribution(df: pd.DataFrame, home: str, away: str,
                               home_advantage_factor: float = 1.1,
                               max_goals: int = 5):
    """
    Predict goal distribution for home and away using a Poisson model.
    Returns a DataFrame of scoreline probabilities and expected goals.
    """

    # Compute strengths
    home_attack, home_defense = compute_strengths(df, home, away)
    away_attack, away_defense = compute_strengths(df, away, home)

    # Estimate lambda (expected goals) for each
    lambda_home = home_attack * away_defense * home_advantage_factor
    lambda_away = away_attack * home_defense

    # Compute Poisson probabilities for goal counts
    home_goals_range = np.arange(0, max_goals + 1)
    away_goals_range = np.arange(0, max_goals + 1)

    probs = []
    for hg in home_goals_range:
        for ag in away_goals_range:
            p = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)
            probs.append({
                "HomeGoals": hg,
                "AwayGoals": ag,
                "Probability": p
            })

    prob_df = pd.DataFrame(probs)
    # Sort by probability descending
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    # Compute expected goals
    exp_home = sum(prob_df["HomeGoals"] * prob_df["Probability"])
    exp_away = sum(prob_df["AwayGoals"] * prob_df["Probability"])

    return prob_df, (exp_home, exp_away)

def get_top_scorelines(prob_df: pd.DataFrame, top_n: int = 5):
    """
    Return the top N scorelines by probability.
    """
    return prob_df.head(top_n)
