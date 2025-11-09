# src/app.py
# --- Windows curses fallback patch (Python 3.13 compatibility) ---
import sys, types
if sys.platform == "win32":
    sys.modules["curses"] = types.ModuleType("curses")
    sys.modules["_curses"] = types.ModuleType("_curses")

#from curses import raw
from multiprocessing import pool
import streamlit as st
import pandas as pd
import pickle, os, datetime, base64, hashlib
from PIL import Image
import plotly.graph_objects as go
import numpy as np

# ---------------------------
# CONFIG / HELPERS
# ---------------------------
st.set_page_config("Premier League Predictor — Story Mode", layout="wide")
st.title("🏆 Premier League Match Predictor — Story Mode")

# Built-in approximate colors for Premier League clubs (can extend)
TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#7A003C", "Bournemouth": "#DA291C", "Brentford": "#ED1B24",
    "Brighton": "#0057B8", "Burnley": "#6C1D45", "Chelsea": "#034694", "Crystal Palace": "#1B458A",
    "Everton": "#003399", "Fulham": "#000000", "Liverpool": "#C8102E", "Luton": "#F57E20",
    "Man City": "#6CABDD", "Man United": "#DA291C", "Newcastle": "#000000", "Norwich": "#009933",
    "Sheffield United": "#EE2737", "Southampton": "#D71920", "Tottenham": "#132257", "Wolves": "#FDB913",
    "Leicester": "#0053A0", "West Ham": "#7A263A", "Nott'm Forest": "#E51636", "Watford": "#FDBB30",
    # Add others as needed. Fallback color chosen below.
}

def safe_color(team):
    return TEAM_COLORS.get(team, "#6c757d")

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_data
def load_assets():
    model_path = "models/match_predictor.pkl"
    hist_path = "data/historic_data.csv"
    if not os.path.exists(model_path):
        st.error("Model not found. Run src/train_model.py first.")
        st.stop()
    if not os.path.exists(hist_path):
        st.error("Missing data/historic_data.csv")
        st.stop()
    model = pickle.load(open(model_path, "rb"))
    hist = pd.read_csv(hist_path)
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    # keep expected columns: Date,HomeTeam,AwayTeam,HomeGoals,AwayGoals,Result
    return model, hist

model, historic_df = load_assets()
team_stats = None

def build_team_stats(df):
    home = df.groupby("HomeTeam").agg({"HomeGoals": "mean", "AwayGoals": "mean"}).reset_index()
    home.columns = ["Team", "Home_AvgGoalsFor", "Home_AvgGoalsAgainst"]
    away = df.groupby("AwayTeam").agg({"AwayGoals": "mean", "HomeGoals": "mean"}).reset_index()
    away.columns = ["Team", "Away_AvgGoalsFor", "Away_AvgGoalsAgainst"]
    stats = pd.merge(home, away, on="Team", how="outer").fillna(0)
    return stats

team_stats = build_team_stats(historic_df)
teams = sorted(team_stats["Team"].dropna().unique())
logo_dir = "data/team_logos"

# session state for selections
if "home_team" not in st.session_state: st.session_state["home_team"] = None
if "away_team" not in st.session_state: st.session_state["away_team"] = None

# ---------------------------
# Layout Tabs
# ---------------------------
tabs = st.tabs(["🎯 Predictor", "📜 Match Story Mode", "📂 Batch Predictions"])

# ---------- TAB 0: Predictor (quick single match) ----------
with tabs[0]:
    st.subheader("Single Match Predictor (click a logo below to select)")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 🏠 Home Team")
        home_cols = st.columns(5)
        for i, team in enumerate(teams):
            img_path = os.path.join(logo_dir, f"{team}.png")
            if not os.path.exists(img_path):
                continue
            with home_cols[i % 5]:
                if st.button(team, key=f"home_btn_{team}"):
                    # toggle selection
                    st.session_state["home_team"] = None if st.session_state["home_team"] == team else team
                # show image with border highlight
                border = "3px solid " + safe_color(team) if st.session_state["home_team"] == team else "1px solid #ddd"
                st.markdown(f"<div style='text-align:center; border:{border}; padding:6px; border-radius:8px;'>"
                            f"<img src='data:image/png;base64,{img_to_base64(img_path)}' width=80><div style='font-size:12px'>{team}</div></div>",
                            unsafe_allow_html=True)
    with cols[1]:
        st.markdown("### 🚗 Away Team")
        away_cols = st.columns(5)
        for i, team in enumerate(teams):
            img_path = os.path.join(logo_dir, f"{team}.png")
            if not os.path.exists(img_path):
                continue
            with away_cols[i % 5]:
                if st.button(team, key=f"away_btn_{team}"):
                    st.session_state["away_team"] = None if st.session_state["away_team"] == team else team
                border = "3px solid " + safe_color(team) if st.session_state["away_team"] == team else "1px solid #ddd"
                st.markdown(f"<div style='text-align:center; border:{border}; padding:6px; border-radius:8px;'>"
                            f"<img src='data:image/png;base64,{img_to_base64(img_path)}' width=80><div style='font-size:12px'>{team}</div></div>",
                            unsafe_allow_html=True)

    home_team = st.session_state["home_team"]
    away_team = st.session_state["away_team"]

    if home_team and away_team:
        if home_team == away_team:
            st.warning("Please select two different teams.")
        else:
            st.success(f"Selected: **{home_team} vs {away_team}**")

            # prepare features (reuse same features as training)
            h_row = team_stats.loc[team_stats["Team"] == home_team].iloc[0]
            a_row = team_stats.loc[team_stats["Team"] == away_team].iloc[0]
            X = pd.DataFrame([{
                "Home_Home_AvgGoalsFor": h_row["Home_AvgGoalsFor"],
                "Home_Home_AvgGoalsAgainst": h_row["Home_AvgGoalsAgainst"],
                "Away_Away_AvgGoalsFor": a_row["Away_AvgGoalsFor"],
                "Away_Away_AvgGoalsAgainst": a_row["Away_AvgGoalsAgainst"],
                "GoalDiff": h_row["Home_AvgGoalsFor"] - a_row["Away_AvgGoalsFor"],
                "HomeAdvantage": 1
            }])

            probs = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            result = labels[pred]

            # Card with color
            color_map = {"Home Win": safe_color(home_team), "Draw": "#f1c40f", "Away Win": safe_color(away_team)}
            st.markdown(f"<div style='background:{color_map[result]}; padding:16px; border-radius:10px; color:white;'>"
                        f"<h3 style='margin:0'>{home_team} vs {away_team} — Predicted: <b>{result}</b></h3>"
                        f"<div style='margin-top:8px'>Home: {probs[0]*100:.1f}% &nbsp;&nbsp; Draw: {probs[1]*100:.1f}% &nbsp;&nbsp; Away: {probs[2]*100:.1f}%</div>"
                        f"</div>", unsafe_allow_html=True)

# ---------- TAB 1: Match Story Mode ----------
with tabs[1]:
    st.subheader("📜 Match Story Mode — Rivalry Insights & Timeline")
    home_team = st.session_state["home_team"]
    away_team = st.session_state["away_team"]

    if not (home_team and away_team):
        st.info("Select Home and Away teams in the Predictor tab (or pick below) to see Story Mode.")
        # quick pick fallback
        c1, c2 = st.columns(2)
        with c1:
            pick_home = st.selectbox("Pick Home (for Story Mode)", [""] + teams)
        with c2:
            pick_away = st.selectbox("Pick Away (for Story Mode)", [""] + teams)
        if pick_home and pick_away:
            st.session_state["home_team"], st.session_state["away_team"] = pick_home, pick_away
            st.experimental_rerun()
    else:
        st.markdown(f"### 🔎 Rivalry: {home_team} vs {away_team}")
        # compute H2H
        h2h = historic_df[
            ((historic_df["HomeTeam"] == home_team) & (historic_df["AwayTeam"] == away_team)) |
            ((historic_df["HomeTeam"] == away_team) & (historic_df["AwayTeam"] == home_team))
        ].dropna(subset=["HomeGoals","AwayGoals"]).sort_values("Date", ascending=False)

        # Summary stats
        total_matches = len(h2h)
        total_goals = int((h2h["HomeGoals"] + h2h["AwayGoals"]).sum()) if total_matches>0 else 0
        avg_goals = (total_goals / total_matches) if total_matches>0 else 0.0
        home_wins = int(((h2h["Result"]=="H") & (h2h["HomeTeam"]==home_team)).sum() + ((h2h["Result"]=="A") & (h2h["AwayTeam"]==home_team)).sum())
        away_wins = int(((h2h["Result"]=="H") & (h2h["HomeTeam"]==away_team)).sum() + ((h2h["Result"]=="A") & (h2h["AwayTeam"]==away_team)).sum())
        draws = int((h2h["Result"]=="D").sum())

        st.markdown(f"**Matches:** {total_matches} • **Total goals:** {total_goals} • **Avg goals/match:** {avg_goals:.2f}")
        st.markdown(f"**{home_team} wins:** {home_wins} • **{away_team} wins:** {away_wins} • **Draws:** {draws}")

        # Narrative generator (simple deterministic text)
        narr = (f"Over the recorded period, {home_team} and {away_team} met {total_matches} times. "
                 f"They averaged {avg_goals:.2f} goals per match. "
                 f"{home_team} has {home_wins} wins, {away_team} has {away_wins}, and there were {draws} draws. ")
        # add sentiment about momentum (based on last 5)
        last5 = h2h.head(5)
        if not last5.empty:
            lw = last5["Result"].map({"H":home_team, "A":away_team, "D":"Draw"})
            recent = lw.value_counts().idxmax()
            narr += f"In the recent 5 encounters, {recent} has shown stronger form."
        st.info(narr)

        # TOP 3 CLASSIC MATCHES (as earlier)
        st.markdown("#### 🔥 Top Classic Matches")
        if h2h.empty:
            st.info("No head-to-head matches in history.")
        else:
            df = h2h.copy()
            df["TotalGoals"] = df["HomeGoals"] + df["AwayGoals"]
            df["GoalDiffAbs"] = (df["HomeGoals"] - df["AwayGoals"]).abs()
            df["Excitement"] = df["TotalGoals"] + (5 - df["GoalDiffAbs"])
            df["Season"] = df["Date"].apply(lambda d: f"{d.year-1}/{str(d.year)[2:]}" if d.month<7 else f"{d.year}/{str(d.year+1)[2:]}")
            df["Comeback"] = df.apply(lambda r: (r["HomeTeam"] if r["Result"]=="H" else r["AwayTeam"]) + " comeback" 
                                      if ((r["Result"]=="H" and r["HomeGoals"]<r["AwayGoals"]) or (r["Result"]=="A" and r["AwayGoals"]<r["HomeGoals"])) else None, axis=1)
            classics = df.sort_values("Excitement", ascending=False).head(3)
            for _, row in classics.iterrows():
                date = row["Date"].date()
                score = f"{int(row['HomeGoals'])}-{int(row['AwayGoals'])}"
                tag = ("🔥 Comeback!" if row["Comeback"] else "🎆 Goal Fest!" if row["TotalGoals"]>=6 else "🤝 Close contest")
                yt_q = f"{row['HomeTeam']} {row['AwayTeam']} {row['Season']} highlights"
                yt_link = "https://www.youtube.com/results?search_query=" + yt_q.replace(" ","+")
                st.markdown(f"""
                    <div style="background:#0f1720;color:white;padding:12px;border-radius:10px;margin-bottom:8px">
                      <b>{row['HomeTeam']} {score} {row['AwayTeam']}</b> — {date} • Season: {row['Season']}<br>
                      <i>{tag}</i> • Total goals: {int(row['TotalGoals'])} • Goal diff: {int(row['GoalDiffAbs'])}<br>
                      <a href="{yt_link}" target="_blank"><button style="padding:6px;border-radius:6px;background:#e74c3c;color:white;border:none">🎥 Watch Highlights</button></a>
                    </div>
                """, unsafe_allow_html=True)

        # Animated-ish / interactive goal timeline across matches
        st.markdown("#### ⏱️ Goal Timeline (H2H) — each dot = a goal at minute X in one match")
        if not h2h.empty:
            # build synthetic per-goal minutes for each match deterministically
            events = []
            for idx, r in df.iterrows():
                match_id = f"{r.Date.date()}_{r.HomeTeam}_{r.AwayTeam}"
                seed = int(hashlib.sha1(match_id.encode()).hexdigest(), 16) % (2**32)
                rng = np.random.default_rng(seed)
                home_goals = int(r.HomeGoals)
                away_goals = int(r.AwayGoals)
                # spread minutes 1-90 randomly but reproducibly
                if home_goals > 0:
                    mins = rng.choice(np.arange(1,91), size=home_goals, replace=False)
                    for m in mins:
                        events.append({"match_date": r.Date.date(), "team": r.HomeTeam, "minute": int(m)})
                if away_goals > 0:
                    mins = rng.choice(np.arange(1,91), size=away_goals, replace=False)
                    for m in mins:
                        events.append({"match_date": r.Date.date(), "team": r.AwayTeam, "minute": int(m)})
            ev_df = pd.DataFrame(events)
            if ev_df.empty:
                st.info("No goal events to plot.")
            else:
                # Plotly scatter: x=minute, y=match date (as categorical); color by team
                ev_df["match_label"] = ev_df["match_date"].astype(str)
                fig = go.Figure()
                for t in [home_team, away_team]:
                    sub = ev_df[ev_df["team"]==t]
                    fig.add_trace(go.Scatter(
                        x=sub["minute"], y=sub["match_label"], mode="markers",
                        name=t, marker=dict(size=10, color=safe_color(t))
                    ))
                fig.update_layout(height=400, xaxis_title="Minute", yaxis_title="Match Date",
                                  template="plotly_dark")
                st.plotly_chart(fig, width="stretch")

            # Simulated top scorers (deterministic, synthetic)
            st.markdown("#### 🏅 Top Scorers (SIMULATED — no player data available)")
            # create deterministic fake scorers per team: pick names from team + hash
            def simulate_scorers(team, goals_total):
                rng = np.random.default_rng(int(hashlib.md5(team.encode()).hexdigest(), 16) % (2**32))
                pool = [f"{team.split()[0]} Player {i}" for i in range(1, 9)]

                # Generate random positive values and normalize
                raw = rng.random(8)
                pvals = raw / raw.sum()  # now guaranteed to sum to 1

                counts = rng.multinomial(goals_total, pvals)
                dfp = (
                    pd.DataFrame({"Player": pool, "Goals": counts})
                    .sort_values("Goals", ascending=False)
                    .head(5)
                )
                return dfp

            total_home_goals = int(df[df["HomeTeam"]==home_team]["HomeGoals"].sum() + df[df["AwayTeam"]==home_team]["AwayGoals"].sum())
            total_away_goals = int(df[df["HomeTeam"]==away_team]["HomeGoals"].sum() + df[df["AwayTeam"]==away_team]["AwayGoals"].sum())
            home_top = simulate_scorers(home_team, total_home_goals if total_home_goals>0 else 1)
            away_top = simulate_scorers(away_team, total_away_goals if total_away_goals>0 else 1)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{home_team} — Top (simulated)**")
                st.table(home_top)
            with c2:
                st.markdown(f"**{away_team} — Top (simulated)**")
                st.table(away_top)

# ---------- TAB 2: Batch Predictions ----------
with tabs[2]:
    st.subheader("📂 Upload upcoming_fixtures.csv for Bulk Predictions")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        upcoming = pd.read_csv(up)
        stats = build_team_stats(historic_df)
        merged = upcoming.merge(stats.add_prefix("Home_"), left_on="HomeTeam", right_on="Home_Team", how="left")\
                         .merge(stats.add_prefix("Away_"), left_on="AwayTeam", right_on="Away_Team", how="left")\
                         .fillna(stats.mean(numeric_only=True))
        merged["GoalDiff"] = merged["Home_Home_AvgGoalsFor"] - merged["Away_Away_AvgGoalsFor"]
        merged["HomeAdvantage"] = 1
        X = merged[["Home_Home_AvgGoalsFor", "Home_Home_AvgGoalsAgainst",
                    "Away_Away_AvgGoalsFor", "Away_Away_AvgGoalsAgainst",
                    "GoalDiff", "HomeAdvantage"]]
        preds = model.predict(X)
        probs = model.predict_proba(X)
        lbl = {0:"Home Win",1:"Draw",2:"Away Win"}
        merged["Predicted"] = [lbl[i] for i in preds]
        merged["Home Win"] = (probs[:,0]*100).round(2)
        merged["Draw"] = (probs[:,1]*100).round(2)
        merged["Away Win"] = (probs[:,2]*100).round(2)
        st.dataframe(merged[["Date","HomeTeam","AwayTeam","Predicted","Home Win","Draw","Away Win"]], width="stretch")
        merged.to_csv("data/predictions.csv", index=False)
        st.success("💾 Saved → data/predictions.csv")

st.caption("Phase 5 — Match Story Mode. Player-level data unavailable")
# ---------------------------