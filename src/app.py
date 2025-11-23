# src/app.py
"""
Premier Predictor — Polymarket-style Premium Dashboard (Theme A: Purple/Cyan Neon)
Full app: blending model + market, scoreline predictor, heatmap inspector, logos, updater.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import poisson
import subprocess
import hashlib
import io
import re
import math
from typing import Optional

# Optional Plotly click events package - non-fatal if missing
_HAS_PLOTLY_EVENTS = True
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    _HAS_PLOTLY_EVENTS = False

# ---------------------------
# Config / paths
# ---------------------------
HISTORIC_PATHS = ["data/historic_data.csv", "/mnt/data/historic_data.csv"]
MODEL_PATH = "models/match_predictor.pkl"
TEAM_LOGOS_CSV = "data/team_logos.csv"
UPDATE_SCRIPT = "scripts/update_dataset.py"

# ---------------------------
# Built-in color map (normalized keys)
# Theme A colors (neon purple -> cyan accents)
STATIC_COLOR_MAP = {
    "liverpool":      "#C8102E",
    "west ham":       "#7A263A",
    "bournemouth":    "#DA291C",
    "burnley":        "#6C1D45",
    "crystal palace": "#1B458F",
    "watford":        "#E2B007",
    "tottenham":      "#132257",
    "leicester":      "#003090",
    "newcastle":      "#241F20",
    "man united":     "#DA291C",
    "arsenal":        "#EF0107",
    "aston villa":    "#95BFE5",
    "brighton":       "#0057B8",
    "everton":        "#003399",
    "norwich":        "#FFF200",
    "southampton":    "#D71920",
    "man city":       "#6CABDD",
    "sheffield united":"#D71920",
    "chelsea":        "#034694",
    "wolves":         "#FDB913",
}

# ---------------------------
# Polymarket-style CSS (Theme A)
# ---------------------------
st.set_page_config(page_title="Premier Predictor — Neon", layout="wide", initial_sidebar_state="expanded")

THEME_CSS = """
<style>
:root{
  --bg: #05060a;
  --panel: linear-gradient(180deg, rgba(10,10,25,0.95), rgba(5,5,12,0.95));
  --muted: #9aa6b2;
  --neon1: #6E4BFF; /* purple */
  --neon2: #00E5FF; /* cyan */
}
html, body, [data-testid="stAppViewContainer"] > .main {
  background: radial-gradient(circle at 10% 10%, rgba(110,75,255,0.06), transparent 10%),
              radial-gradient(circle at 90% 90%, rgba(0,229,255,0.04), transparent 10%),
              var(--bg);
  color: #E9F1FF;
}
.panel {
  background: var(--panel);
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 12px;
  border: 1px solid rgba(110,75,255,0.07);
  box-shadow: 0 8px 30px rgba(0,0,0,0.6);
}
.muted { color: var(--muted); font-size:13px; }
.kpi { font-weight:700; font-size:20px; color:#F1F7FF; }
.team-name { font-weight:700; font-size:15px; color:#EAF2FF; }
.small { font-size:12px; color:var(--muted); }
.team-accent {
  padding:6px 10px;
  border-radius:8px;
  display:inline-block;
  color: #fff;
  font-weight:700;
}
.gradient-bar {
  height:12px;
  border-radius:8px;
  margin-top:10px;
}
.card-title {
  font-weight:700;
  font-size:16px;
  color:#EAF2FF;
}
.control-desc {
  font-size:12px;
  color: #bcd0ff;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers: normalization, loading
# ---------------------------
def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = re.sub(r"[^a-z0-9 ]", "", name.lower())
    s = s.replace("fc", "").replace("afc", "").replace("the ", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def find_historical_path() -> Optional[str]:
    for p in HISTORIC_PATHS:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(ttl=3600)
def load_historic(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = find_historical_path()
    if path is None:
        st.warning("No historic_data.csv found. Create data/historic_data.csv with Date,HomeTeam,AwayTeam,HomeGoals,AwayGoals,Result")
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result"])
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if df["Date"].isna().mean() > 0.25:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    for c in ["HomeGoals", "AwayGoals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_resource
def load_model(path: str = MODEL_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                st.warning("Model exists but couldn't be loaded (version mismatch). Falling back to uniform model.")
                return None
    st.info("No model found. Predictions will be uniform (1/3 each).")
    return None

# ---------------------------
# Team logos + wiki fallback
# ---------------------------
@st.cache_data(ttl=86400)
def load_team_logo_map():
    if os.path.exists(TEAM_LOGOS_CSV):
        try:
            tdf = pd.read_csv(TEAM_LOGOS_CSV)
            if "Team" in tdf.columns and "LogoURL" in tdf.columns:
                return {row["Team"]: row["LogoURL"] for _, row in tdf.iterrows()}
        except Exception:
            pass
    return {}

def fetch_wikipedia_thumbnail(team_name: str) -> Optional[str]:
    try:
        q = team_name + " football club"
        params = {"action": "query", "prop": "pageimages", "format": "json", "pithumbsize": 300, "titles": q, "redirects": 1}
        r = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=6)
        r.raise_for_status()
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                return thumb
    except Exception:
        return None
    return None

def get_logo_url_for_team(team: str, logo_map: dict) -> str:
    if team in logo_map:
        return logo_map[team]
    n = normalize_name(team)
    for k, v in logo_map.items():
        if normalize_name(k) == n:
            return v
    wiki = fetch_wikipedia_thumbnail(team)
    if wiki:
        return wiki
    label = team.split()[0] if team else "Team"
    color = STATIC_COLOR_MAP.get(n, "#6C7A89").lstrip("#")
    return f"https://via.placeholder.com/96/{color}/FFFFFF?text={requests.utils.requote_uri(label)}"

# ---------------------------
# Team stats builder
# ---------------------------
def build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Team","Home_AvgGoalsFor","Home_AvgGoalsAgainst","Away_AvgGoalsFor","Away_AvgGoalsAgainst"])
    home = df.groupby("HomeTeam").agg({"HomeGoals":"mean","AwayGoals":"mean"}).reset_index()
    home.columns = ["Team","Home_AvgGoalsFor","Home_AvgGoalsAgainst"]
    away = df.groupby("AwayTeam").agg({"AwayGoals":"mean","HomeGoals":"mean"}).reset_index()
    away.columns = ["Team","Away_AvgGoalsFor","Away_AvgGoalsAgainst"]
    stats = pd.merge(home, away, on="Team", how="outer").fillna(0)
    return stats

def safe_mean(series, fallback=0.0) -> float:
    try:
        return float(pd.to_numeric(series, errors="coerce").mean())
    except Exception:
        return fallback

# ---------------------------
# Scoreline engine: recency + H2H + Poisson
# ---------------------------
def compute_strengths(df: pd.DataFrame, team: str, opponent: str,
                      recent_n: int = 8, h2h_n: int = 4, decay: float = 0.7,
                      recent_weight: float = 0.6, h2h_weight: float = 0.4):
    team_matches = df[(df["HomeTeam"]==team)|(df["AwayTeam"]==team)].sort_values("Date", ascending=False).head(recent_n)
    if team_matches.empty:
        # fallback to league averages
        home_mean = df["HomeGoals"].mean() if "HomeGoals" in df and not df["HomeGoals"].isna().all() else 1.2
        away_mean = df["AwayGoals"].mean() if "AwayGoals" in df and not df["AwayGoals"].isna().all() else 1.2
        return max(home_mean, 0.05), max(away_mean, 0.05)
    L = len(team_matches)
    weights = np.array([decay**i for i in range(L)])
    weights = weights / weights.sum()
    gf = []; ga = []
    for i,row in enumerate(team_matches.itertuples()):
        w = weights[i]
        if row.HomeTeam == team:
            gf.append(w*row.HomeGoals); ga.append(w*row.AwayGoals)
        else:
            gf.append(w*row.AwayGoals); ga.append(w*row.HomeGoals)
    avg_for_recent = sum(gf)
    avg_against_recent = sum(ga)
    # h2h
    h2h = df[((df["HomeTeam"]==team)&(df["AwayTeam"]==opponent))|((df["HomeTeam"]==opponent)&(df["AwayTeam"]==team))].sort_values("Date", ascending=False).head(h2h_n)
    if not h2h.empty:
        gf_h2h = []; ga_h2h = []
        for row in h2h.itertuples():
            if row.HomeTeam == team:
                gf_h2h.append(row.HomeGoals); ga_h2h.append(row.AwayGoals)
            else:
                gf_h2h.append(row.AwayGoals); ga_h2h.append(row.HomeGoals)
        avg_for_h2h = float(np.mean(gf_h2h))
        avg_against_h2h = float(np.mean(ga_h2h))
    else:
        avg_for_h2h = avg_for_recent; avg_against_h2h = avg_against_recent
    attack_strength = recent_weight*avg_for_recent + h2h_weight*avg_for_h2h
    defense_weakness = recent_weight*avg_against_recent + h2h_weight*avg_against_h2h
    attack_strength = max(attack_strength, 0.01); defense_weakness = max(defense_weakness, 0.01)
    return attack_strength, defense_weakness

def predict_score_distribution(df: pd.DataFrame, home: str, away: str,
                               recent_n=8, h2h_n=4, decay=0.7,
                               home_advantage_factor=1.08, max_goals=5,
                               blended_probs=None, blend_adjustment=0.12):
    home_attack, home_def = compute_strengths(df, home, away, recent_n=recent_n, h2h_n=h2h_n, decay=decay)
    away_attack, away_def = compute_strengths(df, away, home, recent_n=recent_n, h2h_n=h2h_n, decay=decay)
    lam_home = home_attack * away_def * home_advantage_factor
    lam_away = away_attack * home_def
    # blend adjustment nudges exp goals toward favored side
    if blended_probs is not None:
        fav = blended_probs[0] - blended_probs[2]
        lam_home *= (1 + blend_adjustment * fav)
        lam_away *= (1 - blend_adjustment * fav)
        lam_home = max(lam_home, 0.01); lam_away = max(lam_away, 0.01)
    rows = []
    for hg in range(0, max_goals+1):
        for ag in range(0, max_goals+1):
            p = poisson.pmf(hg, lam_home) * poisson.pmf(ag, lam_away)
            rows.append({"HomeGoals": hg, "AwayGoals": ag, "Probability": p})
    prob_df = pd.DataFrame(rows)
    s = prob_df["Probability"].sum()
    if s > 0:
        prob_df["Probability"] = prob_df["Probability"] / s
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
    exp_home = float((prob_df["HomeGoals"] * prob_df["Probability"]).sum())
    exp_away = float((prob_df["AwayGoals"] * prob_df["Probability"]).sum())
    return prob_df, (exp_home, exp_away)

def get_top_scorelines(prob_df: pd.DataFrame, top_n: int = 6):
    return prob_df.head(top_n)

# ---------------------------
# Load data, model, logos
# ---------------------------
historic_df = load_historic()
model = load_model()
team_stats = build_team_stats(historic_df)
teams = sorted(team_stats["Team"].dropna().unique()) if not team_stats.empty else []
logo_map = load_team_logo_map()

# ---------------------------
# Sidebar: controls, logos upload/paste, updater
# ---------------------------
st.sidebar.markdown("## Controls & Data")
blend_default = st.sidebar.slider("Model weight % (higher = trust model more)", 0, 100, 70)
recent_n = st.sidebar.slider("Recent matches per team (window used in scoreline)", 3, 12, 8)
h2h_n = st.sidebar.slider("H2H matches to include", 1, 6, 4)
decay = st.sidebar.slider("Recency decay (higher = recent matches weigh more)", 0.5, 0.95, 0.7)
max_goals = st.sidebar.slider("Max goals per side (scoreline granularity)", 3, 8, 5)
st.sidebar.markdown("<div class='control-desc'>Tip: increase 'Recent matches' to rely more on recent form; increase 'Model weight' to trust your trained model more than market odds.</div>", unsafe_allow_html=True)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Team logos (optional)")
st.sidebar.markdown("Place a CSV at data/team_logos.csv with columns: Team,LogoURL OR upload here OR paste JSON mapping.")
uploaded = st.sidebar.file_uploader("Upload team logos CSV (optional)", type=["csv"])
json_input = st.sidebar.text_area("Or paste JSON mapping (optional)", height=120, help='{"Arsenal":"https://...png","Chelsea":"https://...png"}')

if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        if "Team" in df_upload.columns and "LogoURL" in df_upload.columns:
            for _, r in df_upload.iterrows():
                logo_map[str(r["Team"])] = str(r["LogoURL"])
            st.sidebar.success("Uploaded logo CSV loaded.")
        else:
            st.sidebar.error("CSV must have columns: Team,LogoURL")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

if json_input:
    try:
        j = json.loads(json_input)
        if isinstance(j, dict):
            for k, v in j.items():
                logo_map[str(k)] = str(v)
            st.sidebar.success("JSON mapping loaded.")
        else:
            st.sidebar.error("JSON must be an object mapping 'Team'->'LogoURL'")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

# Dataset updater
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset updater")
use_api = st.sidebar.checkbox("Use API-Football (optional)", value=False)
api_key = st.sidebar.text_input("API-Football key (if using)", value="", type="password") if use_api else ""
season_code = st.sidebar.text_input("football-data season code (e.g. 2526)", value="")
if st.sidebar.button("Update dataset now"):
    st.sidebar.info("Updating dataset...")
    try:
        if os.path.exists(UPDATE_SCRIPT):
            cmd = ["python", UPDATE_SCRIPT]
            if use_api and api_key:
                cmd += ["--api_key", api_key]
            if season_code:
                cmd += ["--season", season_code]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=180)
            st.sidebar.success("Updater script finished.")
            st.sidebar.text(out)
        else:
            if not season_code:
                st.sidebar.error("No update script installed and no season_code provided for fallback CSV.")
            else:
                url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                tmp = pd.read_csv(io.StringIO(r.text))
                if "Date" in tmp.columns and "FTHG" in tmp.columns and "FTAG" in tmp.columns:
                    tmp2 = tmp.rename(columns={"FTHG":"HomeGoals","FTAG":"AwayGoals","FTR":"Result"})[["Date","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result"]]
                    tmp2["Date"] = pd.to_datetime(tmp2["Date"], dayfirst=True, errors="coerce")
                    local_path = find_historical_path() or "data/historic_data.csv"
                    if os.path.exists(local_path):
                        existing = pd.read_csv(local_path)
                    else:
                        existing = pd.DataFrame(columns=["Date","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result"])
                    existing["Date_parsed"] = pd.to_datetime(existing["Date"], errors="coerce").dt.date
                    tmp2["Date_parsed"] = tmp2["Date"].dt.date
                    existing_keys = set(existing.apply(lambda r: (r["Date_parsed"], r["HomeTeam"], r["AwayTeam"]), axis=1).tolist())
                    new_rows = []
                    for _, r in tmp2.iterrows():
                        key = (r["Date_parsed"], r["HomeTeam"], r["AwayTeam"])
                        if key not in existing_keys:
                            new_rows.append(r[["Date","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result"]])
                    if new_rows:
                        appended = pd.concat([existing.drop(columns=["Date_parsed"]), pd.DataFrame(new_rows)], ignore_index=True, sort=False)
                        appended.to_csv(local_path, index=False)
                        st.sidebar.success(f"Appended {len(new_rows)} new matches to {local_path}")
                    else:
                        st.sidebar.info("No new matches found in CSV.")
                else:
                    st.sidebar.error("CSV format unexpected (football-data).")
        # reload
        historic_df = load_historic()
        team_stats = build_team_stats(historic_df)
        teams = sorted(team_stats["Team"].dropna().unique()) if not team_stats.empty else []
        st.experimental_rerun()
    except subprocess.CalledProcessError as e:
        st.sidebar.error(f"Updater script failed:\n{e.output}")
    except Exception as e:
        st.sidebar.error(f"Update failed: {e}")

# ---------------------------
# Main UI: header with logos & dynamic coloring
# ---------------------------
st.title("⚽ Premier Predictor — Neon (Polymarket style)")

left_col, right_col = st.columns([2,1])
with left_col:
    home_team = st.selectbox("Home team", [""] + list(teams), index=0)
with right_col:
    away_team = st.selectbox("Away team", [""] + list(teams), index=0)

if not home_team or not away_team:
    st.info("Pick both Home and Away teams to see predictions.")
    st.stop()
if home_team == away_team:
    st.error("Home and Away team must be different.")
    st.stop()

home_logo = get_logo_url_for_team(home_team, logo_map)
away_logo = get_logo_url_for_team(away_team, logo_map)
home_color = STATIC_COLOR_MAP.get(normalize_name(home_team), "#6C7A89")
away_color = STATIC_COLOR_MAP.get(normalize_name(away_team), "#6C7A89")

# gradient CSS for header accents
grad_css = f"""
<style>
.header-gradient {{
  background: linear-gradient(90deg, {home_color}, {away_color});
  -webkit-background-clip: text;
  color: transparent;
}}
.panel-left-accent {{ border-left:6px solid {home_color}; }}
.panel-right-accent {{ border-left:6px solid {away_color}; }}
</style>
"""
st.markdown(grad_css, unsafe_allow_html=True)

# header with logos
c1, c2, c3 = st.columns([1,3,1])
with c1:
    try:
        st.image(home_logo, width=88)
    except Exception:
        st.markdown(f"**{home_team}**")
with c2:
    st.markdown(f"<div class='header-gradient'><h2 style='margin:0'>{home_team}  vs  {away_team}</h2></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small muted'>Model + Market blended predictions, scoreline forecast & story mode</div>", unsafe_allow_html=True)
with c3:
    try:
        st.image(away_logo, width=88)
    except Exception:
        st.markdown(f"**{away_team}**")

# ---------------------------
# Build X and model probs
# ---------------------------
if not team_stats.empty and home_team in team_stats["Team"].values and away_team in team_stats["Team"].values:
    ht = team_stats[team_stats["Team"]==home_team].iloc[0]
    at = team_stats[team_stats["Team"]==away_team].iloc[0]
else:
    ht = at = pd.Series({"Home_AvgGoalsFor":1.2,"Home_AvgGoalsAgainst":1.2,"Away_AvgGoalsFor":1.2,"Away_AvgGoalsAgainst":1.2})

X_row = {
    "Home_Home_AvgGoalsFor": ht.get("Home_AvgGoalsFor", 1.2),
    "Home_Home_AvgGoalsAgainst": ht.get("Home_AvgGoalsAgainst", 1.2),
    "Away_Away_AvgGoalsFor": at.get("Away_AvgGoalsFor", 1.2),
    "Away_Away_AvgGoalsAgainst": at.get("Away_AvgGoalsAgainst", 1.2),
    "GoalDiff": ht.get("Home_AvgGoalsFor", 1.2) - at.get("Away_AvgGoalsFor", 1.2),
    "HomeAdvantage": 1
}
X = pd.DataFrame([X_row])

if model is not None:
    try:
        probs = model.predict_proba(X)[0]
        model_arr = np.array([probs[0], probs[1], probs[2]])
    except Exception:
        model_arr = np.array([1/3,1/3,1/3])
else:
    model_arr = np.array([1/3,1/3,1/3])

# ---------------------------
# Market odds mirror + scrape
# ---------------------------
def _normalize(text: Optional[str]) -> str:
    return "".join(c.lower() for c in (text or "") if c.isalnum() or c.isspace()).strip()

def fetch_from_mirror(home: str, away: str):
    try:
        r = requests.get("https://api.polymarket.xyz/v4/markets", timeout=6)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return None
    try:
        for m in j:
            q = _normalize(m.get("question") or m.get("title") or "")
            if _normalize(home) in q and _normalize(away) in q:
                out = {}
                outcomes = m.get("outcomes") or m.get("prices") or []
                for o in outcomes:
                    name = o.get("name") or o.get("title") or str(o.get("id"))
                    price = o.get("price") or o.get("probability") or 0
                    try:
                        price = float(price)
                    except Exception:
                        price = 0.0
                    out[name] = price*100 if price<=1.001 else price
                s = sum(out.values())
                if s > 0:
                    return {k: round(v/s*100,2) for k,v in out.items()}
                return out
    except Exception:
        return None
    return None

def scrape_polymarket_sports(home: str, away: str):
    try:
        idx_url = f"https://polymarket.com/sports/premier-league-2025/games"
        r = requests.get(idx_url, timeout=6)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        return None
    try:
        candidates = []
        for a in soup.select("a"):
            href = a.get("href") or ""
            txt = (a.get_text() or "")
            if _normalize(home) in _normalize(txt) and _normalize(away) in _normalize(txt):
                candidates.append(href)
        for href in candidates:
            page_url = href if href.startswith("http") else f"https://polymarket.com{href}"
            try:
                rr = requests.get(page_url, timeout=6)
                rr.raise_for_status()
                text = rr.text.lower()
                import re
                out = {}
                m_home = re.search(rf"{_normalize(home)}[^0-9]{{0,30}}(\\d+\\.?\\d*%?)", text)
                m_away = re.search(rf"{_normalize(away)}[^0-9]{{0,30}}(\\d+\\.?\\d*%?)", text)
                m_draw = re.search(r"draw[^0-9]{0,30}(\\d+\\.?\\d*%?)", text)
                def parse_val(m):
                    if not m:
                        return None
                    v = m.group(1)
                    if "%" in v:
                        return float(v.replace("%",""))
                    val = float(v)
                    return val*100 if val<=1.001 else val
                hv = parse_val(m_home); av = parse_val(m_away); dv = parse_val(m_draw)
                if hv is not None: out["Home Win"] = hv
                if dv is not None: out["Draw"] = dv
                if av is not None: out["Away Win"] = av
                if out:
                    s = sum(out.values())
                    if s>0:
                        return {k: round(v/s*100,2) for k,v in out.items()}
                    return out
            except Exception:
                continue
    except Exception:
        return None
    return None

def get_polymarket_odds_combined(home: str, away: str):
    mirror = fetch_from_mirror(home, away)
    if mirror:
        return mirror
    return scrape_polymarket_sports(home, away)

with st.spinner("Fetching market odds..."):
    pm = get_polymarket_odds_combined(home_team, away_team)

market_arr = np.array([1/3,1/3,1/3])
if pm:
    keys = list(pm.keys())
    home_val = pm.get("Home Win") or pm.get(home_team) or pm.get(keys[0], 33.33)
    away_val = pm.get("Away Win") or pm.get(away_team) or pm.get(keys[-1], 33.33)
    draw_val = pm.get("Draw")
    if draw_val is None:
        try:
            draw_val = max(0.0, 100.0 - (float(home_val) + float(away_val)))
        except Exception:
            draw_val = 33.33
    try:
        market_arr = np.array([float(home_val)/100.0, float(draw_val)/100.0, float(away_val)/100.0])
        market_arr = market_arr / market_arr.sum()
    except Exception:
        market_arr = np.array([1/3,1/3,1/3])

# blending
w = blend_default / 100.0
blended = w * model_arr + (1 - w) * market_arr
blended = blended / blended.sum()

# ---------------------------
# Header cards + big probability bars (Neon)
# ---------------------------
colA, colB = st.columns([3,1])
with colA:
    st.markdown(f"<div class='panel panel-left-accent'>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'><div><span class='team-accent' style='background:{home_color}'>{home_team}</span></div><div style='text-align:right'><span class='small muted'>Blended pick</span><br><strong style='font-size:18px'>{['Home','Draw','Away'][int(np.argmax(blended))]}</strong></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<div class='panel panel-right-accent'>", unsafe_allow_html=True)
    st.metric(label="Top model confidence", value=f"{max(model_arr)*100:.1f}%")
    grad = f"linear-gradient(90deg, {home_color}, {away_color})"
    st.markdown(f"<div class='gradient-bar' style='background:{grad}'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# horizontal big bars
fig_probs = go.Figure()
labels = ["Home Win", "Draw", "Away Win"]
values = [blended[0]*100, blended[1]*100, blended[2]*100]
colors = [home_color, "#9aa6b2", away_color]
fig_probs.add_trace(go.Bar(x=values, y=labels, orientation='h', marker=dict(color=colors, line=dict(width=0)), text=[f"{v:.1f}%" for v in values], textposition='inside'))
fig_probs.update_layout(height=220, margin=dict(l=80,r=20,t=10,b=10), plot_bgcolor="#05060a", paper_bgcolor="#05060a", xaxis=dict(range=[0,100], title="Probability (%)"), font=dict(color="#E9F1FF"))
st.plotly_chart(fig_probs, use_container_width=True)

# ---------------------------
# Scoreline forecast, heatmap, interactions
# ---------------------------
st.markdown("---")
st.subheader("Scoreline Forecast (Poisson + Recent form + H2H)")

prob_df, (exp_home, exp_away) = predict_score_distribution(
    historic_df, home_team, away_team,
    recent_n=recent_n, h2h_n=h2h_n, decay=decay,
    blended_probs=blended, blend_adjustment=0.12, max_goals=max_goals
)
top_lines = get_top_scorelines(prob_df, top_n=6)

st.markdown(f"**Expected goals:** <span style='color:{home_color}'>{home_team}</span>: **{exp_home:.2f}**, <span style='color:{away_color}'>{away_team}</span>: **{exp_away:.2f}**", unsafe_allow_html=True)

top_table = top_lines.copy().assign(**{"Prob (%)": lambda d: (d["Probability"]*100).round(2)}).drop(columns=["Probability"]).reset_index(drop=True)
st.table(top_table)

# build heat matrix
heat = prob_df.pivot_table(index="HomeGoals", columns="AwayGoals", values="Probability", fill_value=0)
annotations = []
for _, r in top_lines.head(6).iterrows():
    annotations.append(dict(x=str(int(r.AwayGoals)), y=str(int(r.HomeGoals)), text=f"{r.Probability*100:.1f}%", showarrow=False, font=dict(color='white', size=12)))

heatmap = go.Figure(data=go.Heatmap(
    z=heat.values,
    x=[str(c) for c in heat.columns.astype(int)],
    y=[str(r) for r in heat.index.astype(int)],
    colorscale=[[0, home_color], [0.5, "#6E6E6E"], [1, away_color]],
    colorbar=dict(title="Probability")
))
heatmap.update_layout(
    title="Scoreline probability heatmap",
    xaxis_title=f"Away goals — {away_team}",
    yaxis_title=f"Home goals — {home_team}",
    annotations=annotations,
    plot_bgcolor="#05060a",
    paper_bgcolor="#05060a",
    font=dict(color="#E9F1FF"),
    height=520,
    margin=dict(l=60, r=20, t=60, b=60)
)
heatmap.update_xaxes(tickmode='array', tickvals=[str(c) for c in heat.columns.astype(int)])
heatmap.update_yaxes(tickmode='array', tickvals=[str(r) for r in heat.index.astype(int)])

left, right = st.columns([3,1])
with left:
    st.plotly_chart(heatmap, use_container_width=True)
    # capture click event if package present
    clicked = []
    if _HAS_PLOTLY_EVENTS:
        try:
            clicked = plotly_events(heatmap, click_event=True, hover_event=False)
        except Exception:
            clicked = []
else_click_info = None  # fallback selection chosen manually below

with right:
    inspector_title = st.empty()
    inspector_box = st.empty()
    inspector_title.markdown(f"<div class='panel'><strong style='color:{home_color}'>Scoreline Inspector</strong><div class='muted small'>Click a heatmap cell (or choose from list)</div></div>", unsafe_allow_html=True)
    inspector_box.markdown("<div class='panel'>No selection yet.</div>", unsafe_allow_html=True)
    # Fallback selector if click didn't work or file lacks plotly-events
    # Build selectable list of top lines for explicit selection
    select_options = [f"{int(r.HomeGoals)}-{int(r.AwayGoals)} ({r.Probability*100:.2f}%)" for _, r in top_lines.iterrows()]
    select_options = select_options or ["No lines"]
    chosen = st.selectbox("Or pick a top scoreline to inspect", select_options, index=0)
    if _HAS_PLOTLY_EVENTS and clicked:
        evt = clicked[0]
        if isinstance(evt, dict) and ("x" in evt or "points" in evt):
            # handle shape of event returned
            if "points" in evt and isinstance(evt["points"], list) and len(evt["points"])>0:
                p = evt["points"][0]
                x_val = p.get("x") or p.get("x_val") or p.get("x0")
                y_val = p.get("y") or p.get("y_val") or p.get("y0")
            else:
                x_val = evt.get("x")
                y_val = evt.get("y")
            try:
                ag = int(float(x_val))
                hg = int(float(y_val))
                else_click_info = f"{hg}-{ag}"
            except Exception:
                else_click_info = None
    # choose final selection (click wins over manual select)
    final_sel = None
    if else_click_info:
        final_sel = else_click_info
    else:
        # parse chosen from dropdown
        try:
            final_sel = chosen.split()[0]
        except Exception:
            final_sel = None

    if final_sel:
        try:
            hg, ag = [int(x) for x in final_sel.split("-")]
            row = prob_df[(prob_df["HomeGoals"]==hg) & (prob_df["AwayGoals"]==ag)]
            prob_val = float(row["Probability"].iloc[0]) if not row.empty else 0.0
            hist_mask = (historic_df["HomeGoals"]==hg) & (historic_df["AwayGoals"]==ag)
            freq = historic_df[hist_mask].shape[0]
            last_date = None
            if freq>0:
                last_date_raw = historic_df[hist_mask].sort_values("Date", ascending=False)["Date"].iloc[0]
                last_date = last_date_raw.date().isoformat() if not pd.isna(last_date_raw) else "Unknown"
            cat = "Home Win" if hg>ag else ("Draw" if hg==ag else "Away Win")
            inspector_title.markdown(f"<div class='panel' style='border-left:6px solid {home_color};'><strong>Scoreline Inspector — {hg}-{ag}</strong></div>", unsafe_allow_html=True)
            info_html = f"""<div class='panel'>
                <div><strong>Probability:</strong> {prob_val*100:.2f}%</div>
                <div><strong>Category:</strong> {cat}</div>
                <div><strong>Historic frequency:</strong> {freq} occurrences</div>
                <div><strong>Last seen:</strong> {last_date if last_date else 'Never in dataset'}</div>
            </div>"""
            inspector_box.markdown(info_html, unsafe_allow_html=True)
            if freq>0:
                examples = historic_df[hist_mask].sort_values("Date", ascending=False).head(5)[["Date","HomeTeam","HomeGoals","AwayGoals","AwayTeam"]]
                examples = examples.assign(Score = examples["HomeGoals"].astype(int).astype(str)+"-"+examples["AwayGoals"].astype(int).astype(str))
                inspector_box.table(examples.assign(Date=lambda d: d["Date"].dt.date.astype(str)).rename(columns={"HomeTeam":"Home","AwayTeam":"Away","Score":"Score"}).reset_index(drop=True))
        except Exception as e:
            inspector_box.markdown("<div class='panel'>Error reading selection.</div>", unsafe_allow_html=True)
            if show_debug:
                st.write(e)

# ---------------------------
# Story mode & H2H insights
# ---------------------------
st.markdown("---")
st.header("Story Mode — Rivalry Narrative")
h2h = historic_df[((historic_df["HomeTeam"]==home_team)&(historic_df["AwayTeam"]==away_team))|((historic_df["HomeTeam"]==away_team)&(historic_df["AwayTeam"]==home_team))].sort_values("Date", ascending=False)
total_matches = len(h2h)
total_goals = int((h2h["HomeGoals"] + h2h["AwayGoals"]).sum()) if total_matches>0 else 0
avg_goals = (total_goals / total_matches) if total_matches>0 else 0.0
home_wins = int(((h2h["Result"]=="H") & (h2h["HomeTeam"]==home_team)).sum() + ((h2h["Result"]=="A") & (h2h["AwayTeam"]==home_team)).sum())
away_wins = int(((h2h["Result"]=="H") & (h2h["HomeTeam"]==away_team)).sum() + ((h2h["Result"]=="A") & (h2h["AwayTeam"]==away_team)).sum())
draws = int((h2h["Result"]=="D").sum())
narr = [f"Over the recorded period, **{home_team}** and **{away_team}** met **{total_matches}** times."]
if total_matches>0:
    narr.append(f"They scored a combined **{total_goals}** goals (avg **{avg_goals:.2f}**).")
    narr.append(f"{home_team} wins: **{home_wins}**, {away_team} wins: **{away_wins}**, draws: **{draws}**.")
    dfc = h2h.copy()
    if not dfc.empty:
        dfc["TotalGoals"] = dfc["HomeGoals"] + dfc["AwayGoals"]
        dfc["GoalDiffAbs"] = (dfc["HomeGoals"]-dfc["AwayGoals"]).abs()
        dfc["Excitement"] = dfc["TotalGoals"] + (5-dfc["GoalDiffAbs"])
        best = dfc.sort_values("Excitement", ascending=False).head(3)
        narr.append("Notable matches:")
        for _, r in best.iterrows():
            date_str = r["Date"].date().isoformat() if not pd.isna(r["Date"]) else "Unknown"
            tag = "Goal Fest" if r["TotalGoals"]>=6 else ("Tight finish" if r["GoalDiffAbs"]<=1 else "Comfortable win")
            narr.append(f"- {date_str}: **{r['HomeTeam']} {int(r['HomeGoals'])}-{int(r['AwayGoals'])} {r['AwayTeam']}** ({tag})")
else:
    narr.append("No historic matches between these teams in dataset.")
st.markdown("<div class='panel'>", unsafe_allow_html=True)
for p in narr:
    st.markdown(p)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Team mini-cards & recent form
# ---------------------------
st.markdown("---")
st.subheader("Team Insights & Recent Form")
def recent_form(df: pd.DataFrame, team: str, n: int = 6):
    m = df[(df["HomeTeam"]==team)|(df["AwayTeam"]==team)].sort_values("Date", ascending=False).head(n)
    results = []
    for r in m.itertuples():
        if r.HomeTeam == team:
            results.append("W" if r.HomeGoals>r.AwayGoals else ("D" if r.HomeGoals==r.AwayGoals else "L"))
        else:
            results.append("W" if r.AwayGoals>r.HomeGoals else ("D" if r.AwayGoals==r.HomeGoals else "L"))
    return "".join(results), m

left, right = st.columns([1,1])
with left:
    st.markdown(f"<div class='panel' style='border-left:6px solid {home_color}'>", unsafe_allow_html=True)
    form_str, dfm = recent_form(historic_df, home_team, n=6)
    st.markdown(f"<div class='small muted'>Recent form (last 6):</div><div class='team-name'>{form_str or 'No matches'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small muted'>Avg goals (for/against at home): {safe_mean(historic_df[historic_df['HomeTeam']==home_team]['HomeGoals'],1.0):.2f} / {safe_mean(historic_df[historic_df['HomeTeam']==home_team]['AwayGoals'],1.0):.2f}</div>", unsafe_allow_html=True)
    if not dfm.empty:
        st.table(dfm.head(5)[["Date","HomeTeam","HomeGoals","AwayGoals","AwayTeam"]].assign(Date=lambda d: d["Date"].dt.date.astype(str)).reset_index(drop=True))
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown(f"<div class='panel' style='border-left:6px solid {away_color}'>", unsafe_allow_html=True)
    form_str2, dfm2 = recent_form(historic_df, away_team, n=6)
    st.markdown(f"<div class='small muted'>Recent form (last 6):</div><div class='team-name'>{form_str2 or 'No matches'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small muted'>Avg goals (for/against at home): {safe_mean(historic_df[historic_df['HomeTeam']==away_team]['HomeGoals'],1.0):.2f} / {safe_mean(historic_df[historic_df['HomeTeam']==away_team]['AwayGoals'],1.0):.2f}</div>", unsafe_allow_html=True)
    if not dfm2.empty:
        st.table(dfm2.head(5)[["Date","HomeTeam","HomeGoals","AwayGoals","AwayTeam"]].assign(Date=lambda d: d["Date"].dt.date.astype(str)).reset_index(drop=True))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Save blended prediction
# ---------------------------
st.markdown("---")
if st.button("Save blended prediction to data/predictions.csv"):
    out = {
        "Date": datetime.utcnow().isoformat(),
        "HomeTeam": home_team, "AwayTeam": away_team,
        "Model_Home": round(model_arr[0]*100,2), "Model_Draw": round(model_arr[1]*100,2), "Model_Away": round(model_arr[2]*100,2),
        "Market_Home": round(market_arr[0]*100,2), "Market_Draw": round(market_arr[1]*100,2), "Market_Away": round(market_arr[2]*100,2),
        "Blended_Home": round(blended[0]*100,2), "Blended_Draw": round(blended[1]*100,2), "Blended_Away": round(blended[2]*100,2),
        "ExpGoals_Home": round(exp_home,2), "ExpGoals_Away": round(exp_away,2),
        "TopScorelines": "; ".join([f"{int(r.HomeGoals)}-{int(r.AwayGoals)}:{r.Probability*100:.2f}%" for _,r in top_lines.iterrows()])
    }
    out_df = pd.DataFrame([out])
    path = "data/predictions.csv"
    if os.path.exists(path):
        prev = pd.read_csv(path)
        out_df = pd.concat([prev, out_df], ignore_index=True)
    out_df.to_csv(path, index=False)
    st.success(f"Saved → {path}")

# End app
