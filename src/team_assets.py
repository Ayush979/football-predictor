# src/team_assets.py
import os
import re
import requests
from urllib.parse import quote_plus

# Local cache folder for downloaded logos
LOGO_CACHE_DIR = "data/logos"
os.makedirs(LOGO_CACHE_DIR, exist_ok=True)

# Static color map for common Premier League teams (primary color hex)
# Matches the team list you provided earlier (adjust keys if your dataset uses different names)
STATIC_COLOR_MAP = {
    "liverpool":      {"color": "#C8102E"},
    "west ham":       {"color": "#7A263A"},
    "bournemouth":    {"color": "#DA291C"},
    "burnley":        {"color": "#6C1D45"},
    "crystal palace": {"color": "#1B458F"},
    "watford":        {"color": "#E2B007"},
    "tottenham":      {"color": "#132257"},
    "leicester":      {"color": "#003090"},
    "newcastle":      {"color": "#241F20"},
    "man united":     {"color": "#DA291C"},
    "arsenal":        {"color": "#EF0107"},
    "aston villa":    {"color": "#95BFE5"},
    "brighton":       {"color": "#0057B8"},
    "everton":        {"color": "#003399"},
    "norwich":        {"color": "#FFF200"},
    "southampton":    {"color": "#D71920"},
    "man city":       {"color": "#6CABDD"},
    "sheffield united":{"color":"#D71920"},
    "chelsea":        {"color": "#034694"},
    "wolves":         {"color": "#FDB913"},
    # add more if you have other names/variants
}

# normalization helper (so "Man United" -> "man united")
def normalize_name(name: str) -> str:
    if name is None:
        return ""
    s = re.sub(r"[^a-z0-9 ]", "", name.lower())
    s = s.replace("fc", "").replace("afc", "").replace("the ", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

# Try Wikipedia API to fetch a page thumbnail (fallback if static mapping logo absent)
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

def fetch_logo_from_wikipedia(team_name: str, save_local=True):
    """
    Query wikipedia for the team and return a logo image URL.
    If save_local is True, download and return local path under data/logos/.
    """
    q = team_name + " football club"
    params = {
        "action": "query",
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 400,
        "titles": q,
        "redirects": 1
    }
    try:
        r = requests.get(WIKI_API_URL, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                if not save_local:
                    return thumb
                # save local copy
                try:
                    resp = requests.get(thumb, timeout=8)
                    resp.raise_for_status()
                    ext = thumb.split("?")[0].split(".")[-1]
                    safe_name = normalize_name(team_name).replace(" ", "_")
                    fname = os.path.join(LOGO_CACHE_DIR, f"{safe_name}.{ext}")
                    with open(fname, "wb") as fh:
                        fh.write(resp.content)
                    return fname
                except Exception:
                    return thumb
    except Exception:
        return None
    return None

def get_team_asset(team_name: str, prefer_local=True):
    """
    Return {"logo": <url_or_local_path_or_None>, "color": <hex>} for a given team name.
    """
    n = normalize_name(team_name)
    color = STATIC_COLOR_MAP.get(n, {}).get("color", "#6C7A89")  # default muted
    # try cache: check file with normalized name
    for ext in ("png","jpg","jpeg","webp"):
        local = os.path.join(LOGO_CACHE_DIR, f"{n}.{ext}")
        if os.path.exists(local):
            return {"logo": local, "color": color}
    # try wikipedia fetch, save local if possible
    logo = fetch_logo_from_wikipedia(team_name, save_local=prefer_local)
    if logo:
        return {"logo": logo, "color": color}
    # fallback to a shaped placeholder (we can use a generated SVG via shields.io)
    placeholder = f"https://via.placeholder.com/96/{color.lstrip('#')}/FFFFFF?text={quote_plus(team_name.split()[0])}"
    return {"logo": placeholder, "color": color}
