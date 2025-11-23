# src/polymarket_api.py
"""
Robust Polymarket client with multiple fallbacks (P5):
- Try direct Polymarket endpoints (stable list)
- If direct fails, try proxy relays (AllOrigins and ThingProxy)
- Gracefully return None when no market/odds found

Public function:
    get_polymarket_odds(home_team: str, away_team: str) -> dict | None

Returned dict example:
    {"Liverpool": 54.12, "Chelsea": 38.40, "Draw": 7.48}
(probabilities as percentages)

Notes:
- Uses requests with retries and timeouts.
- Normalizes team names for fuzzy matching.
"""
from typing import Optional, Dict, Any, List
import requests
import time
import urllib.parse
import difflib

# Direct Polymarket endpoints (preferred)
DIRECT_ENDPOINTS = [
    "https://api.polymarket.com/events",
    "https://data-api.polymarket.com/events",
    "https://api.polymarket.com/markets",  # backup
]

# Proxy fallbacks (AllOrigins + ThingProxy)
PROXY_PREFIXES = [
    "https://api.allorigins.win/raw?url=",
    "https://thingproxy.freeboard.io/fetch/"
]

USER_AGENT = "PremierPredictor/1.0 (+https://github.com/your-repo)"

DEFAULT_TIMEOUT = 8.0
RETRY_SLEEP = 0.8
MAX_RETRIES = 2


def _get(url: str, headers: dict = None, timeout: float = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    headers = headers or {"User-Agent": USER_AGENT}
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            # if server returns 503/429 etc, let fallback handle it
        except Exception:
            pass
        time.sleep(RETRY_SLEEP)
    return None


def _normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("fc", "").replace("&", "and")
    s = s.replace("manchester united", "man united").replace("manchester city", "man city")
    s = s.replace("tottenham hotspur", "tottenham").replace("nottingham", "nott")
    s = s.strip()
    return s


def _event_matches_title(title: str, home: str, away: str) -> bool:
    t = _normalize_name(title)
    return _normalize_name(home) in t and _normalize_name(away) in t


def _choose_best_candidate(candidates: List[Dict[str, Any]], home: str, away: str) -> Optional[Dict[str, Any]]:
    # Prefer candidate that contains both names clearly, else fuzzy-match on title
    if not candidates:
        return None
    for c in candidates:
        title = c.get("title", "") or c.get("question", "")
        if title and _event_matches_title(title, home, away):
            return c
    # fallback fuzzy
    titles = [(i, (c.get("title") or c.get("question") or "")) for i, c in enumerate(candidates)]
    query = f"{home} vs {away}"
    title_list = [t for _, t in titles]
    best = difflib.get_close_matches(query, title_list, n=1, cutoff=0.2)
    if best:
        for i, t in titles:
            if t == best[0]:
                return candidates[i]
    return candidates[0]


def _fetch_events_from_endpoint(endpoint: str) -> Optional[List[Dict[str, Any]]]:
    # endpoint is expected to return JSON array of events or a dict with "events"
    r = _get(endpoint)
    if not r:
        return None
    try:
        j = r.json()
    except Exception:
        return None
    # normalize shape
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        if "events" in j and isinstance(j["events"], list):
            return j["events"]
        # sometimes APIs return dict-of-markets
        if "markets" in j and isinstance(j["markets"], list):
            return j["markets"]
    return None


def _fetch_event_details(event_obj: Dict[str, Any], direct_fetch_base: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Given an event-like object (may have id), try to fetch detailed event markets/outcomes.
    If event_obj has "id", attempt standard event detail URL via direct_fetch_base or common URL.
    """
    # If the event object already contains markets/outcomes, return it as-is
    if "markets" in event_obj or "outcomes" in event_obj:
        return event_obj

    # If it has an id, try to call the event detail path
    event_id = event_obj.get("id") or event_obj.get("_id")
    possible_detail_urls = []
    if event_id:
        possible_detail_urls += [
            f"https://api.polymarket.com/events/{event_id}",
            f"https://data-api.polymarket.com/events/{event_id}",
            f"https://api.polymarket.com/markets/{event_id}"
        ]
    # try each possible detail url directly
    for url in possible_detail_urls:
        r = _get(url)
        if r:
            try:
                return r.json()
            except Exception:
                continue
    # otherwise return the original object (best-effort)
    return event_obj


def _extract_outcome_probs_from_event_detail(detail: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Tries to find outcomes/prices under common keys and returns name->prob(%) mapping.
    """
    # common places: detail["markets"][0]["outcomes"], detail["outcomes"], detail.get("market",{}).get("outcomes")
    outcomes = None
    if not detail:
        return None
    if isinstance(detail, dict):
        if "markets" in detail and isinstance(detail["markets"], list) and len(detail["markets"]) > 0:
            # prefer winner market (first market)
            market = detail["markets"][0]
            outcomes = market.get("outcomes") or market.get("outcome") or market.get("prices") or None
        elif "outcomes" in detail and isinstance(detail["outcomes"], list):
            outcomes = detail["outcomes"]
        elif "market" in detail and isinstance(detail["market"], dict):
            outcomes = detail["market"].get("outcomes")
    if not outcomes:
        return None

    probs = {}
    # iterate outcomes and try to extract name and price/price_usd/price_percent/price
    for o in outcomes:
        if not isinstance(o, dict):
            continue
        name = o.get("title") or o.get("name") or o.get("outcome") or o.get("label") or str(o.get("id", "outcome"))
        # possible price keys
        price = None
        for key in ("price", "price_usd", "probability", "prob", "pricePercent", "price_percent"):
            if key in o:
                try:
                    price = float(o[key])
                    break
                except Exception:
                    pass
        # sometimes price is 0..1, sometimes 0..100. Normalize:
        if price is None:
            # maybe nested
            if "price" in o and isinstance(o["price"], dict):
                for pk in ("value", "raw", "usd"):
                    if pk in o["price"]:
                        try:
                            price = float(o["price"][pk])
                            break
                        except Exception:
                            pass
        if price is None:
            continue
        # normalize to percentage
        if price <= 1.001:
            prob = round(float(price) * 100, 2)
        else:
            prob = round(float(price), 2)
        probs[name] = prob
    if not probs:
        return None
    # normalize sum to 100 (if sums differ significantly)
    s = sum(probs.values())
    if s <= 0:
        return None
    # if sum is not close to 100, normalize to 100
    if abs(s - 100.0) > 1.0:
        probs = {k: round((v / s) * 100.0, 2) for k, v in probs.items()}
    return probs


def _try_direct_endpoints(home: str, away: str) -> Optional[Dict[str, float]]:
    for endpoint in DIRECT_ENDPOINTS:
        events = _fetch_events_from_endpoint(endpoint)
        if not events:
            continue
        # find candidates that match both team names in title / question
        candidates = []
        for e in events:
            title = e.get("title") or e.get("question") or e.get("name") or ""
            if title and _event_matches_title(title, home, away):
                candidates.append(e)
        if not candidates:
            # try looser contains (some endpoints may give markets, not event titles)
            for e in events:
                title = e.get("title") or e.get("question") or e.get("name") or ""
                if title and (_normalize_name(home) in title.lower() or _normalize_name(away) in title.lower()):
                    candidates.append(e)
        if not candidates:
            continue
        chosen = _choose_best_candidate(candidates, home, away)
        detail = _fetch_event_details(chosen)
        probs = _extract_outcome_probs_from_event_detail(detail)
        if probs:
            return probs
    return None


def _try_proxies(home: str, away: str) -> Optional[Dict[str, float]]:
    # Attempt each proxy prefix; for each, call the direct endpoints through the proxy.
    for proxy_prefix in PROXY_PREFIXES:
        for endpoint in DIRECT_ENDPOINTS:
            url = proxy_prefix + urllib.parse.quote(endpoint, safe="")
            r = _get(url)
            if not r:
                continue
            try:
                j = r.json()
            except Exception:
                continue
            # normalize as earlier: if top-level list/dict of events
            events = None
            if isinstance(j, list):
                events = j
            elif isinstance(j, dict):
                if "events" in j and isinstance(j["events"], list):
                    events = j["events"]
                elif "markets" in j and isinstance(j["markets"], list):
                    events = j["markets"]
                else:
                    # if proxy returns event detail directly, try to parse it
                    probs = _extract_outcome_probs_from_event_detail(j)
                    if probs:
                        return probs
            if not events:
                continue
            candidates = []
            for e in events:
                title = e.get("title") or e.get("question") or e.get("name") or ""
                if title and _event_matches_title(title, home, away):
                    candidates.append(e)
            if not candidates:
                for e in events:
                    title = e.get("title") or e.get("question") or e.get("name") or ""
                    if title and (_normalize_name(home) in title.lower() or _normalize_name(away) in title.lower()):
                        candidates.append(e)
            if not candidates:
                continue
            chosen = _choose_best_candidate(candidates, home, away)
            # If chosen contains id, fetch its details via proxy
            # attempt to fetch details via proxy too
            detail_url_candidates = []
            event_id = chosen.get("id") or chosen.get("_id")
            if event_id:
                for ep in DIRECT_ENDPOINTS:
                    # derive possible detail urls
                    detail_url_candidates.append(ep.rstrip("/") + f"/{event_id}")
            parsed = None
            for durl in detail_url_candidates:
                proxied = proxy_prefix + urllib.parse.quote(durl, safe="")
                rr = _get(proxied)
                if not rr:
                    continue
                try:
                    parsed_json = rr.json()
                except Exception:
                    continue
                parsed = parsed_json
                break
            if parsed:
                probs = _extract_outcome_probs_from_event_detail(parsed)
                if probs:
                    return probs
    return None


def get_polymarket_odds(home: str, away: str) -> Optional[Dict[str, float]]:
    """
    Top-level function to fetch odds.
    Tries direct endpoints first, then proxies. Returns dict or None.
    """
    # quick guards
    if not home or not away:
        return None
    # 1) direct endpoints
    try:
        probs = _try_direct_endpoints(home, away)
        if probs:
            return probs
    except Exception:
        pass
    # 2) proxies fallback
    try:
        probs = _try_proxies(home, away)
        if probs:
            return probs
    except Exception:
        pass
    # nothing found
    return None
