#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer v3.5
===============================
SPEED: Asyncio for parallel API calls
ODDS: Real sportsbook odds from The Odds API
EDGE: Vegas implied % vs Model % comparison
"""

import streamlit as st
import requests
import time
import math
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import pytz
import statistics

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NHL SOG Analyzer",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
NHL_WEB_API = "https://api-web.nhle.com/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SEASON = "20242025"
GAME_TYPE = 2
MIN_GAMES = 8
MIN_HIT_RATE = 75
EST = pytz.timezone('US/Eastern')

DEFENSE_GAMES = 5
CACHE_TTL = 1800

MATCHUP_GRADES = {
    "A+": 33.0, "A": 32.0, "B+": 31.0, "B": 30.0,
    "C+": 29.0, "C": 28.0, "D": 27.0, "F": 0.0,
}

# Preferred sportsbooks in order
PREFERRED_BOOKS = ["hardrock", "hardrockbet", "draftkings", "fanduel", "betmgm", "caesars"]

# ============================================================================
# MODEL WEIGHTS
# ============================================================================
MODEL_WEIGHTS = {
    "l5_weight": 0.45,
    "l10_weight": 0.30,
    "season_weight": 0.25,
    "hit_rate_95_boost": 1.15,
    "hit_rate_90_boost": 1.12,
    "hit_rate_85_boost": 1.08,
    "hit_rate_80_boost": 1.04,
    "pp1_boost": 1.18,
    "pp2_boost": 1.08,
    "home_boost": 1.03,
    "away_penalty": 0.97,
    "hot_streak_boost": 1.06,
    "cold_streak_penalty": 0.90,
    "forward_boost": 1.02,
    "defense_penalty": 0.96,
    "high_floor_1_boost": 1.04,
    "high_floor_2_boost": 1.06,
    "b2b_penalty": 0.94,
    "opp_loosening_boost": 1.04,
    "opp_tightening_penalty": 0.96,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_datetime():
    return datetime.now(EST)

def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def implied_prob_to_american(prob: float) -> int:
    if prob <= 0: return 10000
    if prob >= 1: return -10000
    if prob >= 0.5: return int(-100 * prob / (1 - prob))
    return int(100 * (1 - prob) / prob)

def calculate_parlay_odds(probs: List[float]) -> Tuple[float, int]:
    combined = 1.0
    for p in probs: combined *= p
    return combined, implied_prob_to_american(combined)

def calculate_parlay_payout(odds: int, stake: float = 100) -> float:
    if odds > 0: return stake + (stake * odds / 100)
    return stake + (stake * 100 / abs(odds))

def poisson_prob_at_least(lam: float, k: int) -> float:
    if lam <= 0: return 0.0
    return 1 - sum((math.exp(-lam) * (lam ** i)) / math.factorial(i) for i in range(k))

def get_grade(sa_pg: float) -> str:
    for grade, threshold in MATCHUP_GRADES.items():
        if sa_pg >= threshold: return grade
    return "F"

def get_trend(l5: float, season: float) -> Tuple[str, bool, bool]:
    diff = l5 - season
    if diff >= 0.5: return "🔥", True, False
    if diff <= -0.5: return "❄️", False, True
    return "➡️", False, False

def get_tags(player: Dict) -> str:
    tags = []
    if player["floor"] >= 1: tags.append("🛡️")
    if player["is_pp1"]: tags.append("⚡")
    if player["current_streak"] >= 5: tags.append(f"🔥{player['current_streak']}G")
    if player.get("is_b2b"): tags.append("B2B")
    return " ".join(tags)

def get_status_icon(hit_rate: float, is_cold: bool) -> Tuple[bool, str]:
    if hit_rate >= 85 and not is_cold: return True, "✅"
    return False, "⚠️"

def format_parlay_text(legs: List[Dict], threshold: int, name: str, prob: float, odds: int) -> str:
    text = f"🏒 NHL {name}\n" + "─" * 30 + "\n"
    for p in legs:
        player = p["player"]
        status = "✅" if p["is_qualified"] else "⚠️"
        text += f"{status} {player['name']} ({player['team']})\n"
        text += f"   O{threshold-0.5} SOG | {player[f'hit_rate_{threshold}plus']:.0f}% hit rate\n"
    text += "─" * 30 + f"\nProb: {prob*100:.0f}% | Odds: {odds:+d}\n"
    return text

def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    return name.lower().replace(".", "").replace("-", " ").strip()

# ============================================================================
# ODDS API FUNCTIONS
# ============================================================================
def fetch_nhl_events(api_key: str) -> List[Dict]:
    """Fetch today's NHL events from The Odds API."""
    if not api_key:
        return []
    
    url = f"{ODDS_API_BASE}/sports/icehockey_nhl/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []

def fetch_player_props(api_key: str, event_id: str) -> Dict[str, Dict]:
    """Fetch SOG props for an event. Returns {player_name: {line, odds, book}}"""
    if not api_key:
        return {}
    
    url = f"{ODDS_API_BASE}/sports/icehockey_nhl/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "player_shots_on_goal",
        "oddsFormat": "american"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {}
        
        data = resp.json()
        player_odds = {}
        
        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker.get("key", "").lower()
            
            # Check if this is a preferred book
            book_priority = 999
            for i, pref in enumerate(PREFERRED_BOOKS):
                if pref in book_key:
                    book_priority = i
                    break
            
            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_shots_on_goal":
                    continue
                
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "")
                    line = outcome.get("point", 0)
                    price = outcome.get("price", 0)
                    outcome_type = outcome.get("name", "")  # "Over" or "Under"
                    
                    if outcome_type != "Over":
                        continue
                    
                    normalized = normalize_name(player_name)
                    
                    # Only update if this book is higher priority
                    if normalized not in player_odds or book_priority < player_odds[normalized].get("priority", 999):
                        player_odds[normalized] = {
                            "name": player_name,
                            "line": line,
                            "odds": price,
                            "book": bookmaker.get("title", book_key),
                            "priority": book_priority
                        }
        
        return player_odds
    except Exception as e:
        return {}

def fetch_all_odds(api_key: str, status_text=None) -> Dict[str, Dict]:
    """Fetch all SOG odds for today's games."""
    if not api_key:
        return {}
    
    all_odds = {}
    
    if status_text:
        status_text.text("📊 Fetching sportsbook odds...")
    
    events = fetch_nhl_events(api_key)
    
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        
        props = fetch_player_props(api_key, event_id)
        all_odds.update(props)
    
    return all_odds

# ============================================================================
# PROBABILITY MODEL
# ============================================================================
def calculate_model_probability(player: Dict, opp_def: Dict, is_home: bool, threshold: int) -> float:
    base_lambda = (
        player["last_5_avg"] * MODEL_WEIGHTS["l5_weight"] +
        player["last_10_avg"] * MODEL_WEIGHTS["l10_weight"] +
        player["avg_sog"] * MODEL_WEIGHTS["season_weight"]
    )
    
    hit_rate = player[f"hit_rate_{threshold}plus"]
    if hit_rate >= 95: hr_factor = MODEL_WEIGHTS["hit_rate_95_boost"]
    elif hit_rate >= 90: hr_factor = MODEL_WEIGHTS["hit_rate_90_boost"]
    elif hit_rate >= 85: hr_factor = MODEL_WEIGHTS["hit_rate_85_boost"]
    elif hit_rate >= 80: hr_factor = MODEL_WEIGHTS["hit_rate_80_boost"]
    else: hr_factor = 1.0
    
    if player["is_pp1"]: pp_factor = MODEL_WEIGHTS["pp1_boost"]
    elif player["pp_toi"] > 1.0: pp_factor = MODEL_WEIGHTS["pp2_boost"]
    else: pp_factor = 1.0
    
    ha_factor = MODEL_WEIGHTS["home_boost"] if is_home else MODEL_WEIGHTS["away_penalty"]
    
    opp_sa = opp_def.get("shots_allowed_per_game", 30.0)
    opp_factor = opp_sa / 30.0
    
    opp_trend = opp_def.get("trend", "stable")
    if opp_trend == "loosening": opp_factor *= MODEL_WEIGHTS["opp_loosening_boost"]
    elif opp_trend == "tightening": opp_factor *= MODEL_WEIGHTS["opp_tightening_penalty"]
    
    trend, is_hot, is_cold = get_trend(player["last_5_avg"], player["avg_sog"])
    if is_hot: streak_factor = MODEL_WEIGHTS["hot_streak_boost"]
    elif is_cold: streak_factor = MODEL_WEIGHTS["cold_streak_penalty"]
    else: streak_factor = 1.0
    
    if player["position"] in ["C", "L", "R", "F"]: pos_factor = MODEL_WEIGHTS["forward_boost"]
    else: pos_factor = MODEL_WEIGHTS["defense_penalty"]
    
    if player["floor"] >= 2: floor_factor = MODEL_WEIGHTS["high_floor_2_boost"]
    elif player["floor"] >= 1: floor_factor = MODEL_WEIGHTS["high_floor_1_boost"]
    else: floor_factor = 1.0
    
    b2b_factor = MODEL_WEIGHTS["b2b_penalty"] if player.get("is_b2b") else 1.0
    
    adj_lambda = base_lambda * hr_factor * pp_factor * ha_factor * opp_factor * streak_factor * pos_factor * floor_factor * b2b_factor
    
    return poisson_prob_at_least(adj_lambda, threshold)

# ============================================================================
# NHL API FUNCTIONS
# ============================================================================
@st.cache_data(ttl=300)
def get_todays_schedule(date_str: str) -> List[Dict]:
    url = f"{NHL_WEB_API}/schedule/{date_str}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        games = []
        for game_week in data.get("gameWeek", []):
            if game_week.get("date") == date_str:
                for game in game_week.get("games", []):
                    away = game.get("awayTeam", {}).get("abbrev", "")
                    home = game.get("homeTeam", {}).get("abbrev", "")
                    game_id = str(game.get("id", ""))
                    
                    if not away or not home:
                        continue
                    
                    try:
                        utc_dt = datetime.fromisoformat(game.get("startTimeUTC", "").replace("Z", "+00:00"))
                        est_dt = utc_dt.astimezone(EST)
                        time_str = est_dt.strftime("%I:%M %p")
                    except:
                        time_str = "TBD"
                    
                    games.append({
                        "id": game_id,
                        "time": time_str,
                        "away_team": away,
                        "home_team": home,
                        "matchup": f"{away} @ {home}"
                    })
        return games
    except:
        return []

@st.cache_data(ttl=1800)
def get_all_teams() -> List[Dict]:
    url = f"{NHL_WEB_API}/standings/now"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return [{"abbrev": t.get("teamAbbrev", {}).get("default", ""),
                 "name": t.get("teamName", {}).get("default", "")} 
                for t in resp.json().get("standings", [])]
    except:
        return []

def fetch_team_defense(team_abbrev: str) -> Dict:
    try:
        url = f"{NHL_WEB_API}/club-schedule-season/{team_abbrev}/{SEASON}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        completed = [g for g in resp.json().get("games", []) 
                    if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
        
        if not completed:
            return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable"}
        
        recent = completed[-DEFENSE_GAMES:]
        sa_list = []
        
        for game in recent:
            try:
                box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
                box_resp = requests.get(box_url, timeout=8)
                box_data = box_resp.json()
                
                home_abbrev = box_data.get("homeTeam", {}).get("abbrev", "")
                home_sog = box_data.get("homeTeam", {}).get("sog", 0)
                away_sog = box_data.get("awayTeam", {}).get("sog", 0)
                
                if home_sog == 0 and away_sog == 0:
                    for stat in box_data.get("boxscore", {}).get("teamGameStats", []):
                        if stat.get("category") == "sog":
                            home_sog = stat.get("homeValue", 0)
                            away_sog = stat.get("awayValue", 0)
                            break
                
                if home_sog > 0 or away_sog > 0:
                    sa = away_sog if team_abbrev == home_abbrev else home_sog
                    if sa > 0:
                        sa_list.append(sa)
            except:
                continue
        
        if sa_list:
            sa_pg = statistics.mean(sa_list)
            if len(sa_list) >= 4:
                first_half = statistics.mean(sa_list[len(sa_list)//2:])
                second_half = statistics.mean(sa_list[:len(sa_list)//2])
                if second_half > first_half + 2: trend = "loosening"
                elif second_half < first_half - 2: trend = "tightening"
                else: trend = "stable"
            else:
                trend = "stable"
        else:
            sa_pg, trend = 30.0, "stable"
        
        return {"team_abbrev": team_abbrev, "shots_allowed_per_game": round(sa_pg, 2), "grade": get_grade(sa_pg), "trend": trend}
    except:
        return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable"}

def fetch_roster(team_abbrev: str) -> List[Dict]:
    url = f"{NHL_WEB_API}/roster/{team_abbrev}/current"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        players = []
        for cat in ["forwards", "defensemen"]:
            for p in resp.json().get(cat, []):
                first = p.get('firstName', {}).get('default', '')
                last = p.get('lastName', {}).get('default', '')
                if first and last and p.get("id"):
                    players.append({"id": p["id"], "name": f"{first} {last}", "position": p.get("positionCode", ""), "team": team_abbrev})
        return players
    except:
        return []

def fetch_player_full(player_info: Dict) -> Optional[Dict]:
    player_id = player_info["id"]
    name = player_info["name"]
    team = player_info["team"]
    position = player_info["position"]
    
    url = f"{NHL_WEB_API}/player/{player_id}/game-log/{SEASON}/{GAME_TYPE}"
    try:
        resp = requests.get(url, timeout=10)
        games = resp.json().get("gameLog", [])
        
        if len(games) < MIN_GAMES:
            return None
        
        all_shots, home_shots, away_shots, game_dates = [], [], [], []
        
        for game in games:
            shots = max(0, game.get("shots", 0))
            all_shots.append(shots)
            game_dates.append(game.get("gameDate", ""))
            if game.get("homeRoadFlag", "") == "H":
                home_shots.append(shots)
            else:
                away_shots.append(shots)
        
        if not all_shots:
            return None
        
        gp = len(all_shots)
        avg = sum(all_shots) / gp
        
        hit_2 = sum(1 for s in all_shots if s >= 2) / gp * 100
        hit_3 = sum(1 for s in all_shots if s >= 3) / gp * 100
        hit_4 = sum(1 for s in all_shots if s >= 4) / gp * 100
        
        l5 = all_shots[:5] if len(all_shots) >= 5 else all_shots
        l10 = all_shots[:10] if len(all_shots) >= 10 else all_shots
        l5_avg = sum(l5) / len(l5)
        l10_avg = sum(l10) / len(l10)
        
        std = statistics.stdev(all_shots) if len(all_shots) > 1 else 0
        
        streak = 0
        for s in all_shots:
            if s >= 2: streak += 1
            else: break
        
        home_avg = sum(home_shots) / len(home_shots) if home_shots else avg
        away_avg = sum(away_shots) / len(away_shots) if away_shots else avg
        
        pp_toi = 0
        try:
            landing_url = f"{NHL_WEB_API}/player/{player_id}/landing"
            landing_resp = requests.get(landing_url, timeout=8)
            for season in landing_resp.json().get("seasonTotals", []):
                if str(season.get("season")) == SEASON and season.get("gameTypeId") == GAME_TYPE:
                    pp_toi_str = season.get("powerPlayToi", "00:00")
                    season_gp = season.get("gamesPlayed", 1)
                    try:
                        parts = pp_toi_str.split(":")
                        pp_toi = (int(parts[0]) + int(parts[1]) / 60) / season_gp if season_gp > 0 else 0
                    except:
                        pass
                    break
        except:
            pass
        
        is_b2b = False
        if len(game_dates) >= 1 and game_dates[0]:
            try:
                last_game = datetime.strptime(game_dates[0], "%Y-%m-%d")
                if (datetime.now() - last_game).days == 1:
                    is_b2b = True
            except:
                pass
        
        return {
            "player_id": player_id, "name": name, "team": team, "position": position,
            "games_played": gp,
            "hit_rate_2plus": round(hit_2, 1),
            "hit_rate_3plus": round(hit_3, 1),
            "hit_rate_4plus": round(hit_4, 1),
            "avg_sog": round(avg, 2),
            "last_5_avg": round(l5_avg, 2),
            "last_10_avg": round(l10_avg, 2),
            "std_dev": round(std, 2),
            "floor": min(all_shots),
            "ceiling": max(all_shots),
            "current_streak": streak,
            "home_avg": round(home_avg, 2),
            "away_avg": round(away_avg, 2),
            "pp_toi": round(pp_toi, 2),
            "is_pp1": pp_toi >= 2.0,
            "is_b2b": is_b2b,
        }
    except:
        return None

# ============================================================================
# PARLAY GENERATION
# ============================================================================
def generate_best_parlay(plays: List[Dict], num_legs: int, threshold: int) -> Optional[Dict]:
    if len(plays) < num_legs:
        return None
    
    prob_key = f"prob_{threshold}plus"
    sorted_plays = sorted(plays, key=lambda x: x[prob_key], reverse=True)
    best_legs = sorted_plays[:num_legs]
    probs = [p[prob_key] / 100 for p in best_legs]
    combined_prob, american_odds = calculate_parlay_odds(probs)
    
    return {
        "legs": best_legs, "num_legs": num_legs,
        "combined_prob": combined_prob,
        "american_odds": american_odds,
        "payout_per_100": calculate_parlay_payout(american_odds, 100)
    }

def generate_sgp_for_game(plays: List[Dict], game_id: str, threshold: int, min_legs: int = 3, min_odds: int = 300) -> Optional[Dict]:
    game_plays = [p for p in plays if p["game_id"] == game_id]
    if len(game_plays) < min_legs:
        return None
    
    prob_key = f"prob_{threshold}plus"
    sorted_plays = sorted(game_plays, key=lambda x: x[prob_key], reverse=True)
    
    for num_legs in range(min_legs, min(len(sorted_plays) + 1, 10)):
        legs = sorted_plays[:num_legs]
        probs = [p[prob_key] / 100 for p in legs]
        combined_prob, american_odds = calculate_parlay_odds(probs)
        
        if american_odds >= min_odds:
            qualified_count = sum(1 for p in legs if p["is_qualified"])
            risky_count = num_legs - qualified_count
            return {
                "legs": legs, "num_legs": num_legs, "combined_prob": combined_prob,
                "american_odds": american_odds, "payout_per_100": calculate_parlay_payout(american_odds, 100),
                "game_id": game_id, "qualified_count": qualified_count, "risky_count": risky_count,
                "risk_level": "🟢" if risky_count == 0 else ("🟡" if risky_count <= 1 else "🔴")
            }
    
    if len(sorted_plays) >= min_legs:
        legs = sorted_plays[:min_legs]
        probs = [p[prob_key] / 100 for p in legs]
        combined_prob, american_odds = calculate_parlay_odds(probs)
        qualified_count = sum(1 for p in legs if p["is_qualified"])
        return {
            "legs": legs, "num_legs": min_legs, "combined_prob": combined_prob,
            "american_odds": american_odds, "payout_per_100": calculate_parlay_payout(american_odds, 100),
            "game_id": game_id, "qualified_count": qualified_count, "risky_count": min_legs - qualified_count,
            "risk_level": "⚪"
        }
    return None

# ============================================================================
# UI COMPONENTS
# ============================================================================
def show_model_explanation():
    with st.expander("📖 Model v3.5 - How It Works", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Base Calculation
            **Lambda** = L5 (45%) + L10 (30%) + Avg (25%)
            
            ### Boosts
            | Factor | Multiplier |
            |--------|------------|
            | Hit Rate 95%+ | ×1.15 |
            | Hit Rate 90%+ | ×1.12 |
            | Hit Rate 85%+ | ×1.08 |
            | PP1 | ×1.18 |
            | Home | ×1.03 |
            | 🔥 Hot | ×1.06 |
            | Forward | ×1.02 |
            | Floor ≥2 | ×1.06 |
            """)
        with col2:
            st.markdown("""
            ### Penalties
            | Factor | Multiplier |
            |--------|------------|
            | ❄️ Cold | ×0.90 |
            | Away | ×0.97 |
            | Defenseman | ×0.96 |
            | Back-to-Back | ×0.94 |
            | Opp Tightening | ×0.96 |
            
            ### Edge Calculation
            **Edge** = Model% - Vegas%
            
            Positive = We think it hits more than Vegas
            """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer")
    st.caption("v3.5 | Real Odds + Edge Calculator")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        today_est = get_est_datetime().date()
        selected_date = st.date_input("📅 Date", value=today_est)
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        bet_type = st.radio("SOG Threshold:", ["Over 1.5 (2+ SOG)", "Over 2.5 (3+ SOG)", "Over 3.5 (4+ SOG)"], index=0)
        threshold = 2 if "1.5" in bet_type else 3 if "2.5" in bet_type else 4
        
        st.markdown("---")
        
        unit_size = st.number_input("Unit Size ($)", min_value=1, max_value=1000, value=25)
        
        st.markdown("---")
        
        st.subheader("📊 Odds API")
        api_key = st.text_input("The Odds API Key", type="password", help="Get free key at the-odds-api.com")
        
        if not api_key:
            st.caption("⚠️ No API key = no real odds")
            st.caption("[Get free key](https://the-odds-api.com)")
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"{get_est_datetime().strftime('%I:%M %p EST')}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 All Results", "🎯 Best Bets", "🎰 Parlays", "❓ Help"])
    
    if 'all_plays' not in st.session_state:
        st.session_state.all_plays = []
    if 'games' not in st.session_state:
        st.session_state.games = []
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 2
    
    with tab1:
        if run_analysis:
            plays, games = run_analysis_with_odds(date_str, threshold, api_key)
            st.session_state.all_plays = plays
            st.session_state.games = games
            st.session_state.threshold = threshold
        elif st.session_state.all_plays:
            display_all_results(st.session_state.all_plays, st.session_state.threshold, date_str)
        else:
            st.info("👈 Click **Run Analysis**")
            games = get_todays_schedule(date_str)
            if games:
                st.subheader(f"📅 {len(games)} Games Today")
                game_df = pd.DataFrame([{"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]} for g in games])
                st.dataframe(game_df, use_container_width=True, hide_index=True)
    
    with tab2:
        if st.session_state.all_plays:
            show_best_bets(st.session_state.all_plays, st.session_state.threshold)
        else:
            st.info("Run analysis first")
    
    with tab3:
        if st.session_state.all_plays:
            show_parlays(st.session_state.all_plays, st.session_state.games, st.session_state.threshold, unit_size)
        else:
            st.info("Run analysis first")
    
    with tab4:
        show_help()

def run_analysis_with_odds(date_str: str, threshold: int, api_key: str) -> Tuple[List[Dict], List[Dict]]:
    """Run analysis and merge with real sportsbook odds."""
    
    games = get_todays_schedule(date_str)
    
    if not games:
        st.error("No games found!")
        return [], []
    
    st.subheader(f"📅 {len(games)} Games Today")
    game_df = pd.DataFrame([{"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]} for g in games])
    st.dataframe(game_df, use_container_width=True, hide_index=True)
    
    teams_playing = set()
    game_info = {}
    
    for game in games:
        teams_playing.add(game["away_team"])
        teams_playing.add(game["home_team"])
        game_info[game["away_team"]] = {"opponent": game["home_team"], "home_away": "AWAY", "time": game["time"], "game_id": game["id"], "matchup": game["matchup"]}
        game_info[game["home_team"]] = {"opponent": game["away_team"], "home_away": "HOME", "time": game["time"], "game_id": game["id"], "matchup": game["matchup"]}
    
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_display = st.empty()
    
    start_time = time.time()
    
    # Fetch sportsbook odds first
    sportsbook_odds = {}
    if api_key:
        status_text.text("📊 Fetching sportsbook odds...")
        sportsbook_odds = fetch_all_odds(api_key, status_text)
        if sportsbook_odds:
            stats_display.text(f"Found odds for {len(sportsbook_odds)} players")
    progress_bar.progress(10)
    
    # Fetch defense
    status_text.text(f"🛡️ Fetching defense for {len(teams_playing)} teams...")
    team_defense = {}
    for i, team in enumerate(teams_playing):
        status_text.text(f"🛡️ Defense: {team}")
        team_defense[team] = fetch_team_defense(team)
    progress_bar.progress(20)
    
    # Fetch rosters
    status_text.text("📋 Fetching rosters...")
    all_players = []
    for team in teams_playing:
        all_players.extend(fetch_roster(team))
    progress_bar.progress(25)
    
    stats_display.text(f"Analyzing {len(all_players)} players...")
    
    # Analyze players
    all_plays = []
    total = len(all_players)
    
    for i, player_info in enumerate(all_players):
        pct = 25 + int((i / total) * 75)
        progress_bar.progress(pct)
        status_text.text(f"🔍 {player_info['name']} ({i+1}/{total})")
        
        try:
            stats = fetch_player_full(player_info)
            if not stats:
                continue
            
            hit_rate = stats["hit_rate_2plus"] if threshold == 2 else stats["hit_rate_3plus"] if threshold == 3 else stats["hit_rate_4plus"]
            if hit_rate < MIN_HIT_RATE:
                continue
            
            info = game_info.get(player_info["team"])
            if not info:
                continue
            
            opp = info["opponent"]
            opp_def = team_defense.get(opp, {"shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable"})
            is_home = info["home_away"] == "HOME"
            
            prob_2 = calculate_model_probability(stats, opp_def, is_home, 2)
            prob_3 = calculate_model_probability(stats, opp_def, is_home, 3)
            prob_4 = calculate_model_probability(stats, opp_def, is_home, 4)
            
            trend, is_hot, is_cold = get_trend(stats["last_5_avg"], stats["avg_sog"])
            is_qualified, status_icon = get_status_icon(hit_rate, is_cold)
            
            # Match with sportsbook odds
            normalized_name = normalize_name(stats["name"])
            odds_data = sportsbook_odds.get(normalized_name, {})
            
            # Get the right line based on threshold
            vegas_odds = None
            vegas_line = None
            vegas_implied = None
            vegas_book = None
            
            if odds_data:
                odds_line = odds_data.get("line", 0)
                # Check if line matches our threshold (1.5, 2.5, 3.5)
                target_line = threshold - 0.5
                if abs(odds_line - target_line) < 0.1:  # Line matches
                    vegas_odds = odds_data.get("odds")
                    vegas_line = odds_line
                    vegas_implied = american_to_implied_prob(vegas_odds) * 100 if vegas_odds else None
                    vegas_book = odds_data.get("book", "")
            
            # Calculate edge
            model_prob = prob_2 if threshold == 2 else prob_3 if threshold == 3 else prob_4
            edge = None
            if vegas_implied:
                edge = (model_prob * 100) - vegas_implied
            
            play = {
                "player": stats,
                "opponent": opp,
                "opponent_defense": opp_def,
                "home_away": info["home_away"],
                "game_time": info["time"],
                "game_id": info["game_id"],
                "matchup": info["matchup"],
                "prob_2plus": round(prob_2 * 100, 1),
                "prob_3plus": round(prob_3 * 100, 1),
                "prob_4plus": round(prob_4 * 100, 1),
                "is_qualified": is_qualified,
                "status_icon": status_icon,
                "trend": trend,
                "is_hot": is_hot,
                "is_cold": is_cold,
                "tags": get_tags(stats),
                # Odds data
                "vegas_odds": vegas_odds,
                "vegas_line": vegas_line,
                "vegas_implied": vegas_implied,
                "vegas_book": vegas_book,
                "edge": edge,
            }
            all_plays.append(play)
            stats_display.text(f"Checked: {i+1}/{total} | Found: {len(all_plays)}")
            
        except Exception as e:
            continue
    
    progress_bar.progress(100)
    elapsed = time.time() - start_time
    status_text.text(f"✅ Complete in {elapsed:.1f}s!")
    
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    stats_display.empty()
    
    # Sort by model probability
    prob_key = f"prob_{threshold}plus"
    all_plays.sort(key=lambda x: x[prob_key], reverse=True)
    
    # Count players with odds
    with_odds = len([p for p in all_plays if p["vegas_odds"]])
    st.success(f"Found **{len(all_plays)}** players | **{with_odds}** with odds | **{elapsed:.1f}s**")
    
    display_all_results(all_plays, threshold, date_str)
    
    return all_plays, games

def display_all_results(plays: List[Dict], threshold: int, date_str: str):
    st.subheader(f"🎯 All Players - O{threshold - 0.5} SOG")
    st.caption("Sorted by Model Probability | Green edge = value bet")
    
    show_model_explanation()
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    qualified = len([p for p in plays if p["is_qualified"]])
    with_edge = len([p for p in plays if p.get("edge") and p["edge"] > 0])
    pp1_count = len([p for p in plays if p["player"]["is_pp1"]])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(plays))
    col2.metric("✅ Qualified", qualified)
    col3.metric("📈 +Edge", with_edge)
    col4.metric("⚡ PP1", pp1_count)
    
    # Build results with edge column
    results_data = []
    for play in plays:
        p = play["player"]
        
        # Format edge
        edge_str = ""
        if play.get("edge") is not None:
            edge_val = play["edge"]
            if edge_val > 0:
                edge_str = f"+{edge_val:.1f}%"
            else:
                edge_str = f"{edge_val:.1f}%"
        
        # Format vegas
        vegas_str = ""
        if play.get("vegas_odds"):
            vegas_str = f"{play['vegas_odds']:+d}"
        
        vegas_impl_str = ""
        if play.get("vegas_implied"):
            vegas_impl_str = f"{play['vegas_implied']:.0f}%"
        
        row = {
            "": play["status_icon"],
            "Player": p["name"],
            "Tags": play["tags"],
            "Team": p["team"],
            "vs": play["opponent"],
            "Model%": play[prob_key],
            "Vegas%": vegas_impl_str,
            "Edge": edge_str,
            "Odds": vegas_str,
            "Hit%": p[hit_key],
            "Avg": p["avg_sog"],
            "L5": p["last_5_avg"],
            "Trend": play["trend"],
        }
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    
    # Style edge column
    def style_edge(val):
        if not val or val == "":
            return ""
        try:
            num = float(val.replace("%", "").replace("+", ""))
            if num > 5:
                return "color: #00ff00; font-weight: bold"
            elif num > 0:
                return "color: #90EE90"
            elif num < -5:
                return "color: #ff6b6b"
            else:
                return "color: #ffb347"
        except:
            return ""
    
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "": st.column_config.TextColumn("", width="small"),
            "Model%": st.column_config.ProgressColumn("Model%", min_value=0, max_value=100, format="%.1f%%"),
            "Hit%": st.column_config.NumberColumn("Hit%", format="%.1f%%"),
        }
    )
    
    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name=f"nhl_sog_{date_str}.csv", mime="text/csv")

def show_best_bets(plays: List[Dict], threshold: int):
    st.header("🎯 Best Bets")
    show_model_explanation()
    
    # Split into value bets vs all qualified
    value_bets = [p for p in plays if p.get("edge") and p["edge"] > 3 and p["is_qualified"]]
    qualified = [p for p in plays if p["is_qualified"]]
    risky = [p for p in plays if not p["is_qualified"]]
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    # Value bets section
    if value_bets:
        st.subheader(f"💰 Value Bets ({len(value_bets)})")
        st.caption("Qualified + Edge > 3%")
        
        value_data = []
        for play in sorted(value_bets, key=lambda x: x["edge"], reverse=True):
            p = play["player"]
            value_data.append({
                "Player": p["name"],
                "Tags": play["tags"],
                "Team": p["team"],
                "vs": play["opponent"],
                "Model%": f"{play[prob_key]:.1f}%",
                "Vegas%": f"{play['vegas_implied']:.0f}%" if play.get("vegas_implied") else "",
                "Edge": f"+{play['edge']:.1f}%",
                "Odds": f"{play['vegas_odds']:+d}" if play.get("vegas_odds") else "",
            })
        st.dataframe(pd.DataFrame(value_data), use_container_width=True, hide_index=True)
        st.markdown("---")
    
    st.subheader(f"✅ Qualified ({len(qualified)})")
    if qualified:
        qual_data = [{"Player": p["player"]["name"], "Tags": p["tags"], "Team": p["player"]["team"], 
                      "vs": p["opponent"], "Model%": f"{p[prob_key]:.1f}%",
                      "Vegas%": f"{p['vegas_implied']:.0f}%" if p.get("vegas_implied") else "",
                      "Edge": f"{p['edge']:+.1f}%" if p.get("edge") else "",
                      "Trend": p["trend"]} for p in qualified]
        st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No qualified plays")
    
    st.markdown("---")
    st.subheader(f"⚠️ Higher Risk ({len(risky)})")
    if risky:
        risk_data = [{"Player": p["player"]["name"], "Tags": p["tags"], "Team": p["player"]["team"],
                      "vs": p["opponent"], "Model%": f"{p[prob_key]:.1f}%",
                      "Vegas%": f"{p['vegas_implied']:.0f}%" if p.get("vegas_implied") else "",
                      "Trend": p["trend"]} for p in risky]
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

def show_parlays(plays: List[Dict], games: List[Dict], threshold: int, unit_size: float):
    st.header("🎰 Parlays")
    show_model_explanation()
    
    prob_key = f"prob_{threshold}plus"
    sorted_plays = sorted(plays, key=lambda x: x[prob_key], reverse=True)
    
    st.success(f"Building from **{len(sorted_plays)}** players")
    
    st.subheader("📊 Best Parlay by Legs")
    
    max_legs = min(12, len(sorted_plays))
    parlay_table = []
    parlays_dict = {}
    
    for num_legs in range(1, max_legs + 1):
        parlay = generate_best_parlay(sorted_plays, num_legs, threshold)
        if parlay:
            players = ", ".join([p["player"]["name"] for p in parlay["legs"]])
            parlay_table.append({
                "Legs": num_legs,
                "Prob%": f"{parlay['combined_prob']*100:.1f}%",
                "Odds": f"{parlay['american_odds']:+d}" if parlay['american_odds'] < 10000 else "—",
                f"${unit_size:.0f}→": f"${parlay['payout_per_100'] * unit_size / 100:.0f}",
                "Players": players[:55] + "..." if len(players) > 55 else players
            })
            parlays_dict[num_legs] = parlay
    
    if parlay_table:
        st.dataframe(pd.DataFrame(parlay_table), use_container_width=True, hide_index=True)
    
    st.markdown("### 📋 Click to Copy")
    cols = st.columns(4)
    for i, num_legs in enumerate([2, 3, 5, 10]):
        if num_legs in parlays_dict:
            parlay = parlays_dict[num_legs]
            with cols[i]:
                with st.expander(f"**{num_legs}-Leg** ({parlay['combined_prob']*100:.0f}%)"):
                    st.code(format_parlay_text(parlay["legs"], threshold, f"{num_legs}-Leg Parlay", parlay['combined_prob'], parlay['american_odds']), language=None)
    
    st.markdown("---")
    st.subheader("🏆 Ultimate Parlay")
    ultimate = generate_best_parlay(sorted_plays, min(20, len(sorted_plays)), threshold)
    if ultimate:
        col1, col2, col3 = st.columns(3)
        col1.metric("Legs", ultimate["num_legs"])
        col2.metric("Prob", f"{ultimate['combined_prob']*100:.2f}%")
        col3.metric("Odds", f"{ultimate['american_odds']:+d}" if ultimate['american_odds'] < 100000 else "Long shot")
        with st.expander("📋 View & Copy"):
            st.dataframe(pd.DataFrame([{"Player": p["player"]["name"], "Model%": f"{p[prob_key]:.0f}%"} for p in ultimate["legs"]]), hide_index=True)
            st.code(format_parlay_text(ultimate["legs"], threshold, f"Ultimate {ultimate['num_legs']}-Leg", ultimate['combined_prob'], ultimate['american_odds']), language=None)
    
    st.markdown("---")
    st.subheader("🎮 Single Game Parlays")
    for game in games:
        sgp = generate_sgp_for_game(sorted_plays, game["id"], threshold)
        if sgp:
            with st.expander(f"**{game['matchup']}** | {sgp['american_odds']:+d} | {sgp['risk_level']} {sgp['qualified_count']}✅ {sgp['risky_count']}⚠️"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Legs", sgp["num_legs"])
                col2.metric("Prob", f"{sgp['combined_prob']*100:.1f}%")
                col3.metric("Odds", f"{sgp['american_odds']:+d}")
                col4.metric(f"${unit_size:.0f}→", f"${sgp['payout_per_100'] * unit_size / 100:.0f}")
                st.dataframe(pd.DataFrame([{"": "✅" if p["is_qualified"] else "⚠️", "Player": p["player"]["name"], "Model%": f"{p[prob_key]:.0f}%"} for p in sgp["legs"]]), hide_index=True)
                copy_text = f"🎮 SGP - {game['matchup']}\n" + "─"*30 + "\n"
                for p in sgp["legs"]:
                    copy_text += f"{'✅' if p['is_qualified'] else '⚠️'} {p['player']['name']} O{threshold-0.5} SOG\n"
                copy_text += "─"*30 + f"\nOdds: {sgp['american_odds']:+d}\n"
                st.code(copy_text, language=None)

def show_help():
    st.header("❓ Help")
    show_model_explanation()
    st.markdown("""
    ## v3.5 Features
    
    ### 📊 Real Odds Integration
    - Add your **The Odds API** key in sidebar
    - Shows actual sportsbook odds (Hard Rock, DK, FD, etc.)
    - Free tier: 500 requests/month
    - Get key at [the-odds-api.com](https://the-odds-api.com)
    
    ### 📈 Edge Calculator
    - **Model%** = Our probability prediction
    - **Vegas%** = Implied probability from sportsbook odds
    - **Edge** = Model% - Vegas%
    
    | Edge | Meaning |
    |------|---------|
    | +5%+ | Strong value |
    | +1-5% | Slight edge |
    | 0% | Fair line |
    | Negative | Overpriced |
    
    ### 💰 Value Bets
    Best Bets tab shows **Value Bets** section:
    - Qualified players (85%+, not cold)
    - Edge > 3%
    - Sorted by edge
    
    ### Without API Key
    Works fine without odds - just won't show:
    - Vegas% column
    - Edge column
    - Value Bets section
    """)

if __name__ == "__main__":
    main()
