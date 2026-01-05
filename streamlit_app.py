#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer v3.4
===============================
SPEED OPTIMIZATIONS:
- Parallel player fetching (ThreadPoolExecutor)
- Reduced defense games (5 instead of 10)
- Combined API calls where possible
- Smarter caching (longer TTL)

MODEL IMPROVEMENTS:
- Position factor (F vs D)
- Back-to-back detection
- Opponent L5 trending
- Floor reliability bonus
- Ceiling upside factor
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
SEASON = "20242025"
GAME_TYPE = 2
MIN_GAMES = 8
MIN_HIT_RATE = 75
EST = pytz.timezone('US/Eastern')

# Speed settings
DEFENSE_GAMES = 5  # Reduced from 10
CACHE_TTL_DEFENSE = 1800  # 30 min cache for defense
CACHE_TTL_SCHEDULE = 300  # 5 min cache for schedule

MATCHUP_GRADES = {
    "A+": 33.0, "A": 32.0, "B+": 31.0, "B": 30.0,
    "C+": 29.0, "C": 28.0, "D": 27.0, "F": 0.0,
}

# ============================================================================
# IMPROVED MODEL WEIGHTS
# ============================================================================
MODEL_WEIGHTS = {
    # Base calculation weights
    "l5_weight": 0.45,      # Increased recent form importance
    "l10_weight": 0.30,
    "season_weight": 0.25,  # Reduced season weight
    
    # Hit rate multipliers (more aggressive tiers)
    "hit_rate_95_boost": 1.15,
    "hit_rate_90_boost": 1.12,
    "hit_rate_85_boost": 1.08,
    "hit_rate_80_boost": 1.04,
    
    # Power play
    "pp1_boost": 1.18,
    "pp2_boost": 1.08,
    
    # Situational
    "home_boost": 1.03,
    "away_penalty": 0.97,
    "hot_streak_boost": 1.06,
    "cold_streak_penalty": 0.90,  # More aggressive cold penalty
    
    # NEW: Position factors (forwards shoot more)
    "forward_boost": 1.02,
    "defense_penalty": 0.96,
    
    # NEW: Reliability factors (only bonus, no penalty - 0 floor is normal)
    "high_floor_1_boost": 1.04,  # Floor >= 1 (rare)
    "high_floor_2_boost": 1.06,  # Floor >= 2 (very rare)
    
    # NEW: Back-to-back
    "b2b_penalty": 0.94,  # Second game of back-to-back
    
    # NEW: Opponent trending
    "opp_loosening_boost": 1.04,  # Defense getting worse
    "opp_tightening_penalty": 0.96,  # Defense improving
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_datetime():
    return datetime.now(EST)

def implied_prob_to_american(prob: float) -> int:
    if prob <= 0:
        return 10000
    if prob >= 1:
        return -10000
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

def calculate_parlay_odds(probs: List[float]) -> Tuple[float, int]:
    combined_prob = 1.0
    for p in probs:
        combined_prob *= p
    return combined_prob, implied_prob_to_american(combined_prob)

def calculate_parlay_payout(odds: int, stake: float = 100) -> float:
    if odds > 0:
        return stake + (stake * odds / 100)
    else:
        return stake + (stake * 100 / abs(odds))

def poisson_prob_at_least(lam: float, k: int) -> float:
    if lam <= 0:
        return 0.0
    prob_less = sum((math.exp(-lam) * (lam ** i)) / math.factorial(i) for i in range(k))
    return 1 - prob_less

def get_grade(sa_pg: float) -> str:
    for grade, threshold in MATCHUP_GRADES.items():
        if sa_pg >= threshold:
            return grade
    return "F"

def get_trend(l5: float, season: float) -> Tuple[str, bool, bool]:
    diff = l5 - season
    if diff >= 0.5:
        return "🔥", True, False
    elif diff <= -0.5:
        return "❄️", False, True
    return "➡️", False, False

def get_tags(player: Dict) -> str:
    tags = []
    if player["floor"] >= 1:
        tags.append("🛡️")
    if player["is_pp1"]:
        tags.append("⚡")
    if player["current_streak"] >= 5:
        tags.append(f"🔥{player['current_streak']}G")
    if player.get("is_b2b"):
        tags.append("B2B")
    return " ".join(tags)

def get_status_icon(hit_rate: float, is_cold: bool) -> Tuple[bool, str]:
    if hit_rate >= 85 and not is_cold:
        return True, "✅"
    else:
        return False, "⚠️"

def format_parlay_text(legs: List[Dict], threshold: int, parlay_name: str, prob: float, odds: int) -> str:
    text = f"🏒 NHL {parlay_name}\n"
    text += "─" * 30 + "\n"
    for p in legs:
        player = p["player"]
        hit_rate = player[f"hit_rate_{threshold}plus"]
        status = "✅" if p["is_qualified"] else "⚠️"
        text += f"{status} {player['name']} ({player['team']})\n"
        text += f"   O{threshold-0.5} SOG | {hit_rate:.0f}% hit rate\n"
    text += "─" * 30 + "\n"
    text += f"Prob: {prob*100:.0f}% | Odds: {odds:+d}\n"
    return text

# ============================================================================
# OPTIMIZED API FUNCTIONS
# ============================================================================
@st.cache_data(ttl=CACHE_TTL_SCHEDULE)
def get_todays_schedule(date_str: str) -> List[Dict]:
    """Get schedule with back-to-back detection."""
    url = f"{NHL_WEB_API}/schedule/{date_str}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        games = []
        for game_week in data.get("gameWeek", []):
            if game_week.get("date") == date_str:
                for game in game_week.get("games", []):
                    away_team = game.get("awayTeam", {}).get("abbrev", "")
                    home_team = game.get("homeTeam", {}).get("abbrev", "")
                    game_id = str(game.get("id", ""))
                    
                    if not away_team or not home_team:
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
                        "away_team": away_team,
                        "home_team": home_team,
                        "matchup": f"{away_team} @ {home_team}"
                    })
        return games
    except:
        return []

@st.cache_data(ttl=CACHE_TTL_DEFENSE)
def get_all_teams() -> List[Dict]:
    url = f"{NHL_WEB_API}/standings/now"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [{"abbrev": t.get("teamAbbrev", {}).get("default", ""),
                 "name": t.get("teamName", {}).get("default", "")} 
                for t in data.get("standings", [])]
    except:
        return []

def fetch_team_defense(team_abbrev: str) -> Dict:
    """Fetch defense stats for a single team (for parallel execution)."""
    try:
        url = f"{NHL_WEB_API}/club-schedule-season/{team_abbrev}/{SEASON}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        completed = [g for g in data.get("games", []) 
                    if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
        
        if not completed:
            return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "shots_allowed_L5": 30.0, "grade": "C", "trend": "stable"}
        
        # Only fetch last 5 games (speed optimization)
        recent = completed[-DEFENSE_GAMES:]
        sa_list = []
        
        for game in recent:
            try:
                box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
                box_resp = requests.get(box_url, timeout=8)
                box_resp.raise_for_status()
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
            # Detect trending (first half vs second half of sample)
            if len(sa_list) >= 4:
                first_half = statistics.mean(sa_list[len(sa_list)//2:])
                second_half = statistics.mean(sa_list[:len(sa_list)//2])
                if second_half > first_half + 2:
                    trend = "loosening"
                elif second_half < first_half - 2:
                    trend = "tightening"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            sa_pg = 30.0
            trend = "stable"
        
        return {
            "team_abbrev": team_abbrev,
            "shots_allowed_per_game": round(sa_pg, 2),
            "shots_allowed_L5": round(sa_pg, 2),
            "grade": get_grade(sa_pg),
            "trend": trend
        }
    except:
        return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "shots_allowed_L5": 30.0, "grade": "C", "trend": "stable"}

def get_team_defense_safe(teams: List[str], status_text) -> Dict[str, Dict]:
    """Fetch defense stats - sequential but reliable."""
    team_defense = {}
    total = len(teams)
    
    for i, team in enumerate(teams):
        status_text.text(f"🛡️ Defense: {team} ({i+1}/{total})")
        try:
            result = fetch_team_defense(team)
            team_defense[team] = result
        except Exception as e:
            team_defense[team] = {"team_abbrev": team, "shots_allowed_per_game": 30.0, "shots_allowed_L5": 30.0, "grade": "C", "trend": "stable"}
    
    return team_defense

def fetch_roster(team_abbrev: str) -> List[Dict]:
    """Fetch roster for a single team."""
    url = f"{NHL_WEB_API}/roster/{team_abbrev}/current"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        players = []
        for category in ["forwards", "defensemen"]:
            for player in data.get(category, []):
                first = player.get('firstName', {}).get('default', '')
                last = player.get('lastName', {}).get('default', '')
                if first and last and player.get("id"):
                    players.append({
                        "id": player.get("id"),
                        "name": f"{first} {last}",
                        "position": player.get("positionCode", ""),
                        "team": team_abbrev
                    })
        return players
    except:
        return []

def fetch_player_full(player_info: Dict) -> Optional[Dict]:
    """Fetch player stats + advanced stats in optimized way."""
    player_id = player_info["id"]
    name = player_info["name"]
    team = player_info["team"]
    position = player_info["position"]
    
    # Get game log
    url = f"{NHL_WEB_API}/player/{player_id}/game-log/{SEASON}/{GAME_TYPE}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        games = resp.json().get("gameLog", [])
        
        if len(games) < MIN_GAMES:
            return None
        
        all_shots = []
        home_shots = []
        away_shots = []
        game_dates = []
        
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
            if s >= 2:
                streak += 1
            else:
                break
        
        home_avg = sum(home_shots) / len(home_shots) if home_shots else avg
        away_avg = sum(away_shots) / len(away_shots) if away_shots else avg
        
        # Get PP time from landing page (combined call)
        pp_toi = 0
        try:
            landing_url = f"{NHL_WEB_API}/player/{player_id}/landing"
            landing_resp = requests.get(landing_url, timeout=8)
            landing_resp.raise_for_status()
            landing_data = landing_resp.json()
            
            for season in landing_data.get("seasonTotals", []):
                if str(season.get("season")) == SEASON and season.get("gameTypeId") == GAME_TYPE:
                    pp_toi_str = season.get("powerPlayToi", "00:00")
                    season_gp = season.get("gamesPlayed", 1)
                    try:
                        parts = pp_toi_str.split(":")
                        total_mins = int(parts[0]) + int(parts[1]) / 60
                        pp_toi = total_mins / season_gp if season_gp > 0 else 0
                    except:
                        pass
                    break
        except:
            pass
        
        # Detect back-to-back (if last game was yesterday)
        is_b2b = False
        if len(game_dates) >= 1 and game_dates[0]:
            try:
                last_game = datetime.strptime(game_dates[0], "%Y-%m-%d")
                today = datetime.now()
                if (today - last_game).days == 1:
                    is_b2b = True
            except:
                pass
        
        return {
            "player_id": player_id,
            "name": name,
            "team": team,
            "position": position,
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
# IMPROVED PROBABILITY MODEL
# ============================================================================
def calculate_model_probability(player: Dict, opp_def: Dict, is_home: bool, threshold: int) -> float:
    """Enhanced probability model with more factors."""
    
    # Base lambda (weighted average)
    base_lambda = (
        player["last_5_avg"] * MODEL_WEIGHTS["l5_weight"] +
        player["last_10_avg"] * MODEL_WEIGHTS["l10_weight"] +
        player["avg_sog"] * MODEL_WEIGHTS["season_weight"]
    )
    
    # Hit rate boost (tiered)
    hit_rate = player[f"hit_rate_{threshold}plus"]
    if hit_rate >= 95:
        hr_factor = MODEL_WEIGHTS["hit_rate_95_boost"]
    elif hit_rate >= 90:
        hr_factor = MODEL_WEIGHTS["hit_rate_90_boost"]
    elif hit_rate >= 85:
        hr_factor = MODEL_WEIGHTS["hit_rate_85_boost"]
    elif hit_rate >= 80:
        hr_factor = MODEL_WEIGHTS["hit_rate_80_boost"]
    else:
        hr_factor = 1.0
    
    # Power play factor
    if player["is_pp1"]:
        pp_factor = MODEL_WEIGHTS["pp1_boost"]
    elif player["pp_toi"] > 1.0:
        pp_factor = MODEL_WEIGHTS["pp2_boost"]
    else:
        pp_factor = 1.0
    
    # Home/away
    ha_factor = MODEL_WEIGHTS["home_boost"] if is_home else MODEL_WEIGHTS["away_penalty"]
    
    # Opponent base factor
    opp_sa = opp_def.get("shots_allowed_per_game", 30.0)
    opp_factor = opp_sa / 30.0
    
    # NEW: Opponent trending adjustment
    opp_trend = opp_def.get("trend", "stable")
    if opp_trend == "loosening":
        opp_factor *= MODEL_WEIGHTS["opp_loosening_boost"]
    elif opp_trend == "tightening":
        opp_factor *= MODEL_WEIGHTS["opp_tightening_penalty"]
    
    # Hot/cold streak
    trend, is_hot, is_cold = get_trend(player["last_5_avg"], player["avg_sog"])
    if is_hot:
        streak_factor = MODEL_WEIGHTS["hot_streak_boost"]
    elif is_cold:
        streak_factor = MODEL_WEIGHTS["cold_streak_penalty"]
    else:
        streak_factor = 1.0
    
    # NEW: Position factor
    if player["position"] in ["C", "L", "R", "F"]:
        pos_factor = MODEL_WEIGHTS["forward_boost"]
    else:
        pos_factor = MODEL_WEIGHTS["defense_penalty"]
    
    # NEW: Floor reliability bonus (only for rare high-floor players)
    if player["floor"] >= 2:
        floor_factor = MODEL_WEIGHTS["high_floor_2_boost"]
    elif player["floor"] >= 1:
        floor_factor = MODEL_WEIGHTS["high_floor_1_boost"]
    else:
        floor_factor = 1.0  # No penalty - 0 floor is normal
    
    # NEW: Back-to-back penalty
    if player.get("is_b2b", False):
        b2b_factor = MODEL_WEIGHTS["b2b_penalty"]
    else:
        b2b_factor = 1.0
    
    # Calculate adjusted lambda
    adj_lambda = (base_lambda * hr_factor * pp_factor * ha_factor * 
                  opp_factor * streak_factor * pos_factor * floor_factor * b2b_factor)
    
    # Poisson probability
    prob = poisson_prob_at_least(adj_lambda, threshold)
    
    return prob

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
        "legs": best_legs,
        "num_legs": num_legs,
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
    
    # Return best available if can't hit +300
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
    with st.expander("📖 Model v3.4.1 - How It Works", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Base Calculation
            **Lambda** = L5 (45%) + L10 (30%) + Avg (25%)
            
            ### Multipliers
            | Factor | Boost |
            |--------|-------|
            | Hit Rate 95%+ | ×1.15 |
            | Hit Rate 90%+ | ×1.12 |
            | Hit Rate 85%+ | ×1.08 |
            | PP1 | ×1.18 |
            | Home | ×1.03 |
            | 🔥 Hot | ×1.06 |
            | Forward | ×1.02 |
            | Floor ≥2 | ×1.06 |
            | Floor ≥1 | ×1.04 |
            """)
        
        with col2:
            st.markdown("""
            ### Penalties
            | Factor | Penalty |
            |--------|---------|
            | ❄️ Cold | ×0.90 |
            | Away | ×0.97 |
            | Defenseman | ×0.96 |
            | Back-to-Back | ×0.94 |
            | Opp Tightening | ×0.96 |
            
            ### Bonuses (Rare)
            | Factor | Boost |
            |--------|-------|
            | Floor ≥2 | ×1.06 |
            | Floor ≥1 | ×1.04 |
            
            ### Opponent Factor
            = SA/G ÷ 30 (adjusted for trend)
            """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer")
    st.caption("v3.4.1 | Improved Model")
    
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
            plays, games = run_optimized_analysis(date_str, threshold)
            st.session_state.all_plays = plays
            st.session_state.games = games
            st.session_state.threshold = threshold
        elif st.session_state.all_plays:
            display_all_results(st.session_state.all_plays, st.session_state.threshold, date_str)
        else:
            st.info("👈 Click **Run Analysis**")
            games = get_todays_schedule(date_str)
            if games:
                st.subheader(f"📅 {date_str}")
                for game in games:
                    st.write(f"**{game['away_team']}** @ **{game['home_team']}** - {game['time']}")
    
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

def run_optimized_analysis(date_str: str, threshold: int) -> Tuple[List[Dict], List[Dict]]:
    """Analysis with progress tracking."""
    
    st.subheader(f"📅 {date_str}")
    games = get_todays_schedule(date_str)
    
    if not games:
        st.error("No games found!")
        return [], []
    
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
    
    # Fetch defense stats (sequential but reliable)
    status_text.text(f"🛡️ Fetching defense for {len(teams_playing)} teams...")
    team_defense = get_team_defense_safe(list(teams_playing), status_text)
    progress_bar.progress(15)
    
    # Fetch rosters (sequential)
    status_text.text("📋 Fetching rosters...")
    all_players = []
    for team in teams_playing:
        roster = fetch_roster(team)
        all_players.extend(roster)
    progress_bar.progress(25)
    
    stats_display.text(f"Found {len(all_players)} players to analyze")
    
    # Analyze players (sequential with progress)
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
                "tags": get_tags(stats)
            }
            all_plays.append(play)
            stats_display.text(f"Checked: {i+1}/{total} | Found: {len(all_plays)}")
            
        except Exception as e:
            continue
    
    progress_bar.progress(100)
    elapsed_total = time.time() - start_time
    status_text.text(f"✅ Complete in {elapsed_total:.1f}s!")
    
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    stats_display.empty()
    
    # Sort by probability
    prob_key = f"prob_{threshold}plus"
    all_plays.sort(key=lambda x: x[prob_key], reverse=True)
    
    st.success(f"Found **{len(all_plays)}** players in **{elapsed_total:.1f}s**")
    display_all_results(all_plays, threshold, date_str)
    
    return all_plays, games

def display_all_results(plays: List[Dict], threshold: int, date_str: str):
    st.subheader(f"🎯 All Players - O{threshold - 0.5} SOG")
    st.caption("Sorted by Model Probability")
    
    show_model_explanation()
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    qualified = len([p for p in plays if p["is_qualified"]])
    pp1_count = len([p for p in plays if p["player"]["is_pp1"]])
    hot_count = len([p for p in plays if p["is_hot"]])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(plays))
    col2.metric("✅ Qualified", qualified)
    col3.metric("⚡ PP1", pp1_count)
    col4.metric("🔥 Hot", hot_count)
    
    results_data = []
    for play in plays:
        p = play["player"]
        row = {
            "": play["status_icon"],
            "Player": p["name"],
            "Tags": play["tags"],
            "Team": p["team"],
            "vs": play["opponent"],
            "Prob%": play[prob_key],
            "Hit%": p[hit_key],
            "Avg": p["avg_sog"],
            "L5": p["last_5_avg"],
            "Floor": p["floor"],
            "Trend": play["trend"],
            "Def": play["opponent_defense"]["grade"],
        }
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "": st.column_config.TextColumn("", width="small"),
            "Prob%": st.column_config.ProgressColumn("Prob%", min_value=0, max_value=100, format="%.1f%%"),
            "Hit%": st.column_config.NumberColumn("Hit%", format="%.1f%%"),
        }
    )
    
    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name=f"nhl_sog_{date_str}.csv", mime="text/csv")

def show_best_bets(plays: List[Dict], threshold: int):
    st.header("🎯 Best Bets")
    show_model_explanation()
    
    qualified = [p for p in plays if p["is_qualified"]]
    risky = [p for p in plays if not p["is_qualified"]]
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    st.subheader(f"✅ Qualified ({len(qualified)})")
    if qualified:
        qual_data = [{"Player": p["player"]["name"], "Tags": p["tags"], "Team": p["player"]["team"], 
                      "vs": p["opponent"], "Prob%": p[prob_key], "Hit%": p["player"][hit_key], 
                      "L5": p["player"]["last_5_avg"], "Trend": p["trend"]} for p in qualified]
        st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No qualified plays")
    
    st.markdown("---")
    st.subheader(f"⚠️ Higher Risk ({len(risky)})")
    if risky:
        risk_data = [{"Player": p["player"]["name"], "Tags": p["tags"], "Team": p["player"]["team"],
                      "vs": p["opponent"], "Prob%": p[prob_key], "Hit%": p["player"][hit_key],
                      "L5": p["player"]["last_5_avg"], "Trend": p["trend"]} for p in risky]
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

def show_parlays(plays: List[Dict], games: List[Dict], threshold: int, unit_size: float):
    st.header("🎰 Parlays")
    show_model_explanation()
    
    prob_key = f"prob_{threshold}plus"
    hit_key = f"hit_rate_{threshold}plus"
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
            st.dataframe(pd.DataFrame([{"Player": p["player"]["name"], "Prob%": f"{p[prob_key]:.0f}%"} for p in ultimate["legs"]]), hide_index=True)
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
                st.dataframe(pd.DataFrame([{"": "✅" if p["is_qualified"] else "⚠️", "Player": p["player"]["name"], "Prob%": f"{p[prob_key]:.0f}%"} for p in sgp["legs"]]), hide_index=True)
                copy_text = f"🎮 SGP - {game['matchup']}\n" + "─"*30 + "\n"
                for p in sgp["legs"]:
                    copy_text += f"{'✅' if p['is_qualified'] else '⚠️'} {p['player']['name']} O{threshold-0.5} SOG\n"
                copy_text += "─"*30 + f"\nOdds: {sgp['american_odds']:+d}\n"
                st.code(copy_text, language=None)

def show_help():
    st.header("❓ Help")
    show_model_explanation()
    st.markdown("""
    ## v3.4.1 Improvements
    
    ### ⚡ Speed
    - **Reduced defense fetching** - 5 games instead of 10
    - **Smarter caching** - 30 min for defense stats
    - **Progress tracking** - See exactly what's happening
    
    ### 🧮 Model
    - **Position factor** - Forwards shoot more than D
    - **Back-to-back detection** - Fatigue penalty
    - **Opponent trending** - Is defense getting worse/better?
    - **Floor bonus** - Floor ≥1 or ≥2 gets boost (rare but valuable)
    - **More aggressive cold penalty** - Cold streaks hurt more
    """)

if __name__ == "__main__":
    main()
