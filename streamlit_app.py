#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer v4.0
===============================
- SAFE%: % of games with threshold+1 shots (buffer for stat corrections)
- DEFENSE: 20 games, proper A-F grading, trend detection
- AUTO-TRACK: Picks auto-saved, one-click results check
- DEBUG: Player ID string conversion, extensive matching debug
- LEGEND: Tags/symbols reference with Safe% explanation
- NO ODDS API: Uses model + hit rate only
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

# Defense analysis
DEFENSE_GAMES = 20  # More games = more accurate

# Defense grades based on Shots Allowed per Game
# Higher SA/G = easier matchup for shooters (good)
# Lower SA/G = tougher matchup (bad)
# League average is ~28-30 SA/G
MATCHUP_GRADES = {
    "A+": 34.0,  # 34+ SA/G - very easy
    "A": 32.0,   # 32-33.9 - easy
    "B": 30.0,   # 30-31.9 - above average
    "C": 28.0,   # 28-29.9 - average
    "D": 26.0,   # 26-27.9 - tough
    "F": 0.0,    # <26 - very tough
}

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
    "pp1_boost": 1.15,
    "home_boost": 1.03,
    "away_penalty": 0.97,
    "hot_streak_boost": 1.06,
    "cold_streak_penalty": 0.90,
    "forward_boost": 1.02,
    "defense_penalty": 0.96,
    "high_floor_1_boost": 1.04,
    "high_floor_2_boost": 1.06,
    "b2b_penalty": 0.94,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_datetime():
    return datetime.now(EST)

def get_est_date():
    return get_est_datetime().strftime("%Y-%m-%d")

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
    if player.get("is_pp1"): tags.append("⚡")
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
        score_emoji = get_score_color(p.get("parlay_score", 0))
        text += f"{score_emoji} {player['name']} ({player['team']})\n"
        text += f"   O{threshold-0.5} SOG | {player[f'hit_rate_{threshold}plus']:.0f}% hit | Score: {p.get('parlay_score', 0):.0f}\n"
    text += "─" * 30 + f"\nProb: {prob*100:.0f}% | Odds: {odds:+d}\n"
    return text

def get_score_color(score: int) -> str:
    if score >= 85: return "🟢"
    if score >= 75: return "🔵"
    if score >= 65: return "🟡"
    if score >= 55: return "🟠"
    return "🔴"

def get_grade_from_score(score: int) -> str:
    if score >= 85: return "A+"
    if score >= 75: return "A"
    if score >= 65: return "B+"
    if score >= 55: return "B"
    if score >= 45: return "C"
    return "D"

# ============================================================================
# PARLAY SCORE CALCULATION
# ============================================================================
def calculate_parlay_score(player: Dict, opp_def: Dict, is_home: bool, threshold: int, is_hot: bool, is_cold: bool) -> int:
    """
    Calculate a 0-100 parlay score based on reliability factors.
    
    Factors (research-backed):
    - Consistency (safe clear rate) = most important for parlays
    - Floor (never busts) = critical safety
    - Hit rate = historical success
    - Volume = more room for error
    - PP1 = guaranteed extra opportunities
    - Matchup = opponent defense quality
    - Trend = current form
    """
    score = 50  # Base score
    
    hit_rate = player[f"hit_rate_{threshold}plus"]
    
    # Hit rate bonus (up to 25 points) - most important
    if hit_rate >= 95: score += 25
    elif hit_rate >= 90: score += 20
    elif hit_rate >= 85: score += 15
    elif hit_rate >= 80: score += 10
    elif hit_rate >= 75: score += 5
    
    # Safe clear bonus (threshold + 1 shots = buffer against stat corrections)
    # For O1.5: 3+ is safe, For O2.5: 4+ is safe, For O3.5: 5+ is safe
    if threshold == 2:
        safe_clear = player.get("hit_rate_3plus", 0)
    elif threshold == 3:
        safe_clear = player.get("hit_rate_4plus", 0)
    else:  # threshold == 4
        safe_clear = player.get("hit_rate_5plus", 0)
    
    # Higher safe clear = better (safely clears with buffer)
    if safe_clear >= 70: score += 10    # Excellent - usually clears by 2+
    elif safe_clear >= 60: score += 7   # Good buffer
    elif safe_clear >= 50: score += 5   # Decent buffer
    elif safe_clear >= 40: score += 2   # Some buffer
    # < 40% = no bonus (thin margins)
    
    # Floor bonus (up to 10 points)
    # Floor >= 1 means player NEVER gets shutout
    if player["floor"] >= 2: score += 10
    elif player["floor"] >= 1: score += 7
    
    # Volume bonus - high average (up to 10 points)
    # Higher volume = more margin for error
    avg = player["avg_sog"]
    if avg >= 4.0: score += 10
    elif avg >= 3.5: score += 8
    elif avg >= 3.0: score += 5
    elif avg >= 2.5: score += 3
    
    # PP1 bonus (8 points)
    # Power play = guaranteed extra shot opportunities
    if player.get("is_pp1"): score += 8
    
    # Matchup bonus (up to 10 points)
    # A+/A = soft defense (allows lots of shots) = good for shooters
    # D/F = tough defense = bad for shooters
    opp_grade = opp_def.get("grade", "C")
    if opp_grade == "A+": score += 10
    elif opp_grade == "A": score += 7
    elif opp_grade == "B": score += 4
    elif opp_grade == "C": score += 0  # Average, no bonus
    elif opp_grade == "D": score -= 5
    elif opp_grade == "F": score -= 8
    
    # Defense trend bonus/penalty
    opp_trend = opp_def.get("trend", "stable")
    if opp_trend == "loosening": score += 4  # Defense getting worse = good
    elif opp_trend == "tightening": score -= 4  # Defense improving = bad
    
    # Trend bonus/penalty
    if is_hot: score += 5
    elif is_cold: score -= 8
    
    # Home bonus (small)
    if is_home: score += 2
    
    # B2B penalty
    if player.get("is_b2b"): score -= 3
    
    return max(0, min(100, score))

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
    
    pp_factor = MODEL_WEIGHTS["pp1_boost"] if player.get("is_pp1") else 1.0
    ha_factor = MODEL_WEIGHTS["home_boost"] if is_home else MODEL_WEIGHTS["away_penalty"]
    
    opp_sa = opp_def.get("shots_allowed_per_game", 30.0)
    opp_factor = opp_sa / 30.0
    
    # Defense trend adjustment
    opp_trend = opp_def.get("trend", "stable")
    if opp_trend == "loosening":
        opp_factor *= 1.04  # Defense getting worse = more shots allowed
    elif opp_trend == "tightening":
        opp_factor *= 0.96  # Defense getting better = fewer shots allowed
    
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
# NHL API FUNCTIONS (OPTIMIZED FOR SPEED)
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
                    
                    if not away or not home: continue
                    
                    try:
                        utc_dt = datetime.fromisoformat(game.get("startTimeUTC", "").replace("Z", "+00:00"))
                        time_str = utc_dt.astimezone(EST).strftime("%I:%M %p")
                    except:
                        time_str = "TBD"
                    
                    games.append({"id": game_id, "time": time_str, "away_team": away, "home_team": home, "matchup": f"{away} @ {home}"})
        return games
    except:
        return []

@st.cache_data(ttl=3600)
def get_team_defense_cached(team_abbrev: str) -> Dict:
    """Fetch defense stats with 1-hour cache. Uses 20 games for accuracy."""
    try:
        url = f"{NHL_WEB_API}/club-schedule-season/{team_abbrev}/{SEASON}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        
        completed = [g for g in resp.json().get("games", []) 
                    if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
        
        if not completed:
            return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable", "games_analyzed": 0}
        
        recent = completed[-DEFENSE_GAMES:]  # Last 20 games
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
                    if sa > 0: sa_list.append(sa)
            except:
                continue
        
        if not sa_list:
            return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable", "games_analyzed": 0}
        
        sa_pg = statistics.mean(sa_list)
        
        # Trend detection: compare older half vs recent half
        # sa_list is ordered oldest to newest, so:
        # first_half = older games, second_half = recent games
        trend = "stable"
        if len(sa_list) >= 10:
            mid = len(sa_list) // 2
            older_half = statistics.mean(sa_list[:mid])  # Older games
            recent_half = statistics.mean(sa_list[mid:])  # Recent games
            
            diff = recent_half - older_half
            if diff >= 2.5:
                trend = "loosening"  # Allowing MORE shots recently = easier
            elif diff <= -2.5:
                trend = "tightening"  # Allowing FEWER shots recently = harder
        
        return {
            "team_abbrev": team_abbrev, 
            "shots_allowed_per_game": round(sa_pg, 2), 
            "grade": get_grade(sa_pg),
            "trend": trend,
            "games_analyzed": len(sa_list)
        }
    except:
        return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable", "games_analyzed": 0}

@st.cache_data(ttl=1800)
def get_team_roster_cached(team_abbrev: str) -> List[Dict]:
    """Fetch roster with 30-min cache."""
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

def fetch_player_stats_fast(player_info: Dict) -> Optional[Dict]:
    """Fetch player stats - single API call, no landing page."""
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
        pp_goals = 0
        
        for game in games:
            shots = max(0, game.get("shots", 0))
            all_shots.append(shots)
            game_dates.append(game.get("gameDate", ""))
            pp_goals += game.get("powerPlayGoals", 0)
            
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
        
        # Safe clear rates (with 1-shot buffer for stat corrections)
        # For O1.5: 3+ is safe, For O2.5: 4+ is safe, For O3.5: 5+ is safe
        hit_5 = sum(1 for s in all_shots if s >= 5) / gp * 100
        
        # Estimate PP1 from PP goals and volume
        is_pp1 = (pp_goals >= 3 and avg >= 2.5) or (pp_goals >= 5)
        
        # Back-to-back detection
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
            "hit_rate_5plus": round(hit_5, 1),
            "avg_sog": round(avg, 2),
            "last_5_avg": round(l5_avg, 2),
            "last_10_avg": round(l10_avg, 2),
            "std_dev": round(std, 2),
            "floor": min(all_shots),
            "ceiling": max(all_shots),
            "current_streak": streak,
            "home_avg": round(home_avg, 2),
            "away_avg": round(away_avg, 2),
            "is_pp1": is_pp1,
            "is_b2b": is_b2b,
            "pp_goals": pp_goals,
        }
    except:
        return None

# ============================================================================
# PARLAY GENERATION
# ============================================================================
def generate_best_parlay(plays: List[Dict], num_legs: int, threshold: int) -> Optional[Dict]:
    if len(plays) < num_legs:
        return None
    
    # Sort by parlay score (not just probability)
    sorted_plays = sorted(plays, key=lambda x: x.get("parlay_score", 0), reverse=True)
    best_legs = sorted_plays[:num_legs]
    
    prob_key = f"prob_{threshold}plus"
    probs = [p[prob_key] / 100 for p in best_legs]
    combined_prob, american_odds = calculate_parlay_odds(probs)
    avg_score = sum(p.get("parlay_score", 0) for p in best_legs) / num_legs
    
    return {
        "legs": best_legs, "num_legs": num_legs,
        "combined_prob": combined_prob,
        "american_odds": american_odds,
        "payout_per_100": calculate_parlay_payout(american_odds, 100),
        "avg_parlay_score": avg_score,
    }

def generate_sgp_for_game(plays: List[Dict], game_id: str, threshold: int, min_legs: int = 3, min_odds: int = 300) -> Optional[Dict]:
    game_plays = [p for p in plays if p["game_id"] == game_id]
    if len(game_plays) < min_legs:
        return None
    
    prob_key = f"prob_{threshold}plus"
    sorted_plays = sorted(game_plays, key=lambda x: x.get("parlay_score", 0), reverse=True)
    
    for num_legs in range(min_legs, min(len(sorted_plays) + 1, 10)):
        legs = sorted_plays[:num_legs]
        probs = [p[prob_key] / 100 for p in legs]
        combined_prob, american_odds = calculate_parlay_odds(probs)
        
        if american_odds >= min_odds:
            qualified_count = sum(1 for p in legs if p["is_qualified"])
            risky_count = num_legs - qualified_count
            avg_score = sum(p.get("parlay_score", 0) for p in legs) / num_legs
            return {
                "legs": legs, "num_legs": num_legs, "combined_prob": combined_prob,
                "american_odds": american_odds, "payout_per_100": calculate_parlay_payout(american_odds, 100),
                "game_id": game_id, "qualified_count": qualified_count, "risky_count": risky_count,
                "risk_level": "🟢" if risky_count == 0 else ("🟡" if risky_count <= 1 else "🔴"),
                "avg_parlay_score": avg_score,
            }
    
    if len(sorted_plays) >= min_legs:
        legs = sorted_plays[:min_legs]
        probs = [p[prob_key] / 100 for p in legs]
        combined_prob, american_odds = calculate_parlay_odds(probs)
        qualified_count = sum(1 for p in legs if p["is_qualified"])
        avg_score = sum(p.get("parlay_score", 0) for p in legs) / min_legs
        return {
            "legs": legs, "num_legs": min_legs, "combined_prob": combined_prob,
            "american_odds": american_odds, "payout_per_100": calculate_parlay_payout(american_odds, 100),
            "game_id": game_id, "qualified_count": qualified_count, "risky_count": min_legs - qualified_count,
            "risk_level": "⚪", "avg_parlay_score": avg_score,
        }
    return None

# ============================================================================
# UI COMPONENTS
# ============================================================================
def show_model_explanation():
    with st.expander("📖 Parlay Score Explained", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Score Components (0-100)
            
            | Factor | Max Points |
            |--------|------------|
            | Hit Rate (95%+) | +25 |
            | Low Bust Rate (<5%) | +10 |
            | Floor ≥ 2 | +10 |
            | Volume (4+ avg) | +10 |
            | Soft Matchup (A+) | +10 |
            | PP1 Player | +8 |
            | 🔥 Hot Trend | +5 |
            | Def Loosening 📈 | +4 |
            | Home | +2 |
            
            ### Bust Rate (% games 0-1 SOG)
            | Bust% | Points | Risk |
            |-------|--------|------|
            | <5% | +10 | Elite |
            | 5-8% | +5 | Good |
            | 8-12% | 0 | Avg |
            | 12-18% | -5 | Risky |
            | >18% | -10 | Danger |
            """)
        with col2:
            st.markdown("""
            ### Penalties
            
            | Factor | Points |
            |--------|--------|
            | High Bust Rate (>18%) | -10 |
            | ❄️ Cold Trend | -8 |
            | Tough Matchup (F) | -8 |
            | Tough Matchup (D) | -5 |
            | Def Tightening 📉 | -4 |
            | Back-to-Back | -3 |
            
            ### Defense Grades (SA/G)
            | Grade | SA/G | Meaning |
            |-------|------|---------|
            | A+ | 34+ | Very easy |
            | A | 32-34 | Easy |
            | B | 30-32 | Above avg |
            | C | 28-30 | Average |
            | D | 26-28 | Tough |
            | F | <26 | Very tough |
            
            ### Defense Trend
            - 📈 = Loosening (allowing more)
            - 📉 = Tightening (allowing fewer)
            """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer")
    st.caption("v4.0 | Bust Rate + Auto-Track + Debug")
    
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 All Results", "🎯 Best Bets", "🎰 Parlays", "📈 Track Results", "❓ Help"])
    
    if 'all_plays' not in st.session_state:
        st.session_state.all_plays = []
    if 'games' not in st.session_state:
        st.session_state.games = []
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 2
    if 'tracking_data' not in st.session_state:
        st.session_state.tracking_data = []
    
    with tab1:
        if run_analysis:
            plays, games = run_fast_analysis(date_str, threshold)
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
                st.dataframe(pd.DataFrame([{"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]} for g in games]), use_container_width=True, hide_index=True)
    
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
        show_results_tracker(date_str, threshold)
    
    with tab5:
        show_help()

def run_fast_analysis(date_str: str, threshold: int) -> Tuple[List[Dict], List[Dict]]:
    """Optimized analysis - single API call per player."""
    
    games = get_todays_schedule(date_str)
    
    if not games:
        st.error("No games found!")
        return [], []
    
    st.subheader(f"📅 {len(games)} Games Today")
    st.dataframe(pd.DataFrame([{"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]} for g in games]), use_container_width=True, hide_index=True)
    
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
    
    # Fetch defense (cached)
    status_text.text(f"🛡️ Fetching defense ({len(teams_playing)} teams)...")
    team_defense = {}
    for team in teams_playing:
        team_defense[team] = get_team_defense_cached(team)
    progress_bar.progress(15)
    
    # Fetch rosters (cached)
    status_text.text("📋 Fetching rosters...")
    all_players = []
    for team in teams_playing:
        all_players.extend(get_team_roster_cached(team))
    progress_bar.progress(20)
    
    stats_display.text(f"Analyzing {len(all_players)} players...")
    
    # Analyze players
    all_plays = []
    total = len(all_players)
    
    for i, player_info in enumerate(all_players):
        pct = 20 + int((i / total) * 80)
        progress_bar.progress(pct)
        status_text.text(f"🔍 {player_info['name']} ({i+1}/{total})")
        
        try:
            stats = fetch_player_stats_fast(player_info)
            if not stats:
                continue
            
            hit_rate = stats["hit_rate_2plus"] if threshold == 2 else stats["hit_rate_3plus"] if threshold == 3 else stats["hit_rate_4plus"]
            if hit_rate < MIN_HIT_RATE:
                continue
            
            info = game_info.get(player_info["team"])
            if not info:
                continue
            
            opp = info["opponent"]
            opp_def = team_defense.get(opp, {"shots_allowed_per_game": 30.0, "grade": "C"})
            is_home = info["home_away"] == "HOME"
            
            prob_2 = calculate_model_probability(stats, opp_def, is_home, 2)
            prob_3 = calculate_model_probability(stats, opp_def, is_home, 3)
            prob_4 = calculate_model_probability(stats, opp_def, is_home, 4)
            
            trend, is_hot, is_cold = get_trend(stats["last_5_avg"], stats["avg_sog"])
            is_qualified, status_icon = get_status_icon(hit_rate, is_cold)
            
            parlay_score = calculate_parlay_score(stats, opp_def, is_home, threshold, is_hot, is_cold)
            parlay_grade = get_grade_from_score(parlay_score)
            
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
                "parlay_score": parlay_score,
                "parlay_grade": parlay_grade,
            }
            all_plays.append(play)
            stats_display.text(f"Checked: {i+1}/{total} | Found: {len(all_plays)}")
            
        except:
            continue
    
    progress_bar.progress(100)
    elapsed = time.time() - start_time
    status_text.text(f"✅ Complete in {elapsed:.1f}s!")
    
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    stats_display.empty()
    
    # Sort by parlay score
    all_plays.sort(key=lambda x: x["parlay_score"], reverse=True)
    
    st.success(f"Found **{len(all_plays)}** players in **{elapsed:.1f}s**")
    display_all_results(all_plays, threshold, date_str)
    
    return all_plays, games

def display_all_results(plays: List[Dict], threshold: int, date_str: str):
    st.subheader(f"🎯 All Players - O{threshold - 0.5} SOG")
    st.caption("Sorted by Parlay Score (best parlay legs first)")
    
    show_model_explanation()
    
    # Tags Legend
    with st.expander("🏷️ Tags & Symbols Legend", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Status Icons**
            | Icon | Meaning |
            |------|---------|
            | ✅ | Qualified (75%+ hit rate) |
            | ⚠️ | Not qualified / Cold |
            
            **Player Tags**
            | Tag | Meaning |
            |-----|---------|
            | ⚡ | PP1 player (power play) |
            | 🛡️ | Floor ≥1 (never 0 SOG) |
            | 🔥5G | 5+ game hit streak |
            | 🔥7G | 7+ game hit streak |
            | 🔥10G | 10+ game hit streak |
            | B2B | Back-to-back game |
            """)
        with col2:
            st.markdown(f"""
            **Key Columns**
            | Column | Meaning |
            |--------|---------|
            | Hit% | % games hitting O{threshold-0.5} |
            | Safe% | % games with {threshold+1}+ SOG |
            | Model% | Predicted prob today |
            
            **Safe% = Buffer Zone**
            
            For **O{threshold-0.5}**: Safe% shows how often they get **{threshold+1}+ SOG** (one extra shot as buffer for stat corrections).
            
            Higher Safe% = safer parlay leg.
            """)
        with col3:
            st.markdown("""
            **Defense Grade (SA/G)**
            | Grade | Shots Allowed |
            |-------|---------------|
            | A+ | 34+ (easy) |
            | A | 32-34 |
            | B | 30-32 |
            | C | 28-30 (avg) |
            | D | 26-28 |
            | F | <26 (tough) |
            
            **Defense Trend**
            | Icon | Meaning |
            |------|---------|
            | 📈 | Loosening (good) |
            | 📉 | Tightening (bad) |
            """)
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    # Safe clear key: threshold + 1 (e.g., O1.5 → 3+, O2.5 → 4+, O3.5 → 5+)
    safe_key = f"hit_rate_{threshold + 1}plus"
    
    a_plus = len([p for p in plays if p["parlay_grade"] == "A+"])
    a_grade = len([p for p in plays if p["parlay_grade"] == "A"])
    qualified = len([p for p in plays if p["is_qualified"]])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(plays))
    col2.metric("🟢 A+", a_plus)
    col3.metric("🔵 A", a_grade)
    col4.metric("✅ Qualified", qualified)
    
    results_data = []
    for play in plays:
        p = play["player"]
        score_emoji = get_score_color(play["parlay_score"])
        
        # Format defense with trend indicator
        opp_def = play["opponent_defense"]
        def_grade = opp_def.get("grade", "C")
        def_trend = opp_def.get("trend", "stable")
        if def_trend == "loosening":
            def_display = f"{def_grade}📈"  # Getting worse (good for us)
        elif def_trend == "tightening":
            def_display = f"{def_grade}📉"  # Getting better (bad for us)
        else:
            def_display = def_grade
        
        row = {
            "": play["status_icon"],
            "Player": p["name"],
            "Score": f"{score_emoji} {play['parlay_score']}",
            "Grade": play["parlay_grade"],
            "Tags": play["tags"],
            "Team": p["team"],
            "vs": play["opponent"],
            "Model%": play[prob_key],
            "Hit%": p[hit_key],
            "Avg": p["avg_sog"],
            "L5": p["last_5_avg"],
            "Safe%": p.get(safe_key, 0),  # % with buffer (threshold + 1)
            "Floor": p["floor"],
            "Trend": play["trend"],
            "Def": def_display,
        }
        results_data.append(row)
    
    st.dataframe(
        pd.DataFrame(results_data), 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "": st.column_config.TextColumn("", width="small"),
            "Model%": st.column_config.ProgressColumn("Model%", min_value=0, max_value=100, format="%.1f%%"),
            "Hit%": st.column_config.NumberColumn("Hit%", format="%.0f%%"),
            "Safe%": st.column_config.NumberColumn("Safe%", format="%.0f%%", help=f"% of games with {threshold+1}+ SOG (buffer)"),
        }
    )
    
    st.download_button("📥 Download CSV", data=pd.DataFrame(results_data).to_csv(index=False), file_name=f"nhl_sog_{date_str}.csv", mime="text/csv")

def show_best_bets(plays: List[Dict], threshold: int):
    st.header("🎯 Best Bets by Parlay Score")
    show_model_explanation()
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    safe_key = f"hit_rate_{threshold + 1}plus"
    
    elite = [p for p in plays if p["parlay_grade"] in ["A+", "A"]]
    good = [p for p in plays if p["parlay_grade"] in ["B+", "B"]]
    risky = [p for p in plays if p["parlay_grade"] in ["C", "D"]]
    
    st.subheader(f"🏆 Elite Picks ({len(elite)})")
    st.caption("A+ and A grade - best for parlays")
    
    if elite:
        elite_data = []
        for play in elite:
            p = play["player"]
            elite_data.append({
                "Player": p["name"],
                "Score": f"{get_score_color(play['parlay_score'])} {play['parlay_score']}",
                "Grade": play["parlay_grade"],
                "Tags": play["tags"],
                "Team": p["team"],
                "vs": play["opponent"],
                "Model%": f"{play[prob_key]:.1f}%",
                "Hit%": f"{p[hit_key]:.0f}%",
                "Safe%": f"{p.get(safe_key, 0):.0f}%",
                "Avg": p["avg_sog"],
                "Floor": p["floor"],
            })
        st.dataframe(pd.DataFrame(elite_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No elite picks today")
    
    st.markdown("---")
    
    st.subheader(f"👍 Good Picks ({len(good)})")
    st.caption("B+ and B grade")
    
    if good:
        good_data = [{"Player": p["player"]["name"], "Score": p["parlay_score"], "Grade": p["parlay_grade"],
                      "Tags": p["tags"], "vs": p["opponent"], "Model%": f"{p[prob_key]:.1f}%", 
                      "Hit%": f"{p['player'][hit_key]:.0f}%",
                      "Safe%": f"{p['player'].get(safe_key, 0):.0f}%"} for p in good]
        st.dataframe(pd.DataFrame(good_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader(f"⚠️ Risky ({len(risky)})")
    if risky:
        risky_data = [{"Player": p["player"]["name"], "Score": p["parlay_score"], "Grade": p["parlay_grade"],
                       "vs": p["opponent"], "Model%": f"{p[prob_key]:.1f}%",
                       "Safe%": f"{p['player'].get(safe_key, 0):.0f}%"} for p in risky]
        st.dataframe(pd.DataFrame(risky_data), use_container_width=True, hide_index=True)

def show_parlays(plays: List[Dict], games: List[Dict], threshold: int, unit_size: float):
    st.header("🎰 Parlays")
    show_model_explanation()
    
    prob_key = f"prob_{threshold}plus"
    sorted_plays = sorted(plays, key=lambda x: x.get("parlay_score", 0), reverse=True)
    
    st.success(f"Building from **{len(sorted_plays)}** players (sorted by Parlay Score)")
    
    # Best parlay by legs table
    st.subheader("📊 Best Parlay by Legs")
    st.caption("Uses highest Parlay Score players for each leg count")
    
    max_legs = min(12, len(sorted_plays))
    parlay_table = []
    parlays_dict = {}
    
    for num_legs in range(1, max_legs + 1):
        parlay = generate_best_parlay(sorted_plays, num_legs, threshold)
        if parlay:
            players = ", ".join([p["player"]["name"] for p in parlay["legs"]])
            avg_score = parlay.get("avg_parlay_score", 0)
            parlay_table.append({
                "Legs": num_legs,
                "Avg Score": f"{avg_score:.0f}",
                "Prob%": f"{parlay['combined_prob']*100:.1f}%",
                "Odds": f"{parlay['american_odds']:+d}" if parlay['american_odds'] < 10000 else "—",
                f"${unit_size:.0f}→": f"${parlay['payout_per_100'] * unit_size / 100:.0f}",
                "Players": players[:60] + "..." if len(players) > 60 else players
            })
            parlays_dict[num_legs] = parlay
    
    if parlay_table:
        st.dataframe(pd.DataFrame(parlay_table), use_container_width=True, hide_index=True)
    
    # Copy buttons
    st.markdown("### 📋 Click to Copy")
    cols = st.columns(4)
    for i, num_legs in enumerate([2, 3, 5, 10]):
        if num_legs in parlays_dict:
            parlay = parlays_dict[num_legs]
            with cols[i]:
                with st.expander(f"**{num_legs}-Leg** ({parlay['combined_prob']*100:.0f}%)"):
                    st.code(format_parlay_text(parlay["legs"], threshold, f"{num_legs}-Leg Parlay", parlay['combined_prob'], parlay['american_odds']), language=None)
    
    st.markdown("---")
    
    # SGPs
    st.subheader("🎮 Same Game Parlays")
    for game in games:
        sgp = generate_sgp_for_game(sorted_plays, game["id"], threshold)
        if sgp:
            avg_score = sgp.get("avg_parlay_score", 0)
            with st.expander(f"**{game['matchup']}** | {sgp['american_odds']:+d} | Avg Score: {avg_score:.0f} | {sgp['risk_level']}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Legs", sgp["num_legs"])
                col2.metric("Prob", f"{sgp['combined_prob']*100:.1f}%")
                col3.metric("Odds", f"{sgp['american_odds']:+d}")
                col4.metric(f"${unit_size:.0f}→", f"${sgp['payout_per_100'] * unit_size / 100:.0f}")
                
                sgp_data = [{"": "✅" if p["is_qualified"] else "⚠️", 
                            "Player": p["player"]["name"], 
                            "Score": f"{get_score_color(p['parlay_score'])} {p['parlay_score']}", 
                            "Model%": f"{p[prob_key]:.0f}%"} for p in sgp["legs"]]
                st.dataframe(pd.DataFrame(sgp_data), hide_index=True)
                
                copy_text = f"🎮 SGP - {game['matchup']}\n" + "─"*30 + "\n"
                for p in sgp["legs"]:
                    copy_text += f"{get_score_color(p['parlay_score'])} {p['player']['name']} O{threshold-0.5} SOG (Score: {p['parlay_score']})\n"
                copy_text += "─"*30 + f"\nOdds: {sgp['american_odds']:+d} | Avg Score: {avg_score:.0f}\n"
                st.code(copy_text, language=None)

def show_results_tracker(date_str: str, threshold: int):
    st.header("📈 Results Tracker")
    
    # Initialize session state for tracking
    if 'saved_picks' not in st.session_state:
        st.session_state.saved_picks = {}  # {date: [picks]}
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []  # Combined history
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ Today's Picks")
        
        # Auto-save picks when analysis runs
        if st.session_state.all_plays:
            current_date = date_str
            picks_for_date = []
            
            prob_key = f"prob_{st.session_state.threshold}plus"
            hit_key = f"hit_rate_{st.session_state.threshold}plus"
            
            for play in st.session_state.all_plays:
                p = play["player"]
                picks_for_date.append({
                    "date": current_date,
                    "player": p["name"],
                    "player_id": p["player_id"],
                    "team": p["team"],
                    "opponent": play["opponent"],
                    "threshold": st.session_state.threshold,
                    "parlay_score": play["parlay_score"],
                    "parlay_grade": play["parlay_grade"],
                    "model_prob": round(play[prob_key], 1),
                    "hit_rate": round(p[hit_key], 1),
                    "is_qualified": play["is_qualified"],
                    "actual_sog": None,
                    "hit": None
                })
            
            st.session_state.saved_picks[current_date] = picks_for_date
            st.success(f"✅ {len(picks_for_date)} picks saved for {current_date}")
            
            # Show summary
            qualified = [p for p in picks_for_date if p["is_qualified"]]
            st.metric("Qualified Picks", len(qualified))
            st.metric("Total Analyzed", len(picks_for_date))
        else:
            st.info("Run analysis to save picks")
    
    with col2:
        st.subheader("2️⃣ Check Results")
        
        # Select date to check
        saved_dates = list(st.session_state.saved_picks.keys())
        if saved_dates:
            check_date = st.selectbox("Select date to check", saved_dates, index=len(saved_dates)-1)
        else:
            check_date = st.date_input("Date", value=get_est_datetime().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        if st.button("🔍 Check Results", type="primary", use_container_width=True):
            with st.spinner("Fetching box scores..."):
                check_and_update_results(check_date, st.session_state.threshold)
    
    st.markdown("---")
    
    # Show results for selected date
    st.subheader("📊 Results Report")
    
    if saved_dates:
        report_date = st.selectbox("View report for", saved_dates, key="report_date")
        show_date_report(report_date, st.session_state.threshold)
    else:
        st.info("No picks saved yet. Run analysis first.")
    
    st.markdown("---")
    
    # Running totals
    st.subheader("📈 Running Totals")
    show_running_totals()
    
    st.markdown("---")
    
    # Import/Export
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Export History")
        if st.session_state.saved_picks:
            all_picks = []
            for date_picks in st.session_state.saved_picks.values():
                all_picks.extend(date_picks)
            
            if all_picks:
                df = pd.DataFrame(all_picks)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download All History",
                    data=csv,
                    file_name=f"nhl_sog_history_{get_est_date()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"{len(all_picks)} total picks across {len(st.session_state.saved_picks)} days")
    
    with col2:
        st.subheader("📤 Import History")
        uploaded = st.file_uploader("Upload previous history CSV", type="csv")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                # Group by date and restore to saved_picks
                for date in df["date"].unique():
                    date_df = df[df["date"] == date]
                    st.session_state.saved_picks[date] = date_df.to_dict("records")
                st.success(f"✅ Imported {len(df)} picks from {df['date'].nunique()} days")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing: {e}")


def check_and_update_results(check_date: str, threshold: int):
    """Fetch actual results and update saved picks."""
    
    if check_date not in st.session_state.saved_picks:
        st.warning(f"No picks saved for {check_date}")
        return
    
    picks = st.session_state.saved_picks[check_date]
    
    # Build lookups - convert player_id to STRING for consistent comparison
    pick_by_id = {str(p["player_id"]): p for p in picks}
    pick_by_name = {p["player"].lower().strip(): p for p in picks}
    
    # Debug: show what we're looking for
    debug_info = []
    debug_info.append(f"Looking for {len(picks)} players:")
    for p in picks[:5]:  # Show first 5
        debug_info.append(f"  - ID:{p['player_id']} Name:'{p['player']}'")
    if len(picks) > 5:
        debug_info.append(f"  ... and {len(picks)-5} more")
    
    # Fetch games for that date
    games = get_todays_schedule(check_date)
    
    if not games:
        st.error(f"No games found for {check_date}")
        return
    
    results_found = 0
    games_checked = 0
    games_finished = 0
    game_states = []
    boxscore_players = []  # Debug: track players found in boxscores
    
    for game in games:
        try:
            box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
            resp = requests.get(box_url, timeout=15)
            
            if resp.status_code != 200:
                game_states.append(f"{game['away_team']}@{game['home_team']}: API error {resp.status_code}")
                continue
            
            box_data = resp.json()
            games_checked += 1
            
            # Check game state
            game_state = box_data.get("gameState", "UNKNOWN")
            game_states.append(f"{game['away_team']}@{game['home_team']}: {game_state}")
            
            if game_state not in ["OFF", "FINAL"]:
                continue  # Game not finished
            
            games_finished += 1
            
            # Get player stats
            for team_key in ["homeTeam", "awayTeam"]:
                for player_type in ["forwards", "defense"]:
                    players = box_data.get("boxscore", {}).get("playerByGameStats", {}).get(team_key, {}).get(player_type, [])
                    
                    for player in players:
                        pid = player.get("playerId")
                        pid_str = str(pid) if pid else ""
                        
                        # Get player name from boxscore
                        name_data = player.get("name", {})
                        if isinstance(name_data, dict):
                            player_name = name_data.get("default", "")
                        else:
                            player_name = str(name_data)
                        
                        actual_sog = player.get("sog", 0)
                        
                        # Track for debug
                        boxscore_players.append(f"ID:{pid} Name:'{player_name}' SOG:{actual_sog}")
                        
                        # Try to match by ID first (as string)
                        matched_pick = None
                        match_method = ""
                        
                        if pid_str and pid_str in pick_by_id:
                            matched_pick = pick_by_id[pid_str]
                            match_method = "ID"
                        # Fallback to name matching
                        elif player_name and player_name.lower().strip() in pick_by_name:
                            matched_pick = pick_by_name[player_name.lower().strip()]
                            match_method = "name"
                        
                        if matched_pick:
                            matched_pick["actual_sog"] = actual_sog
                            matched_pick["hit"] = 1 if actual_sog >= threshold else 0
                            results_found += 1
                            debug_info.append(f"✅ Matched ({match_method}): {player_name} = {actual_sog} SOG")
            
            time.sleep(0.05)
            
        except Exception as e:
            game_states.append(f"{game['away_team']}@{game['home_team']}: Error - {e}")
            continue
    
    # Update session state
    st.session_state.saved_picks[check_date] = picks
    
    # Show game status
    with st.expander("🔍 Game Status Details"):
        for gs in game_states:
            st.text(gs)
    
    # Show debug info
    with st.expander("🐛 Debug: Matching Details"):
        st.text("\n".join(debug_info))
        st.markdown("---")
        st.text(f"Boxscore players found (first 20 of {len(boxscore_players)}):")
        for bp in boxscore_players[:20]:
            st.text(f"  {bp}")
    
    if games_finished == 0:
        st.warning(f"⏳ No finished games found. {games_checked} games checked - may still be in progress.")
    elif results_found == 0:
        st.error(f"⚠️ {games_finished} games finished but no picks matched. Check debug info above.")
    else:
        st.success(f"✅ Updated {results_found} picks from {games_finished} finished games")


def show_date_report(report_date: str, threshold: int):
    """Show detailed report for a specific date."""
    
    if report_date not in st.session_state.saved_picks:
        st.info("No data for this date")
        return
    
    picks = st.session_state.saved_picks[report_date]
    
    # Check if we have results
    has_results = any(p.get("actual_sog") is not None for p in picks)
    
    if not has_results:
        st.warning("⏳ Results not checked yet. Click 'Check Results' above.")
        # Still show picks without results
        df = pd.DataFrame(picks)[["player", "team", "opponent", "parlay_grade", "model_prob", "is_qualified"]]
        df["is_qualified"] = df["is_qualified"].apply(lambda x: "✅" if x else "")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    
    # Split qualified vs non-qualified
    qualified = [p for p in picks if p.get("is_qualified")]
    non_qualified = [p for p in picks if not p.get("is_qualified")]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    q_with_results = [p for p in qualified if p.get("hit") is not None]
    nq_with_results = [p for p in non_qualified if p.get("hit") is not None]
    
    q_hits = sum(p.get("hit", 0) for p in q_with_results)
    q_total = len(q_with_results)
    nq_hits = sum(p.get("hit", 0) for p in nq_with_results)
    nq_total = len(nq_with_results)
    
    q_rate = (q_hits / q_total * 100) if q_total > 0 else 0
    nq_rate = (nq_hits / nq_total * 100) if nq_total > 0 else 0
    
    col1.metric("✅ Qualified Hits", f"{q_hits}/{q_total}", f"{q_rate:.0f}%")
    col2.metric("⚠️ Non-Qual Hits", f"{nq_hits}/{nq_total}", f"{nq_rate:.0f}%")
    col3.metric("Total Hits", f"{q_hits + nq_hits}/{q_total + nq_total}")
    col4.metric("Threshold", f"O{threshold - 0.5} ({threshold}+ SOG)")
    
    # Qualified picks table
    st.markdown("### ✅ Qualified Picks")
    if q_with_results:
        q_df = pd.DataFrame(q_with_results)
        q_df = q_df[["player", "team", "opponent", "parlay_grade", "parlay_score", "model_prob", "actual_sog", "hit"]]
        q_df["hit"] = q_df["hit"].apply(lambda x: "✅ HIT" if x == 1 else "❌ MISS" if x == 0 else "⏳")
        q_df = q_df.sort_values("parlay_score", ascending=False)
        st.dataframe(q_df, use_container_width=True, hide_index=True)
    else:
        st.info("No qualified picks with results")
    
    # Non-qualified picks table (collapsed)
    with st.expander(f"⚠️ Non-Qualified Picks ({nq_total} players)"):
        if nq_with_results:
            nq_df = pd.DataFrame(nq_with_results)
            nq_df = nq_df[["player", "team", "opponent", "parlay_grade", "parlay_score", "model_prob", "actual_sog", "hit"]]
            nq_df["hit"] = nq_df["hit"].apply(lambda x: "✅ HIT" if x == 1 else "❌ MISS" if x == 0 else "⏳")
            nq_df = nq_df.sort_values("parlay_score", ascending=False)
            st.dataframe(nq_df, use_container_width=True, hide_index=True)
    
    # Grade breakdown
    st.markdown("### 📊 Performance by Grade")
    grade_stats = []
    for grade in ["A+", "A", "B+", "B", "C", "D"]:
        grade_picks = [p for p in picks if p.get("parlay_grade") == grade and p.get("hit") is not None]
        if grade_picks:
            hits = sum(p.get("hit", 0) for p in grade_picks)
            total = len(grade_picks)
            rate = hits / total * 100 if total > 0 else 0
            grade_stats.append({
                "Grade": grade,
                "Hits": hits,
                "Total": total,
                "Hit Rate": f"{rate:.0f}%"
            })
    
    if grade_stats:
        st.dataframe(pd.DataFrame(grade_stats), use_container_width=True, hide_index=True)


def show_running_totals():
    """Show cumulative performance across all tracked days."""
    
    if not st.session_state.saved_picks:
        st.info("No history yet. Track some picks first!")
        return
    
    # Aggregate all picks
    all_picks = []
    for date_picks in st.session_state.saved_picks.values():
        all_picks.extend(date_picks)
    
    # Filter to picks with results
    with_results = [p for p in all_picks if p.get("hit") is not None]
    
    if not with_results:
        st.warning("No results checked yet. Check results for your saved dates.")
        st.caption(f"You have picks saved for: {', '.join(st.session_state.saved_picks.keys())}")
        return
    
    # Overall stats
    qualified = [p for p in with_results if p.get("is_qualified")]
    non_qualified = [p for p in with_results if not p.get("is_qualified")]
    
    q_hits = sum(p.get("hit", 0) for p in qualified)
    q_total = len(qualified)
    nq_hits = sum(p.get("hit", 0) for p in non_qualified)
    nq_total = len(non_qualified)
    
    total_hits = q_hits + nq_hits
    total_picks = q_total + nq_total
    
    st.markdown("### 🏆 All-Time Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    q_rate = (q_hits / q_total * 100) if q_total > 0 else 0
    nq_rate = (nq_hits / nq_total * 100) if nq_total > 0 else 0
    total_rate = (total_hits / total_picks * 100) if total_picks > 0 else 0
    
    col1.metric("✅ Qualified", f"{q_hits}/{q_total}", f"{q_rate:.1f}%")
    col2.metric("⚠️ Non-Qualified", f"{nq_hits}/{nq_total}", f"{nq_rate:.1f}%")
    col3.metric("📊 Overall", f"{total_hits}/{total_picks}", f"{total_rate:.1f}%")
    col4.metric("📅 Days Tracked", len(st.session_state.saved_picks))
    
    # Performance by grade (all-time)
    st.markdown("### 📊 All-Time by Grade")
    grade_stats = []
    for grade in ["A+", "A", "B+", "B", "C", "D"]:
        grade_picks = [p for p in with_results if p.get("parlay_grade") == grade]
        if grade_picks:
            hits = sum(p.get("hit", 0) for p in grade_picks)
            total = len(grade_picks)
            rate = hits / total * 100 if total > 0 else 0
            avg_prob = sum(p.get("model_prob", 0) for p in grade_picks) / total
            grade_stats.append({
                "Grade": grade,
                "Hits": hits,
                "Total": total,
                "Hit Rate": f"{rate:.1f}%",
                "Avg Model%": f"{avg_prob:.1f}%",
                "Calibration": f"{rate - avg_prob:+.1f}%" if total >= 5 else "N/A"
            })
    
    if grade_stats:
        st.dataframe(pd.DataFrame(grade_stats), use_container_width=True, hide_index=True)
        
        # Key insight
        a_plus = [p for p in with_results if p.get("parlay_grade") == "A+"]
        if len(a_plus) >= 3:
            a_plus_rate = sum(p.get("hit", 0) for p in a_plus) / len(a_plus) * 100
            if a_plus_rate >= 85:
                st.success(f"🏆 A+ grade is hitting at **{a_plus_rate:.0f}%** - Model working well!")
            elif a_plus_rate >= 75:
                st.info(f"📊 A+ grade is hitting at **{a_plus_rate:.0f}%** - Solid performance")
            else:
                st.warning(f"⚠️ A+ grade is only hitting at **{a_plus_rate:.0f}%** - May need to tighten criteria")
    
    # Daily breakdown
    with st.expander("📅 Daily Breakdown"):
        daily_stats = []
        for date, picks in st.session_state.saved_picks.items():
            picks_with_results = [p for p in picks if p.get("hit") is not None]
            if picks_with_results:
                q_picks = [p for p in picks_with_results if p.get("is_qualified")]
                q_hits = sum(p.get("hit", 0) for p in q_picks)
                total_hits = sum(p.get("hit", 0) for p in picks_with_results)
                daily_stats.append({
                    "Date": date,
                    "Qualified": f"{q_hits}/{len(q_picks)}" if q_picks else "0/0",
                    "Q Rate": f"{q_hits/len(q_picks)*100:.0f}%" if q_picks else "-",
                    "Total": f"{total_hits}/{len(picks_with_results)}",
                    "Total Rate": f"{total_hits/len(picks_with_results)*100:.0f}%"
                })
        
        if daily_stats:
            st.dataframe(pd.DataFrame(daily_stats), use_container_width=True, hide_index=True)


def show_help():
    st.header("❓ Help")
    show_model_explanation()
    
    st.markdown("""
    ## What is Parlay Score?
    
    A 0-100 score measuring how **reliable** a player is for parlays.
    
    **Higher score = better parlay leg** because:
    - High hit rate (historically delivers)
    - Low bust rate (rarely gets 0-1 SOG)
    - High floor (never gets completely shut out)
    - Good volume (averages lots of shots)
    
    ## What is Safe%?
    
    **Safe% = % of games with EXTRA shots (buffer for stat corrections)**
    
    | Threshold | Hit% Shows | Safe% Shows |
    |-----------|------------|-------------|
    | O1.5 | 2+ SOG | **3+ SOG** |
    | O2.5 | 3+ SOG | **4+ SOG** |
    | O3.5 | 4+ SOG | **5+ SOG** |
    
    ### Why Safe% Matters
    
    Stat corrections happen! If you bet O1.5 and a player gets **exactly 2 SOG**, one correction could make you lose.
    
    | Player | Hit% (2+) | Safe% (3+) | Risk |
    |--------|-----------|------------|------|
    | Player A | 85% | **70%** | Low - usually clears by 2+ |
    | Player B | 85% | **40%** | High - often exactly 2 |
    
    **Same hit rate, but Player A is much safer!**
    
    | Safe% | Meaning |
    |-------|---------|
    | 70%+ | Excellent - safely clears most games |
    | 60-70% | Good buffer |
    | 50-60% | Decent |
    | 40-50% | Thin margins |
    | <40% | Often just barely clears |
    
    ## Grade Meanings
    
    | Grade | What It Means |
    |-------|---------------|
    | **A+** | Lock it in - elite parlay leg |
    | **A** | Very reliable |
    | **B+** | Good, minor concerns |
    | **B** | Average, some risk |
    | **C** | Below average |
    | **D** | Risky - avoid in parlays |
    
    ## Tags
    
    | Tag | Meaning |
    |-----|---------|
    | 🛡️ | Floor ≥1 (never shutout) |
    | ⚡ | PP1 player |
    | 🔥5G | 5+ game hit streak |
    | B2B | Back-to-back game |
    
    ## Defense Column
    
    | Display | Meaning |
    |---------|---------|
    | A+ | Very easy defense (allows 34+ SA/G) |
    | A📈 | Easy + getting worse (loosening) |
    | C📉 | Average + getting better (tightening) |
    | F | Very tough (allows <26 SA/G) |
    
    ## 📈 Tracking Results
    
    ### Workflow (Much Simpler Now!)
    1. **Run Analysis** → Picks auto-saved
    2. **After games end** → Go to Track Results tab → Click "Check Results"
    3. **View report** → See hits/misses, grades performance
    4. **Export** → Download history CSV for backup
    
    ### What We Track
    - Hit rate by Parlay Grade (A+, A, B+, etc.)
    - Qualified vs Non-Qualified performance
    - Running totals across all days
    
    ### Improving the Model
    After 50+ tracked picks, look for:
    - **A+ grades hitting <85%** → Scoring is too generous
    - **B grades hitting >80%** → Can loosen criteria
    - **High bust% players still hitting** → Reduce bust penalty
    
    ## Recommended Strategy
    
    1. **2-3 leg parlays**: Use A+ and A grade players only
    2. **4-5 leg parlays**: Mix A+ with B+ players
    3. **6+ leg parlays**: Higher risk, use carefully
    4. **Track everything**: Export picks, check results, improve over time
    """)

if __name__ == "__main__":
    main()
