#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer V7.2
===============================
HIT RATE IS KING MODEL

KEY CHANGES FROM V7.0:
- Hit rate determines BASE score (not additive from zero)
- Modifiers adjust from base (centered at zero)
- Variance NOT penalized (upside is good, downside captured by shutout%)
- Updated tier thresholds (88/80/70/60)

SCORING PHILOSOPHY:
base_score = 50 + (hit_rate - 70) × 1.5
final_score = base + modifiers

"An 89% hitter starts at 78.5. Other factors fine-tune from there."

v7.2 - January 2026
"""

import streamlit as st
import requests
import time
import math
import json
import os
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from itertools import combinations
import pytz
import statistics

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NHL SOG Analyzer V7.2",
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
MIN_HIT_RATE = 70  # Lowered slightly since continuous scoring handles quality
EST = pytz.timezone('US/Eastern')
DATA_DIR = "nhl_sog_data"
HISTORY_FILE = f"{DATA_DIR}/results_history.json"

# League averages for baseline
LEAGUE_AVG_SAG = 30.0  # Shots allowed per game
LEAGUE_AVG_SOG = 2.8   # Shots on goal per player

# Logistic mapping coefficients (score → probability)
LOGISTIC_A = 0.09      # Steepness
LOGISTIC_B = 4.8       # Midpoint shift

# Tier thresholds (V7.2)
TIERS = {
    "🔒 LOCK": 88,
    "✅ STRONG": 80,
    "📊 SOLID": 70,
    "⚠️ RISKY": 60,
    "❌ AVOID": 0
}

# Kill switch thresholds (auto-exclude from parlays)
KILL_SWITCHES = {
    "min_score": 60,  # Updated for V7.2 thresholds
    "max_l5_shutouts": 1,  # More than this = kill
    "max_variance": 2.8,
    "max_toi_drop_pct": 25,
    "min_l5_hit_rate": 60,
}

# Correlation penalties
CORRELATION_PENALTIES = {
    "same_team": 0.94,
    "same_line": 0.90,
    "same_pp_unit": 0.92,
    "same_game": 0.98,
}

# ============================================================================
# DATA PERSISTENCE
# ============================================================================
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_history() -> Dict:
    ensure_data_dir()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history: Dict):
    ensure_data_dir()
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# ============================================================================
# SESSION STATE INIT
# ============================================================================
if 'plays' not in st.session_state:
    st.session_state.plays = []
if 'games' not in st.session_state:
    st.session_state.games = []
if 'saved_picks' not in st.session_state:
    st.session_state.saved_picks = {}
if 'results_history' not in st.session_state:
    st.session_state.results_history = load_history()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_datetime():
    return datetime.now(EST)

def get_est_date():
    return get_est_datetime().strftime("%Y-%m-%d")

def parse_toi(toi_str: str) -> float:
    """Parse TOI string (MM:SS) to minutes."""
    try:
        if not toi_str or toi_str == "0:00":
            return 0.0
        parts = toi_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0 if len(parts) > 1 else float(parts[0])
    except:
        return 0.0

def implied_prob_to_american(prob: float) -> int:
    if prob <= 0: return 10000
    if prob >= 1: return -10000
    if prob >= 0.5: return int(-100 * prob / (1 - prob))
    return int(100 * (1 - prob) / prob)

def poisson_prob_at_least(lam: float, k: int) -> float:
    if lam <= 0: return 0.0
    return 1 - sum((math.exp(-lam) * (lam ** i)) / math.factorial(i) for i in range(k))

def calculate_percentile(data: List[int], pct: int) -> float:
    """Calculate percentile of data."""
    if not data: return 0.0
    s = sorted(data)
    idx = max(0, min(int(len(s) * pct / 100), len(s) - 1))
    return float(s[idx])

def calculate_parlay_odds(probs: List[float]) -> Tuple[float, int]:
    combined = 1.0
    for p in probs: combined *= p
    return combined, implied_prob_to_american(combined)

def calculate_parlay_payout(odds: int, stake: float = 100) -> float:
    if odds > 0: return stake + (stake * odds / 100)
    return stake + (stake * 100 / abs(odds))

def get_tier_from_score(score: float) -> str:
    """Get tier badge from score."""
    if score >= TIERS["🔒 LOCK"]: return "🔒 LOCK"
    if score >= TIERS["✅ STRONG"]: return "✅ STRONG"
    if score >= TIERS["📊 SOLID"]: return "📊 SOLID"
    if score >= TIERS["⚠️ RISKY"]: return "⚠️ RISKY"
    return "❌ AVOID"

def get_score_color(score: float) -> str:
    """Get color emoji for score."""
    if score >= 88: return "🟢"
    if score >= 80: return "🔵"
    if score >= 70: return "🟡"
    if score >= 60: return "🟠"
    return "🔴"

def get_grade(sa_pg: float) -> str:
    """Get letter grade from SA/G."""
    if sa_pg >= 34.0: return "A+"
    if sa_pg >= 32.0: return "A"
    if sa_pg >= 30.0: return "B"
    if sa_pg >= 28.0: return "C"
    if sa_pg >= 26.0: return "D"
    return "F"

# ============================================================================
# CONTINUOUS LINEAR SCALING FUNCTIONS
# ============================================================================
def linear_scale(value: float, min_val: float, max_val: float, max_points: float, inverted: bool = False) -> float:
    """
    Scale a value linearly between min and max to 0-max_points.
    If inverted, lower values score higher (e.g., shutout rate, variance).
    """
    if max_val == min_val:
        return max_points / 2
    
    # Clamp value to range
    clamped = max(min_val, min(max_val, value))
    
    # Calculate proportion
    if inverted:
        proportion = (max_val - clamped) / (max_val - min_val)
    else:
        proportion = (clamped - min_val) / (max_val - min_val)
    
    return proportion * max_points

def dynamic_weight(deviation: float, max_deviation: float, base_weight: float, min_multiplier: float = 0.5) -> float:
    """
    Dynamically weight a factor based on how extreme the deviation is.
    Neutral values (low deviation) get reduced weight.
    Extreme values (high deviation) get full weight.
    """
    confidence = min(1.0, abs(deviation) / max_deviation)
    return base_weight * (min_multiplier + (1 - min_multiplier) * confidence)

def magnitude_scale(diff: float, max_diff: float, base_points: float, penalty_multiplier: float = 1.0) -> float:
    """
    Scale points based on magnitude of difference.
    Larger differences (positive or negative) have more impact.
    Negative differences can have a penalty multiplier (e.g., cold streaks hurt more than hot streaks help).
    """
    magnitude = min(abs(diff) / max_diff, 1.0)  # 0 to 1
    base_contribution = diff / max_diff * base_points
    
    # Scale by magnitude (extreme values get amplified)
    multiplier = 0.5 + 0.5 * magnitude
    
    if diff < 0:
        return base_contribution * multiplier * penalty_multiplier
    return base_contribution * multiplier

# ============================================================================
# V7: CONTINUOUS PARLAY SCORE CALCULATION
# ============================================================================
def calculate_parlay_score_v7(player: Dict, opp_def: Dict, is_home: bool, threshold: int) -> Tuple[float, List[str], List[str]]:
    """
    V7.2: HIT RATE IS KING scoring model.
    
    Base Score: 50 + (hit_rate - 70) × 1.5
    - 70% → 50, 80% → 65, 85% → 72.5, 89% → 78.5, 90% → 80, 95% → 87.5
    
    Modifiers (centered at 0, adjust from base):
    - Shutout Rate: +4 to -4 (captures downside risk)
    - SOG/60: -2 to +2
    - Volume: -2 to +2
    - Floor P10: -2 to +2
    - Matchup: -5 to +5
    - PP1: +3
    - Home/Away: +1 / -0.5
    - Streak: +1 to +3
    - Form: -2 to +2
    - TOI trend: -2 to +1
    
    NOTE: Variance (σ) NOT penalized - high σ from upside games is fine.
    Downside risk already captured by shutout_rate and P10 floor.
    """
    
    edges = []
    risks = []
    
    # Get key metrics
    hit_rate = player.get(f"hit_rate_{threshold}plus", 75)
    shutout_rate = player.get("shutout_rate", 10)
    std_dev = player.get("std_dev", 1.5)
    p10 = player.get("p10", 1)
    sog60 = player.get("sog_per_60", 7)
    avg_sog = player.get("avg_sog", 2.5)
    cushion = player.get(f"cushion_{threshold}", 50)
    l5_avg = player.get("last_5_avg", avg_sog)
    l10_avg = player.get("last_10_avg", avg_sog)
    season_avg = player.get("avg_sog", avg_sog)
    l5_toi = player.get("l5_toi", 15)
    avg_toi = player.get("avg_toi", 15)
    games_played = player.get("games_played", 20)
    l5_shutouts = player.get("l5_shutouts", 0)
    streak = player.get("current_streak", 0)
    
    # ========== BASE SCORE FROM HIT RATE ==========
    # 70% → 50, 80% → 65, 85% → 72.5, 89% → 78.5, 90% → 80
    if hit_rate < 70:
        base_score = 50 + (hit_rate - 70) * 1.0  # Penalty below 70%
    else:
        base_score = 50 + (hit_rate - 70) * 1.5
    
    if hit_rate >= 90:
        edges.append(f"🎯 Elite {hit_rate:.0f}% hit")
    elif hit_rate >= 85:
        edges.append(f"Strong {hit_rate:.0f}% hit")
    elif hit_rate < 75:
        risks.append(f"⚠️ Low {hit_rate:.0f}% hit")
    
    # ========== MODIFIERS (centered at 0) ==========
    
    # Shutout Rate: 0% = +4, 10% = 0, 20% = -4
    # This captures DOWNSIDE risk
    shutout_mod = linear_scale(shutout_rate, 0, 20, 8, inverted=True) - 4
    
    if shutout_rate >= 18:
        risks.append(f"🚨 High shutout ({shutout_rate:.0f}%)")
    elif shutout_rate <= 5:
        edges.append(f"Rare shutouts ({shutout_rate:.0f}%)")
    
    # SOG/60: 6 = -2, 9 = 0, 12 = +2
    sog60_mod = linear_scale(sog60, 6, 12, 4) - 2
    
    if sog60 >= 11.0:
        edges.append(f"⚡ Elite rate ({sog60:.1f}/60)")
    elif sog60 < 7.0:
        risks.append(f"Low rate ({sog60:.1f}/60)")
    
    # Volume: 2.0 = -2, 3.25 = 0, 4.5 = +2
    vol_mod = linear_scale(avg_sog, 2.0, 4.5, 4) - 2
    
    if avg_sog >= 4.0:
        edges.append(f"High volume ({avg_sog:.1f})")
    
    # Floor P10: 0 = -2, 1 = 0, 2 = +2
    floor_mod = linear_scale(p10, 0, 2, 4) - 2
    
    if p10 >= 2:
        edges.append(f"🛡️ Strong floor (P10={p10})")
    elif p10 == 0:
        risks.append("⚠️ Floor=0 risk")
    
    # ========== MATCHUP ==========
    opp_grade = opp_def.get("grade", "C")
    opp_trend = opp_def.get("trend", "stable")
    opp_sag = opp_def.get("shots_allowed_per_game", 30.0)
    
    matchup_mods = {"A+": 5, "A": 4, "B": 2, "C": 0, "D": -3, "F": -5}
    matchup_mod = matchup_mods.get(opp_grade, 0)
    
    if opp_grade in ["A+", "A"]:
        edges.append(f"Soft matchup ({opp_grade})")
    elif opp_grade in ["D", "F"]:
        risks.append(f"Tough matchup ({opp_grade})")
    
    # Defense trend
    trend_mod = 0
    if opp_trend == "loosening":
        trend_mod = 1
        edges.append("Def loosening 📈")
    elif opp_trend == "tightening":
        trend_mod = -1
        risks.append("Def tightening 📉")
    
    # ========== SITUATIONAL ==========
    
    # PP1: +3
    pp_mod = 0
    if player.get("is_pp1"):
        pp_mod = 3
        edges.append("⚡ PP1")
    elif player.get("pp_goals", 0) >= 2:
        pp_mod = 1.5
    
    # Home/Away
    venue_mod = 1 if is_home else -0.5
    
    # Streak bonus
    streak_mod = 0
    if streak >= 10:
        streak_mod = 3
    elif streak >= 7:
        streak_mod = 2
    elif streak >= 5:
        streak_mod = 1
    
    # Form (L5 vs season)
    form_diff = l5_avg - season_avg
    form_mod = 0
    if form_diff >= 0.8:
        form_mod = 2
        edges.append(f"🔥 Hot (L5: {l5_avg:.1f})")
    elif form_diff >= 0.4:
        form_mod = 1
    elif form_diff <= -0.8:
        form_mod = -2
        risks.append(f"❄️ Cold (L5: {l5_avg:.1f})")
    elif form_diff <= -0.4:
        form_mod = -1
    
    # TOI trend
    toi_mod = 0
    if avg_toi > 0:
        toi_pct = (l5_toi - avg_toi) / avg_toi * 100
        if toi_pct >= 10:
            toi_mod = 1
        elif toi_pct <= -10:
            toi_mod = -2
            risks.append(f"⏱️ TOI down ({toi_pct:.0f}%)")
    
    # B2B penalty
    b2b_mod = 0
    if player.get("is_b2b"):
        b2b_mod = -2
        risks.append("B2B")
    
    # ========== PENALTIES ==========
    
    # Recent shutouts in L5
    l5_penalty = 0
    if l5_shutouts >= 2:
        l5_penalty = -10
        risks.append(f"🚨 {l5_shutouts} shutouts in L5!")
    elif l5_shutouts == 1:
        l5_penalty = -4
        risks.append("⚠️ Shutout in L5")
    
    # Small sample
    sample_penalty = 0
    if games_played < 10:
        sample_penalty = -5
        risks.append(f"⚠️ Small sample (n={games_played})")
    elif games_played < 15:
        sample_penalty = -2
    
    # ========== FINAL SCORE ==========
    modifiers = (shutout_mod + sog60_mod + vol_mod + floor_mod +
                 matchup_mod + trend_mod + pp_mod + venue_mod + 
                 streak_mod + form_mod + toi_mod + b2b_mod +
                 l5_penalty + sample_penalty)
    
    final_score = base_score + modifiers
    final_score = max(0, min(100, final_score))
    
    return final_score, edges, risks

# ============================================================================
# V7: LOGISTIC PROBABILITY MAPPING
# ============================================================================
def score_to_probability(score: float) -> float:
    """
    Convert parlay score to true probability using logistic function.
    
    P(hit) = 1 / (1 + e^(-(a × score - b)))
    
    Calibrated for Over 1.5 SOG props.
    """
    raw_prob = 1 / (1 + math.exp(-(LOGISTIC_A * score - LOGISTIC_B)))
    
    # Cap at 94% to prevent overconfidence, floor at 45%
    return min(0.94, max(0.45, raw_prob))

# ============================================================================
# V7: KILL SWITCH CHECK
# ============================================================================
def check_kill_switches(player: Dict, score: float, threshold: int) -> Tuple[bool, str]:
    """
    Check if player should be excluded from parlays.
    Returns (should_kill, reason).
    """
    
    # Score too low
    if score < KILL_SWITCHES["min_score"]:
        return True, f"Score < {KILL_SWITCHES['min_score']}"
    
    # Too many recent shutouts
    l5_shutouts = player.get("l5_shutouts", 0)
    if l5_shutouts > KILL_SWITCHES["max_l5_shutouts"]:
        return True, f"{l5_shutouts} shutouts in L5"
    
    # Extreme variance
    std_dev = player.get("std_dev", 1.5)
    if std_dev > KILL_SWITCHES["max_variance"]:
        return True, f"σ = {std_dev:.1f}"
    
    # TOI dropping significantly
    l5_toi = player.get("l5_toi", 15)
    avg_toi = player.get("avg_toi", 15)
    if avg_toi > 0:
        toi_drop = (avg_toi - l5_toi) / avg_toi * 100
        if toi_drop > KILL_SWITCHES["max_toi_drop_pct"]:
            return True, f"TOI -{toi_drop:.0f}%"
    
    # L5 hit rate too low
    l5_hit = calculate_l5_hit_rate(player, threshold)
    if l5_hit < KILL_SWITCHES["min_l5_hit_rate"]:
        return True, f"L5 hit = {l5_hit:.0f}%"
    
    return False, ""

def calculate_l5_hit_rate(player: Dict, threshold: int) -> float:
    """Calculate hit rate for last 5 games (estimated from averages)."""
    l5_avg = player.get("last_5_avg", 2.5)
    # Rough estimate: if L5 avg >= threshold + 0.5, likely ~80%+ hit rate
    # This is a heuristic since we don't have game-by-game L5 data here
    if l5_avg >= threshold + 1.0:
        return 90
    elif l5_avg >= threshold + 0.5:
        return 80
    elif l5_avg >= threshold:
        return 70
    elif l5_avg >= threshold - 0.5:
        return 60
    return 50

# ============================================================================
# V7: CORRELATION PENALTY CALCULATION
# ============================================================================
def calculate_correlation_penalty(legs: List[Dict]) -> Tuple[float, List[str]]:
    """
    Calculate total correlation penalty for a parlay.
    Returns (penalty_multiplier, correlation_notes).
    """
    if len(legs) < 2:
        return 1.0, []
    
    penalty = 1.0
    notes = []
    
    for i, leg1 in enumerate(legs):
        for leg2 in legs[i+1:]:
            p1 = leg1.get("player", {})
            p2 = leg2.get("player", {})
            
            # Same team
            if p1.get("team") == p2.get("team"):
                # Same PP unit
                if p1.get("is_pp1") and p2.get("is_pp1"):
                    penalty *= CORRELATION_PENALTIES["same_pp_unit"]
                    notes.append(f"Same PP1: {p1.get('name', '?').split()[-1]}/{p2.get('name', '?').split()[-1]}")
                else:
                    penalty *= CORRELATION_PENALTIES["same_team"]
                    if f"Same team: {p1.get('team')}" not in notes:
                        notes.append(f"Same team: {p1.get('team')}")
            
            # Same game (different teams)
            elif leg1.get("game_id") == leg2.get("game_id"):
                penalty *= CORRELATION_PENALTIES["same_game"]
    
    return penalty, notes

# ============================================================================
# V7: VARIANCE-GATED MAX LEGS
# ============================================================================
def get_max_legs_by_variance(avg_variance: float) -> int:
    """Get maximum recommended parlay legs based on average variance."""
    if avg_variance <= 1.2:
        return 8
    elif avg_variance <= 1.6:
        return 5
    elif avg_variance <= 2.0:
        return 3
    return 2

# ============================================================================
# NHL API FUNCTIONS (Same as V6)
# ============================================================================
@st.cache_data(ttl=300)
def get_todays_schedule(date_str: str) -> List[Dict]:
    url = f"{NHL_WEB_API}/schedule/{date_str}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        games = []
        for gw in data.get("gameWeek", []):
            if gw.get("date") == date_str:
                for game in gw.get("games", []):
                    away = game.get("awayTeam", {}).get("abbrev", "")
                    home = game.get("homeTeam", {}).get("abbrev", "")
                    game_id = str(game.get("id", ""))
                    
                    if not away or not home:
                        continue
                    
                    try:
                        utc_dt = datetime.fromisoformat(game.get("startTimeUTC", "").replace("Z", "+00:00"))
                        time_str = utc_dt.astimezone(EST).strftime("%I:%M %p")
                    except:
                        time_str = "TBD"
                    
                    games.append({"id": game_id, "time": time_str, "away_team": away, "home_team": home})
        return games
    except Exception as e:
        return []

@st.cache_data(ttl=1800)
def get_team_roster(team_abbrev: str) -> List[Dict]:
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

def get_team_defense_stats(team_abbrev: str) -> Dict:
    """Get defense stats with 20-game window."""
    try:
        url = f"{NHL_WEB_API}/club-schedule-season/{team_abbrev}/{SEASON}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        
        completed = [g for g in resp.json().get("games", []) 
                    if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
        
        if not completed:
            return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable", "games": 0}
        
        recent = completed[-20:]
        sa_list = []
        
        for game in recent:
            try:
                box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
                box_resp = requests.get(box_url, timeout=10)
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
                
                time.sleep(0.03)
            except:
                continue
        
        if not sa_list:
            return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable", "games": 0}
        
        sa_pg = statistics.mean(sa_list)
        
        # Trend detection
        trend = "stable"
        if len(sa_list) >= 10:
            mid = len(sa_list) // 2
            older = statistics.mean(sa_list[:mid])
            recent_half = statistics.mean(sa_list[mid:])
            diff = recent_half - older
            if diff >= 2.5: trend = "loosening"
            elif diff <= -2.5: trend = "tightening"
        
        return {
            "team_abbrev": team_abbrev,
            "shots_allowed_per_game": round(sa_pg, 2),
            "grade": get_grade(sa_pg),
            "trend": trend,
            "games": len(sa_list)
        }
    except:
        return {"team_abbrev": team_abbrev, "shots_allowed_per_game": 30.0, "grade": "C", "trend": "stable", "games": 0}

def fetch_player_stats(player_info: Dict) -> Optional[Dict]:
    """Fetch player stats with all V7 metrics."""
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
        
        # Initialize lists
        all_shots, all_toi, all_shifts = [], [], []
        home_shots, away_shots, game_dates = [], [], []
        pp_goals = 0
        
        for game in games:
            shots = max(0, game.get("shots", 0))
            all_shots.append(shots)
            
            # TOI
            toi_str = game.get("toi", "0:00")
            toi_mins = parse_toi(toi_str)
            all_toi.append(toi_mins)
            
            # Shifts
            shifts = game.get("shifts", 0)
            if shifts and shifts > 0:
                all_shifts.append(shifts)
            
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
        
        # Hit rates
        hit_2 = sum(1 for s in all_shots if s >= 2) / gp * 100
        hit_3 = sum(1 for s in all_shots if s >= 3) / gp * 100
        hit_4 = sum(1 for s in all_shots if s >= 4) / gp * 100
        hit_5 = sum(1 for s in all_shots if s >= 5) / gp * 100
        
        # Recent averages
        l5 = all_shots[:5] if len(all_shots) >= 5 else all_shots
        l10 = all_shots[:10] if len(all_shots) >= 10 else all_shots
        l5_avg = sum(l5) / len(l5)
        l10_avg = sum(l10) / len(l10)
        
        std = statistics.stdev(all_shots) if len(all_shots) > 1 else 0
        
        # Streak
        streak = 0
        for s in all_shots:
            if s >= 2: streak += 1
            else: break
        
        home_avg = sum(home_shots) / len(home_shots) if home_shots else avg
        away_avg = sum(away_shots) / len(away_shots) if away_shots else avg
        
        # Bust rate (0-1 SOG games)
        bust_rate = sum(1 for s in all_shots if s <= 1) / gp * 100
        
        # Shutout rate (0 SOG games)
        shutout_rate = sum(1 for s in all_shots if s == 0) / gp * 100
        l5_shutouts = sum(1 for s in l5 if s == 0)
        l10_shutouts = sum(1 for s in l10 if s == 0)
        
        # Cushion rates (threshold + 1)
        cushion_2 = sum(1 for s in all_shots if s >= 3) / gp * 100
        cushion_3 = sum(1 for s in all_shots if s >= 4) / gp * 100
        cushion_4 = sum(1 for s in all_shots if s >= 5) / gp * 100
        
        # TOI metrics
        avg_toi = sum(all_toi) / len(all_toi) if all_toi else 0
        l5_toi_list = all_toi[:5] if len(all_toi) >= 5 else all_toi
        l10_toi_list = all_toi[:10] if len(all_toi) >= 10 else all_toi
        l5_toi = sum(l5_toi_list) / len(l5_toi_list) if l5_toi_list else 0
        l10_toi = sum(l10_toi_list) / len(l10_toi_list) if l10_toi_list else 0
        
        # TOI trend
        toi_trend = "➡️"
        if avg_toi > 0:
            pct = (l5_toi - avg_toi) / avg_toi * 100
            if pct >= 8: toi_trend = "📈"
            elif pct <= -8: toi_trend = "📉"
        
        # SOG per 60
        total_shots = sum(all_shots)
        total_toi = sum(all_toi)
        sog_per_60 = (total_shots / total_toi * 60) if total_toi > 0 else 0
        
        l5_shots_sum = sum(l5)
        l5_toi_sum = sum(l5_toi_list)
        l5_sog_per_60 = (l5_shots_sum / l5_toi_sum * 60) if l5_toi_sum > 0 else 0
        
        # Shifts metrics
        avg_shifts = sum(all_shifts) / len(all_shifts) if all_shifts else 0
        total_shifts = sum(all_shifts) if all_shifts else 0
        sog_per_shift = total_shots / total_shifts if total_shifts > 0 else 0
        
        # PP detection
        is_pp1 = (pp_goals >= 3 and avg >= 2.5) or (pp_goals >= 5)
        
        # B2B detection
        is_b2b = False
        if game_dates and game_dates[0]:
            try:
                last_game = datetime.strptime(game_dates[0], "%Y-%m-%d")
                if (datetime.now() - last_game).days == 1:
                    is_b2b = True
            except:
                pass
        
        # Percentiles
        p10 = calculate_percentile(all_shots, 10)
        p25 = calculate_percentile(all_shots, 25)
        
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
            "bust_rate": round(bust_rate, 1),
            "is_pp1": is_pp1,
            "is_b2b": is_b2b,
            "pp_goals": pp_goals,
            "shutout_rate": round(shutout_rate, 1),
            "l5_shutouts": l5_shutouts,
            "l10_shutouts": l10_shutouts,
            "cushion_2": round(cushion_2, 1),
            "cushion_3": round(cushion_3, 1),
            "cushion_4": round(cushion_4, 1),
            "avg_toi": round(avg_toi, 1),
            "l5_toi": round(l5_toi, 1),
            "l10_toi": round(l10_toi, 1),
            "toi_trend": toi_trend,
            "sog_per_60": round(sog_per_60, 2),
            "l5_sog_per_60": round(l5_sog_per_60, 2),
            "avg_shifts": round(avg_shifts, 1),
            "sog_per_shift": round(sog_per_shift, 3),
            "p10": p10,
            "p25": p25,
        }
    except:
        return None

# ============================================================================
# RESULTS FETCHING (Same as V6)
# ============================================================================
def fetch_results(check_date: str, threshold: int, status_container):
    """Fetch results for saved picks."""
    if check_date not in st.session_state.saved_picks:
        status_container.warning(f"No picks saved for {check_date}")
        return
    
    picks = st.session_state.saved_picks[check_date]
    pick_by_id = {str(p["player_id"]): p for p in picks}
    pick_by_name = {p["player"].lower().strip(): p for p in picks}
    
    games = get_todays_schedule(check_date)
    if not games:
        status_container.error(f"No games found for {check_date}")
        return
    
    results_found = 0
    games_finished = 0
    
    progress = status_container.progress(0, text="Fetching game results...")
    
    for i, game in enumerate(games):
        progress.progress((i + 1) / len(games), text=f"Checking {game['away_team']}@{game['home_team']}...")
        
        try:
            box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
            resp = requests.get(box_url, timeout=15)
            if resp.status_code != 200:
                continue
            
            box_data = resp.json()
            game_state = box_data.get("gameState", "")
            
            if game_state not in ["OFF", "FINAL"]:
                continue
            
            games_finished += 1
            
            # Try multiple paths for player stats
            player_locations = []
            
            if "boxscore" in box_data and "playerByGameStats" in box_data.get("boxscore", {}):
                pbgs = box_data["boxscore"]["playerByGameStats"]
                for team_key in ["homeTeam", "awayTeam"]:
                    if team_key in pbgs:
                        for pos in ["forwards", "defense"]:
                            if pos in pbgs[team_key]:
                                player_locations.extend(pbgs[team_key][pos])
            
            if "playerByGameStats" in box_data:
                pbgs = box_data["playerByGameStats"]
                for team_key in ["homeTeam", "awayTeam"]:
                    if team_key in pbgs:
                        for pos in ["forwards", "defense"]:
                            if pos in pbgs[team_key]:
                                player_locations.extend(pbgs[team_key][pos])
            
            for player in player_locations:
                pid = player.get("playerId") or player.get("id")
                pid_str = str(pid) if pid else ""
                
                name_data = player.get("name", {})
                if isinstance(name_data, dict):
                    player_name = name_data.get("default", "")
                else:
                    player_name = str(name_data) if name_data else ""
                
                actual_sog = player.get("sog", 0) or player.get("shots", 0) or 0
                
                matched_pick = None
                if pid_str and pid_str in pick_by_id:
                    matched_pick = pick_by_id[pid_str]
                elif player_name and player_name.lower().strip() in pick_by_name:
                    matched_pick = pick_by_name[player_name.lower().strip()]
                
                if matched_pick:
                    matched_pick["actual_sog"] = actual_sog
                    matched_pick["hit"] = 1 if actual_sog >= threshold else 0
                    results_found += 1
            
            time.sleep(0.05)
        except:
            continue
    
    progress.empty()
    
    # Update session state
    st.session_state.saved_picks[check_date] = picks
    
    # Save to history
    if results_found > 0:
        st.session_state.results_history[check_date] = picks
        save_history(st.session_state.results_history)
    
    if games_finished == 0:
        status_container.warning("⏳ No finished games found")
    elif results_found == 0:
        status_container.warning(f"⚠️ {games_finished} games finished but no picks matched")
    else:
        status_container.success(f"✅ Updated {results_found} picks from {games_finished} games")

# ============================================================================
# PARLAY GENERATION (V7 Enhanced)
# ============================================================================
def generate_best_parlay_v7(plays: List[Dict], num_legs: int, threshold: int) -> Optional[Dict]:
    """Generate best parlay with V7 correlation adjustments."""
    
    # Filter to parlay-eligible plays (not killed)
    eligible = [p for p in plays if not p.get("killed", False)]
    
    if len(eligible) < num_legs:
        return None
    
    # Sort by score
    sorted_plays = sorted(eligible, key=lambda x: x.get("parlay_score", 0), reverse=True)
    best_legs = sorted_plays[:num_legs]
    
    # Calculate probabilities
    probs = [p.get("model_prob", 70) / 100 for p in best_legs]
    combined_prob, american_odds = calculate_parlay_odds(probs)
    
    # Apply correlation penalty
    corr_penalty, corr_notes = calculate_correlation_penalty(best_legs)
    adjusted_prob = combined_prob * corr_penalty
    adjusted_odds = implied_prob_to_american(adjusted_prob)
    
    avg_score = sum(p.get("parlay_score", 0) for p in best_legs) / num_legs
    min_score = min(p.get("parlay_score", 0) for p in best_legs)
    avg_variance = sum(p.get("player", {}).get("std_dev", 1.5) for p in best_legs) / num_legs
    
    return {
        "legs": best_legs,
        "num_legs": num_legs,
        "combined_prob": combined_prob,
        "adjusted_prob": adjusted_prob,
        "american_odds": american_odds,
        "adjusted_odds": adjusted_odds,
        "correlation_penalty": corr_penalty,
        "correlation_notes": corr_notes,
        "payout_per_100": calculate_parlay_payout(adjusted_odds, 100),
        "avg_parlay_score": avg_score,
        "min_parlay_score": min_score,
        "avg_variance": avg_variance,
        "max_recommended_legs": get_max_legs_by_variance(avg_variance),
    }

# ============================================================================
# MAIN ANALYSIS (V7)
# ============================================================================
def run_analysis_v7(date_str: str, threshold: int, status_container) -> List[Dict]:
    """Run full V7 analysis with compact progress."""
    
    games = get_todays_schedule(date_str)
    if not games:
        status_container.error("No games found!")
        return []
    
    st.session_state.games = games
    
    # R-rated quotes
    QUOTES = [
        "Stop passing and fucking shoot already.",
        "These bitches better get puck on net.",
        "My grandma could hit O1.5 and she's dead.",
    ]
    
    # Ron Burgundy "It's Science" GIF
    SCIENCE_GIF = "https://gifrific.com/wp-content/uploads/2012/06/its-science-anchorman.gif"
    
    import random
    
    # Compact layout
    with status_container:
        st.caption(f"📅 {date_str} | {len(games)} games")
        
        # Single line games summary
        games_str = " • ".join([f"{g['away_team']}@{g['home_team']}" for g in games])
        st.code(games_str, language=None)
        
        # Two columns: status + GIF
        col1, col2 = st.columns([3, 1])
        
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
            quote_text = st.empty()
            quote_text.caption(f"💬 *\"{random.choice(QUOTES)}\"*")
        
        with col2:
            gif_container = st.empty()
            gif_container.image(SCIENCE_GIF, width=150)
    
    # Build team info
    teams_playing = set()
    game_info = {}
    for game in games:
        teams_playing.add(game["away_team"])
        teams_playing.add(game["home_team"])
        game_info[game["away_team"]] = {"opponent": game["home_team"], "home_away": "AWAY", "time": game["time"], "game_id": game["id"]}
        game_info[game["home_team"]] = {"opponent": game["away_team"], "home_away": "HOME", "time": game["time"], "game_id": game["id"]}
    
    # Fetch defense stats
    team_defense = {}
    teams_list = list(teams_playing)
    for i, team in enumerate(teams_list):
        pct = 0.05 + (i / len(teams_list)) * 0.35
        progress_bar.progress(pct)
        status_text.text(f"🛡️ Fetching {team} defense... ({i+1}/{len(teams_list)})")
        team_defense[team] = get_team_defense_stats(team)
        time.sleep(0.05)
    
    # Fetch rosters
    progress_bar.progress(0.45)
    status_text.text("📋 Loading rosters...")
    quote_text.caption(f"💬 *\"{random.choice(QUOTES)}\"*")
    
    all_players = []
    for team in teams_playing:
        roster = get_team_roster(team)
        all_players.extend(roster)
    
    # Analyze players
    plays = []
    total = len(all_players)
    qualified_count = 0
    
    for i, player_info in enumerate(all_players):
        pct = 0.45 + (i / total) * 0.55
        progress_bar.progress(pct)
        status_text.text(f"🔍 {player_info['name'][:18]}... ({i+1}/{total}) | ✅ {qualified_count} found")
        
        # Change quote every ~40 players
        if i % 40 == 0 and i > 0:
            quote_text.caption(f"💬 *\"{random.choice(QUOTES)}\"*")
        
        stats = fetch_player_stats(player_info)
        if not stats:
            continue
        
        hit_rate = stats.get(f"hit_rate_{threshold}plus", 0)
        if hit_rate < MIN_HIT_RATE:
            continue
        
        qualified_count += 1
        
        info = game_info.get(player_info["team"])
        if not info:
            continue
        
        opp = info["opponent"]
        opp_def = team_defense.get(opp, {"grade": "C", "trend": "stable", "shots_allowed_per_game": 30.0})
        is_home = info["home_away"] == "HOME"
        
        # V7 Parlay Score (continuous)
        parlay_score, edges, risks = calculate_parlay_score_v7(stats, opp_def, is_home, threshold)
        
        # V7 Probability (logistic mapping)
        model_prob = score_to_probability(parlay_score) * 100
        
        # Kill switch check
        killed, kill_reason = check_kill_switches(stats, parlay_score, threshold)
        
        # Determine tier
        tier = get_tier_from_score(parlay_score)
        
        # Build highlights
        highlights = []
        if hit_rate >= 92: highlights.append("🎯 Elite Hit")
        if hit_rate <= 72: highlights.append("⚠️ Low Hit")
        if stats["last_5_avg"] >= stats["avg_sog"] + 1.5: highlights.append("🔥 ON FIRE")
        if stats["last_5_avg"] <= stats["avg_sog"] - 1.5: highlights.append("❄️ ICE COLD")
        if stats.get("l5_toi", 0) > 0 and stats.get("avg_toi", 0) > 0:
            toi_change = (stats["l5_toi"] - stats["avg_toi"]) / stats["avg_toi"] * 100
            if toi_change >= 20: highlights.append("📈 TOI Surge")
            elif toi_change <= -20: highlights.append("📉 TOI Drop")
        if stats.get("shutout_rate", 0) >= 18: highlights.append("🚨 Shutout Risk")
        if stats.get("std_dev", 0) >= 2.3: highlights.append("🎲 High Variance")
        if stats.get("std_dev", 0) <= 0.9: highlights.append("🎯 Consistent")
        
        # Tags
        tags = []
        if stats.get("is_pp1"): tags.append("⚡")
        if stats["floor"] >= 1: tags.append("🛡️")
        if stats["current_streak"] >= 5: tags.append(f"🔥{stats['current_streak']}G")
        if stats.get("is_b2b"): tags.append("B2B")
        
        play = {
            "player": stats,
            "player_id": stats["player_id"],
            "opponent": opp,
            "opponent_defense": opp_def,
            "home_away": info["home_away"],
            "game_time": info["time"],
            "game_id": info["game_id"],
            "parlay_score": round(parlay_score, 1),
            "model_prob": round(model_prob, 1),
            "tier": tier,
            "edges": edges,
            "risks": risks,
            "highlights": highlights,
            "tags": " ".join(tags),
            "killed": killed,
            "kill_reason": kill_reason,
        }
        plays.append(play)
    
    # Clear loading UI
    progress_bar.empty()
    status_text.empty()
    quote_text.empty()
    gif_container.empty()
    
    # Sort by score
    plays.sort(key=lambda x: x.get("parlay_score", 0), reverse=True)
    
    # Final message
    locks = len([p for p in plays if "LOCK" in p["tier"]])
    if locks > 0:
        status_container.success(f"🔒 {locks} LOCK(s) found. Don't fuck this up.")
    else:
        status_container.warning(f"😬 {len(plays)} plays, 0 LOCKs. Today might hurt.")
    
    # Save for results tracking
    picks_to_save = [{
        "player_id": p["player"]["player_id"],
        "player": p["player"]["name"],
        "team": p["player"]["team"],
        "opponent": p["opponent"],
        "parlay_score": p["parlay_score"],
        "tier": p["tier"],
        "model_prob": p["model_prob"],
        "hit_rate": p["player"].get(f"hit_rate_{threshold}plus", 0),
        "threshold": threshold,
    } for p in plays]
    
    st.session_state.saved_picks[date_str] = picks_to_save
    
    return plays

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_all_results(plays: List[Dict], threshold: int):
    """Display all results with V7.2 highlighting."""
    
    st.subheader(f"📊 All Results - Over {threshold - 0.5} SOG")
    
    # Summary metrics
    locks = len([p for p in plays if "LOCK" in p["tier"]])
    strong = len([p for p in plays if "STRONG" in p["tier"]])
    solid = len([p for p in plays if "SOLID" in p["tier"]])
    risky = len([p for p in plays if "RISKY" in p["tier"]])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔒 Locks", locks)
    col2.metric("✅ Strong", strong)
    col3.metric("📊 Solid", solid)
    col4.metric("⚠️ Risky", risky)
    col5.metric("Total", len(plays))
    
    # Build table data - compact view (Tier, Team, Loc, vs in Tiered Breakdown)
    rows = []
    for p in plays:
        player = p["player"]
        hit_rate = player.get(f"hit_rate_{threshold}plus", 0)
        cushion = player.get(f"cushion_{threshold}", 0)
        
        # TOI change indicator
        toi_str = f"{player.get('avg_toi', 0):.0f}m"
        if player.get("avg_toi", 0) > 0:
            pct = (player.get("l5_toi", 0) - player["avg_toi"]) / player["avg_toi"] * 100
            if pct >= 10:
                toi_str += f" (+{pct:.0f}%)"
            elif pct <= -10:
                toi_str += f" ({pct:.0f}%)"
        
        row = {
            "Score": f"{get_score_color(p['parlay_score'])} {p['parlay_score']:.0f}",
            "Player": player["name"],
            "Tags": p["tags"],
            "Hit%": f"{hit_rate:.0f}%",
            "Cush%": f"{cushion:.0f}%",
            "Shut%": f"{player.get('shutout_rate', 0):.0f}%",
            "Avg": player["avg_sog"],
            "L5": player["last_5_avg"],
            "L10": player.get("last_10_avg", player["avg_sog"]),
            "SOG/60": player.get("sog_per_60", 0),
            "TOI": toi_str,
            "Prob": p['model_prob'],  # Raw percentage (0-100)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Display with column config - Player auto-fits to content
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True, 
        height=500,
        column_config={
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Tags": st.column_config.TextColumn("Tags", width="small"),
            "Hit%": st.column_config.TextColumn("Hit%", width="small"),
            "Cush%": st.column_config.TextColumn("Cush%", width="small"),
            "Shut%": st.column_config.TextColumn("Shut%", width="small"),
            "Avg": st.column_config.NumberColumn("Avg", format="%.2f", width="small"),
            "L5": st.column_config.NumberColumn("L5", format="%.1f", width="small"),
            "L10": st.column_config.NumberColumn("L10", format="%.1f", width="small"),
            "SOG/60": st.column_config.NumberColumn("SOG/60", format="%.1f", width="small"),
            "TOI": st.column_config.TextColumn("TOI", width="small"),
            "Prob": st.column_config.ProgressColumn(
                "Prob%",
                help="Model probability",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
        }
    )
    
    # Download CSV with full details (includes Tier, Team, vs, Loc)
    csv_rows = []
    for p in plays:
        player = p["player"]
        hit_rate = player.get(f"hit_rate_{threshold}plus", 0)
        cushion = player.get(f"cushion_{threshold}", 0)
        toi_str = f"{player.get('avg_toi', 0):.0f}m"
        if player.get("avg_toi", 0) > 0:
            pct = (player.get("l5_toi", 0) - player["avg_toi"]) / player["avg_toi"] * 100
            if abs(pct) >= 10:
                toi_str += f" ({pct:+.0f}%)"
        
        csv_rows.append({
            "Score": f"{get_score_color(p['parlay_score'])} {p['parlay_score']:.0f}",
            "Tier": p["tier"],
            "Player": player["name"],
            "Tags": p["tags"],
            "Team": player["team"],
            "vs": f"{p['opponent']} ({p['opponent_defense'].get('grade', 'C')})",
            "Loc": "Home" if p["home_away"] == "HOME" else "Away",
            "Hit%": f"{hit_rate:.0f}%",
            "Cush%": f"{cushion:.0f}%",
            "Shut%": f"{player.get('shutout_rate', 0):.0f}%",
            "Avg": player["avg_sog"],
            "L5": player["last_5_avg"],
            "L10": player.get("last_10_avg", player["avg_sog"]),
            "SOG/60": player.get("sog_per_60", 0),
            "TOI": toi_str,
            "Prob": f"{p['model_prob']:.0f}%",
        })
    csv_df = pd.DataFrame(csv_rows)
    csv = csv_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"nhl_sog_v72_{get_est_date()}.csv", "text/csv")

def display_tiered_breakdown(plays: List[Dict], threshold: int):
    """Display plays grouped by tier with edge/risk analysis."""
    
    st.subheader("🎯 Tiered Breakdown")
    
    tiers_order = ["🔒 LOCK", "✅ STRONG", "📊 SOLID", "⚠️ RISKY", "❌ AVOID"]
    
    for tier in tiers_order:
        tier_plays = [p for p in plays if p["tier"] == tier]
        if not tier_plays:
            continue
        
        with st.expander(f"**{tier}** ({len(tier_plays)} plays)", expanded=(tier in ["🔒 LOCK", "✅ STRONG"])):
            for p in tier_plays[:10]:  # Limit to top 10 per tier
                player = p["player"]
                hit_rate = player.get(f"hit_rate_{threshold}plus", 0)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    kill_badge = " 🚫" if p.get("killed") else ""
                    st.markdown(f"**{player['name']}** ({player['team']}) vs {p['opponent']}{kill_badge}")
                    st.caption(f"Score: {p['parlay_score']:.0f} | Hit: {hit_rate:.0f}% | Prob: {p['model_prob']:.0f}%")
                
                with col2:
                    if p["edges"]:
                        st.markdown("**Edges:**")
                        for edge in p["edges"][:3]:
                            st.caption(f"✓ {edge}")
                
                with col3:
                    if p["risks"]:
                        st.markdown("**Risks:**")
                        for risk in p["risks"][:3]:
                            st.caption(f"✗ {risk}")
                
                if p.get("killed"):
                    st.caption(f"⚠️ Killed: {p.get('kill_reason', 'N/A')}")
                
                st.divider()

def display_parlays_v7(plays: List[Dict], threshold: int, unit_size: float):
    """Display V7 parlay recommendations with correlation adjustments."""
    
    st.subheader("💰 Parlay Builder")
    
    eligible = [p for p in plays if not p.get("killed", False)]
    
    if len(eligible) < 2:
        st.warning("Not enough eligible plays for parlays")
        return
    
    # Calculate average variance
    avg_var = sum(p["player"].get("std_dev", 1.5) for p in eligible[:10]) / min(len(eligible), 10)
    max_recommended = get_max_legs_by_variance(avg_var)
    
    st.info(f"📊 Based on average variance (σ={avg_var:.2f}), max recommended legs: **{max_recommended}**")
    
    # Generate parlays for different sizes
    parlay_sizes = [2, 3, 4, 5, 6, 8, 10, len(eligible)]
    parlay_data = []
    
    for size in parlay_sizes:
        if size > len(eligible):
            continue
        
        parlay = generate_best_parlay_v7(eligible, size, threshold)
        if parlay:
            size_label = "MAX" if size == len(eligible) else str(size)
            risk = "✅" if size <= max_recommended else "⚠️"
            
            parlay_data.append({
                "Legs": size_label,
                "Risk": risk,
                "Avg Score": f"{parlay['avg_parlay_score']:.0f}",
                "Min Score": f"{parlay['min_parlay_score']:.0f}",
                "Raw Prob%": f"{parlay['combined_prob']*100:.1f}%",
                "Adj Prob%": f"{parlay['adjusted_prob']*100:.1f}%" if parlay['correlation_penalty'] < 1 else "-",
                "Corr": f"{parlay['correlation_penalty']:.2f}" if parlay['correlation_penalty'] < 1 else "1.00",
                "Odds": f"{parlay['adjusted_odds']:+d}",
                "Payout": f"${parlay['payout_per_100']:.0f}",
                "Players": ", ".join([p["player"]["name"].split()[-1] for p in parlay["legs"][:5]]) + ("..." if size > 5 else ""),
            })
    
    if parlay_data:
        st.dataframe(pd.DataFrame(parlay_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Recommended parlay
    st.subheader("⭐ Recommended Parlay")
    
    # Find best parlay with prob > 30% and within variance limit
    best_parlay = None
    for size in [3, 2, 4, 5]:
        if size > len(eligible):
            continue
        parlay = generate_best_parlay_v7(eligible, size, threshold)
        if parlay and parlay["adjusted_prob"] >= 0.30 and size <= max_recommended:
            best_parlay = parlay
            break
    
    if not best_parlay:
        best_parlay = generate_best_parlay_v7(eligible, 2, threshold)
    
    if best_parlay:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Legs", best_parlay["num_legs"])
        col2.metric("Probability", f"{best_parlay['adjusted_prob']*100:.1f}%")
        col3.metric("Odds", f"{best_parlay['adjusted_odds']:+d}")
        col4.metric("$100 Payout", f"${best_parlay['payout_per_100']:.0f}")
        
        if best_parlay.get("correlation_notes"):
            st.warning(f"⚠️ Correlation: {', '.join(best_parlay['correlation_notes'])}")
        
        st.markdown("**Legs:**")
        for leg in best_parlay["legs"]:
            player = leg["player"]
            st.markdown(f"- **{player['name']}** ({player['team']}) O{threshold-0.5} SOG | Score: {leg['parlay_score']:.0f} | {leg['tier']}")
        
        # Copy button
        copy_text = f"🏒 NHL SOG Parlay ({best_parlay['num_legs']}-leg)\n"
        copy_text += f"Prob: {best_parlay['adjusted_prob']*100:.0f}% | Odds: {best_parlay['adjusted_odds']:+d}\n\n"
        for leg in best_parlay["legs"]:
            copy_text += f"• {leg['player']['name']} O{threshold-0.5} SOG\n"
        
        st.code(copy_text, language=None)

def display_results_tracker(threshold: int):
    """Display results tracking tab."""
    
    st.subheader("📈 Results Tracker")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        check_date = st.date_input("Check Date", value=datetime.now(EST).date())
        check_date_str = check_date.strftime("%Y-%m-%d")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Fetch Results", type="primary"):
            status = st.empty()
            fetch_results(check_date_str, threshold, status)
            st.rerun()
    
    # Show results for selected date
    if check_date_str in st.session_state.results_history:
        picks = st.session_state.results_history[check_date_str]
        picks_with_results = [p for p in picks if "actual_sog" in p]
        
        if picks_with_results:
            hits = sum(1 for p in picks_with_results if p.get("hit", 0) == 1)
            total = len(picks_with_results)
            
            st.success(f"**{check_date_str}**: {hits}/{total} ({hits/total*100:.0f}% hit rate)")
            
            # Results by tier
            tier_results = {}
            for p in picks_with_results:
                tier = p.get("tier", "Unknown")
                if tier not in tier_results:
                    tier_results[tier] = {"hits": 0, "total": 0}
                tier_results[tier]["total"] += 1
                if p.get("hit", 0) == 1:
                    tier_results[tier]["hits"] += 1
            
            st.markdown("**Results by Tier:**")
            tier_data = []
            for tier, data in sorted(tier_results.items()):
                pct = data["hits"] / data["total"] * 100 if data["total"] > 0 else 0
                tier_data.append({
                    "Tier": tier,
                    "Hits": data["hits"],
                    "Total": data["total"],
                    "Win%": f"{pct:.0f}%"
                })
            st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)
            
            # Detailed results
            st.markdown("**Detailed Results:**")
            detail_data = []
            for p in picks_with_results:
                detail_data.append({
                    "Result": "✅" if p.get("hit", 0) == 1 else "❌",
                    "Player": p["player"],
                    "Team": p["team"],
                    "vs": p["opponent"],
                    "Score": p["parlay_score"],
                    "Tier": p["tier"],
                    "Actual SOG": p.get("actual_sog", "?"),
                    "Threshold": threshold,
                })
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
    
    # Historical summary
    st.markdown("---")
    st.subheader("📊 Historical Performance")
    
    if st.session_state.results_history:
        all_picks = []
        for date, picks in st.session_state.results_history.items():
            for p in picks:
                if "actual_sog" in p:
                    all_picks.append(p)
        
        if all_picks:
            total_hits = sum(1 for p in all_picks if p.get("hit", 0) == 1)
            total_picks = len(all_picks)
            overall_pct = total_hits / total_picks * 100 if total_picks > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Picks Tracked", total_picks)
            col2.metric("Total Hits", total_hits)
            col3.metric("Overall Win%", f"{overall_pct:.1f}%")
            
            # By tier
            tier_summary = {}
            for p in all_picks:
                tier = p.get("tier", "Unknown")
                if tier not in tier_summary:
                    tier_summary[tier] = {"hits": 0, "total": 0}
                tier_summary[tier]["total"] += 1
                if p.get("hit", 0) == 1:
                    tier_summary[tier]["hits"] += 1
            
            summary_data = []
            for tier in ["🔒 LOCK", "✅ STRONG", "📊 SOLID", "⚠️ RISKY", "❌ AVOID"]:
                if tier in tier_summary:
                    data = tier_summary[tier]
                    pct = data["hits"] / data["total"] * 100 if data["total"] > 0 else 0
                    summary_data.append({
                        "Tier": tier,
                        "Hits": data["hits"],
                        "Total": data["total"],
                        "Win%": f"{pct:.1f}%"
                    })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    else:
        st.info("No historical data yet. Run analysis and fetch results to start tracking.")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer V7.2")
    st.caption("Hit Rate is King | Variance NOT penalized")
    st.caption("Professional Grade: Continuous Scoring • Dynamic Weighting • Correlation Penalties")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        today_est = get_est_datetime().date()
        selected_date = st.date_input("📅 Select Date", value=today_est)
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        st.subheader("🎯 Bet Type")
        bet_type = st.radio(
            "SOG Threshold:",
            ["Over 1.5 (2+ SOG)", "Over 2.5 (3+ SOG)", "Over 3.5 (4+ SOG)"],
            index=0
        )
        threshold = 2 if "1.5" in bet_type else 3 if "2.5" in bet_type else 4
        
        st.markdown("---")
        
        st.subheader("💰 Bankroll")
        unit_size = st.number_input("Unit Size ($)", min_value=1, max_value=1000, value=25)
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # V7 Model Info
        with st.expander("ℹ️ V7 Model Info"):
            st.markdown("""
            **Continuous Scoring:**
            - No bucket cliffs
            - Linear scaling for all factors
            
            **Dynamic Weighting:**
            - Extreme matchups weighted more
            - Magnitude-based situational factors
            
            **Kill Switches:**
            - Auto-exclude dangerous plays
            - Still shown in All Results
            
            **Correlation Penalties:**
            - Same team: -6%
            - Same line: -10%
            - Same PP: -8%
            
            **V7.2 Tiers:**
            - 🔒 LOCK: 88+
            - ✅ STRONG: 80-87
            - 📊 SOLID: 70-79
            - ⚠️ RISKY: 60-69
            - ❌ AVOID: <60
            
            **Scoring:** Hit Rate is King
            - Base = 50 + (Hit% - 70) × 1.5
            - Modifiers adjust from base
            """)
        
        st.caption(f"Current: {get_est_datetime().strftime('%I:%M %p EST')}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Results",
        "🎯 Tiered Breakdown",
        "💰 Parlays",
        "📈 Results Tracker"
    ])
    
    # Run analysis
    if run_analysis:
        with tab1:
            status = st.container()
            plays = run_analysis_v7(date_str, threshold, status)
            st.session_state.plays = plays
    
    # Display content
    with tab1:
        if st.session_state.plays:
            display_all_results(st.session_state.plays, threshold)
        elif not run_analysis:
            st.info("👈 Click **Run Analysis** to fetch today's plays")
    
    with tab2:
        if st.session_state.plays:
            display_tiered_breakdown(st.session_state.plays, threshold)
        else:
            st.info("Run analysis first to see tiered breakdown")
    
    with tab3:
        if st.session_state.plays:
            display_parlays_v7(st.session_state.plays, threshold, unit_size)
        else:
            st.info("Run analysis first to see parlay recommendations")
    
    with tab4:
        display_results_tracker(threshold)

if __name__ == "__main__":
    main()
