#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer V6
=============================
COMPREHENSIVE MERGE: v4.2 features + v5.1 rate metrics

FEATURES:
- Results tracking with historical storage
- TOI, SOG/60, shifts metrics (from v5.1)
- Shutout rate, cushion rate (from v5.1)  
- Parlay Score system (from v4.2)
- Fetch Results with game status
- Win rate tracking over time
- Three tabs: All Results | Tiered Breakdown | Parlays
- Parlay sizes 1-12 and MAX
- Optimized fetching with progress display

v6.0 - January 2026
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import statistics

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NHL SOG Analyzer V6",
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
DATA_DIR = "nhl_sog_data"
HISTORY_FILE = f"{DATA_DIR}/results_history.json"

# Defense grades (SA/G thresholds)
MATCHUP_GRADES = {
    "A+": 34.0, "A": 32.0, "B": 30.0, "C": 28.0, "D": 26.0, "F": 0.0
}

# Probability caps to prevent overconfidence
PROB_CAPS = {
    "max_absolute": 94.0,
    "high_shutout": 82.0,      # shutout_rate > 8%
    "high_variance": 85.0,     # std_dev > 2.0
    "recent_shutout": 80.0,    # shutout in L5
    "toi_dropping": 85.0,      # TOI trending down
    "small_sample": 88.0,      # <20 games
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

def get_grade(sa_pg: float) -> str:
    for grade, threshold in MATCHUP_GRADES.items():
        if sa_pg >= threshold: return grade
    return "F"

def get_trend(l5: float, season: float) -> Tuple[str, bool, bool]:
    diff = l5 - season
    if diff >= 0.5: return "🔥", True, False
    if diff <= -0.5: return "❄️", False, True
    return "➡️", False, False

def get_toi_trend(l5_toi: float, avg_toi: float) -> str:
    if avg_toi == 0: return "❓"
    pct = (l5_toi - avg_toi) / avg_toi * 100
    if pct >= 8: return "📈"
    elif pct <= -8: return "📉"
    return "➡️"

def calculate_percentile(data: List[int], pct: int) -> int:
    if not data: return 0
    s = sorted(data)
    idx = max(0, min(int(len(s) * pct / 100), len(s) - 1))
    return s[idx]

def calculate_parlay_odds(probs: List[float]) -> Tuple[float, int]:
    combined = 1.0
    for p in probs: combined *= p
    return combined, implied_prob_to_american(combined)

def calculate_parlay_payout(odds: int, stake: float = 100) -> float:
    if odds > 0: return stake + (stake * odds / 100)
    return stake + (stake * 100 / abs(odds))

# Score color based on parlay score
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

def get_tier_from_score(score: int) -> str:
    if score >= 80: return "🔒 LOCK"
    if score >= 70: return "✅ STRONG"
    if score >= 60: return "📊 SOLID"
    if score >= 50: return "⚠️ RISKY"
    return "❌ AVOID"

# ============================================================================
# NHL API FUNCTIONS
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

def get_team_defense_stats(team_abbrev: str, status_callback=None) -> Dict:
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
    """
    V6: Enhanced player stats with TOI and rate-based metrics.
    Single API call for speed.
    """
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
        
        # Shutout rate (0 SOG games) - V6
        shutout_rate = sum(1 for s in all_shots if s == 0) / gp * 100
        l5_shutouts = sum(1 for s in l5 if s == 0)
        l10_shutouts = sum(1 for s in l10 if s == 0)
        
        # Cushion rates - V6
        cushion_2 = sum(1 for s in all_shots if s >= 3) / gp * 100
        cushion_3 = sum(1 for s in all_shots if s >= 4) / gp * 100
        cushion_4 = sum(1 for s in all_shots if s >= 5) / gp * 100
        
        # TOI metrics - V6
        avg_toi = sum(all_toi) / len(all_toi) if all_toi else 0
        l5_toi_list = all_toi[:5] if len(all_toi) >= 5 else all_toi
        l10_toi_list = all_toi[:10] if len(all_toi) >= 10 else all_toi
        l5_toi = sum(l5_toi_list) / len(l5_toi_list) if l5_toi_list else 0
        l10_toi = sum(l10_toi_list) / len(l10_toi_list) if l10_toi_list else 0
        toi_trend = get_toi_trend(l5_toi, avg_toi)
        
        # SOG per 60 - V6
        total_shots = sum(all_shots)
        total_toi = sum(all_toi)
        sog_per_60 = (total_shots / total_toi * 60) if total_toi > 0 else 0
        
        l5_shots_sum = sum(l5)
        l5_toi_sum = sum(l5_toi_list)
        l5_sog_per_60 = (l5_shots_sum / l5_toi_sum * 60) if l5_toi_sum > 0 else 0
        
        # Shifts metrics - V6
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
            # V6 new metrics
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
# V6: ENHANCED PARLAY SCORE
# ============================================================================
def calculate_parlay_score(player: Dict, opp_def: Dict, is_home: bool, threshold: int, is_hot: bool, is_cold: bool) -> Tuple[int, List[str], List[str]]:
    """
    V6: Comprehensive parlay score (0-100) with edge/risk tracking.
    Weighted optimally based on research for maximum predictability.
    """
    score = 50  # Base
    edges = []
    risks = []
    
    hit_rate = player.get(f"hit_rate_{threshold}plus", 0)
    cushion_key = f"cushion_{threshold}"
    cushion = player.get(cushion_key, 0)
    
    # ========== HIT RATE (max +25) - Most important ==========
    if hit_rate >= 95:
        score += 25
        edges.append(f"Elite {hit_rate:.0f}% hit")
    elif hit_rate >= 90:
        score += 20
        edges.append(f"Strong {hit_rate:.0f}% hit")
    elif hit_rate >= 85:
        score += 15
    elif hit_rate >= 80:
        score += 10
    elif hit_rate >= 75:
        score += 5
    elif hit_rate < 70:
        score -= 10
        risks.append(f"Low {hit_rate:.0f}% hit")
    
    # ========== SHUTOUT RATE (max +10/-15) - Key V6 metric ==========
    shutout = player.get("shutout_rate", 10)
    if shutout <= 3:
        score += 10
        edges.append(f"Rare shutouts ({shutout:.0f}%)")
    elif shutout <= 5:
        score += 6
    elif shutout <= 8:
        score += 3
    elif shutout > 15:
        score -= 15
        risks.append(f"🚨 High shutout ({shutout:.0f}%)")
    elif shutout > 10:
        score -= 8
        risks.append(f"⚠️ Shutout risk ({shutout:.0f}%)")
    
    # ========== RECENT SHUTOUTS (max -20) ==========
    l5_so = player.get("l5_shutouts", 0)
    if l5_so >= 2:
        score -= 20
        risks.append(f"🚨 {l5_so} shutouts in L5!")
    elif l5_so == 1:
        score -= 10
        risks.append("⚠️ Shutout in L5")
    
    # ========== CUSHION RATE (max +8) ==========
    if cushion >= 75:
        score += 8
        edges.append(f"🛡️ {cushion:.0f}% cushion")
    elif cushion >= 60:
        score += 4
    elif cushion < 35:
        score -= 5
        risks.append(f"Low cushion ({cushion:.0f}%)")
    
    # ========== FLOOR (max +12) ==========
    floor = player.get("floor", 0)
    if floor >= threshold:
        score += 12
        edges.append(f"🔒 Floor={floor} NEVER misses!")
    elif floor >= 2:
        score += 8
        edges.append("🛡️ Never below 2")
    elif floor >= 1:
        score += 5
        edges.append("🛡️ Never shutout")
    elif floor == 0:
        score -= 5
        risks.append("Floor=0 risk")
    
    # ========== SOG/60 RATE (max +8) ==========
    sog60 = player.get("sog_per_60", 0)
    if sog60 >= 10.0:
        score += 8
        edges.append(f"⚡ Elite rate ({sog60:.1f}/60)")
    elif sog60 >= 8.0:
        score += 4
    elif sog60 < 5.0 and sog60 > 0:
        score -= 4
        risks.append(f"Low rate ({sog60:.1f}/60)")
    
    # ========== TOI TREND (max +6/-8) ==========
    toi_trend = player.get("toi_trend", "➡️")
    if toi_trend == "📈":
        score += 6
        edges.append(f"📈 TOI increasing")
    elif toi_trend == "📉":
        score -= 8
        risks.append(f"📉 TOI decreasing")
    
    # ========== VOLUME (max +8) ==========
    avg = player.get("avg_sog", 0)
    if avg >= 4.0:
        score += 8
        edges.append(f"High volume ({avg:.1f})")
    elif avg >= 3.5:
        score += 6
    elif avg >= 3.0:
        score += 4
    elif avg >= 2.5:
        score += 2
    
    # ========== PP1 (max +8) ==========
    if player.get("is_pp1"):
        score += 8
        edges.append("⚡ PP1")
    
    # ========== VARIANCE (max +8/-12) ==========
    std = player.get("std_dev", 1.5)
    if std < 0.8:
        score += 8
        edges.append(f"Very consistent (σ={std:.1f})")
    elif std < 1.2:
        score += 4
    elif std > 2.5:
        score -= 12
        risks.append(f"🎲 High variance (σ={std:.1f})")
    elif std > 2.0:
        score -= 8
        risks.append(f"⚠️ Volatile (σ={std:.1f})")
    elif std > 1.6:
        score -= 4
    
    # ========== MATCHUP (max +10/-8) ==========
    opp_grade = opp_def.get("grade", "C")
    if opp_grade == "A+":
        score += 10
        edges.append(f"Soft matchup ({opp_grade})")
    elif opp_grade == "A":
        score += 7
        edges.append(f"Good matchup ({opp_grade})")
    elif opp_grade == "B":
        score += 4
    elif opp_grade == "D":
        score -= 5
        risks.append(f"Tough matchup ({opp_grade})")
    elif opp_grade == "F":
        score -= 8
        risks.append(f"Hard matchup ({opp_grade})")
    
    # Defense trend
    opp_trend = opp_def.get("trend", "stable")
    if opp_trend == "loosening":
        score += 4
        edges.append("Def loosening 📈")
    elif opp_trend == "tightening":
        score -= 4
        risks.append("Def tightening 📉")
    
    # ========== FORM (max +5/-8) ==========
    if is_hot:
        score += 5
        edges.append("🔥 Hot")
    elif is_cold:
        score -= 8
        risks.append("❄️ Cold")
    
    # ========== SITUATIONAL (max +2/-3) ==========
    if is_home:
        score += 2
    
    if player.get("is_b2b"):
        score -= 3
        risks.append("B2B")
    
    return max(0, min(100, score)), edges, risks

# ============================================================================
# V6: PROBABILITY MODEL WITH CAPS
# ============================================================================
def calculate_model_probability(player: Dict, opp_def: Dict, is_home: bool, threshold: int) -> Tuple[float, bool, str]:
    """V6: Poisson probability with caps to prevent overconfidence."""
    
    # Rate-based lambda if TOI available
    if player.get("avg_toi", 0) > 0 and player.get("sog_per_60", 0) > 0:
        # Weight recent rates more
        l5_rate = player.get("l5_sog_per_60", 0)
        season_rate = player.get("sog_per_60", 0)
        weighted_rate = l5_rate * 0.5 + season_rate * 0.5
        
        # Project TOI
        toi_trend = player.get("toi_trend", "➡️")
        if toi_trend == "📈":
            expected_toi = player.get("l5_toi", 0) * 1.02
        elif toi_trend == "📉":
            expected_toi = player.get("l5_toi", 0) * 0.98
        else:
            expected_toi = (player.get("l5_toi", 0) * 0.6) + (player.get("avg_toi", 0) * 0.4)
        
        base_lambda = (weighted_rate / 60) * expected_toi
    else:
        # Fallback to traditional weighting
        base_lambda = (
            player["last_5_avg"] * 0.45 +
            player["last_10_avg"] * 0.30 +
            player["avg_sog"] * 0.25
        )
    
    # Adjustments
    hit_rate = player.get(f"hit_rate_{threshold}plus", 75)
    if hit_rate >= 95: hr_factor = 1.15
    elif hit_rate >= 90: hr_factor = 1.12
    elif hit_rate >= 85: hr_factor = 1.08
    elif hit_rate >= 80: hr_factor = 1.04
    else: hr_factor = 1.0
    
    pp_factor = 1.15 if player.get("is_pp1") else 1.0
    ha_factor = 1.03 if is_home else 0.97
    opp_factor = opp_def.get("shots_allowed_per_game", 30.0) / 30.0
    
    adj_lambda = base_lambda * hr_factor * pp_factor * ha_factor * opp_factor
    
    # Raw probability
    raw_prob = poisson_prob_at_least(adj_lambda, threshold) * 100
    
    # Apply caps
    max_prob = PROB_CAPS["max_absolute"]
    cap_reasons = []
    
    if player.get("shutout_rate", 0) > 8:
        max_prob = min(max_prob, PROB_CAPS["high_shutout"])
        cap_reasons.append(f"shut={player.get('shutout_rate', 0):.0f}%")
    
    if player.get("std_dev", 0) > 2.0:
        max_prob = min(max_prob, PROB_CAPS["high_variance"])
        cap_reasons.append(f"σ={player.get('std_dev', 0):.1f}")
    
    if player.get("l5_shutouts", 0) >= 1:
        max_prob = min(max_prob, PROB_CAPS["recent_shutout"])
        cap_reasons.append("L5_shut")
    
    if player.get("toi_trend") == "📉":
        max_prob = min(max_prob, PROB_CAPS["toi_dropping"])
        cap_reasons.append("TOI↓")
    
    if player.get("games_played", 0) < 20:
        max_prob = min(max_prob, PROB_CAPS["small_sample"])
        cap_reasons.append(f"n={player.get('games_played', 0)}")
    
    if raw_prob > max_prob:
        return max_prob, True, ", ".join(cap_reasons)
    
    return raw_prob, False, ""

# ============================================================================
# PARLAY GENERATION
# ============================================================================
def generate_best_parlay(plays: List[Dict], num_legs: int, threshold: int) -> Optional[Dict]:
    if len(plays) < num_legs:
        return None
    
    sorted_plays = sorted(plays, key=lambda x: x.get("parlay_score", 0), reverse=True)
    best_legs = sorted_plays[:num_legs]
    
    prob_key = f"prob_{threshold}plus"
    probs = [p[prob_key] / 100 for p in best_legs]
    combined_prob, american_odds = calculate_parlay_odds(probs)
    avg_score = sum(p.get("parlay_score", 0) for p in best_legs) / num_legs
    min_score = min(p.get("parlay_score", 0) for p in best_legs)
    
    return {
        "legs": best_legs,
        "num_legs": num_legs,
        "combined_prob": combined_prob,
        "american_odds": american_odds,
        "payout_per_100": calculate_parlay_payout(american_odds, 100),
        "avg_parlay_score": avg_score,
        "min_parlay_score": min_score,
    }

# ============================================================================
# RESULTS FETCHING
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
# MAIN ANALYSIS
# ============================================================================
def run_analysis(date_str: str, threshold: int, status_container) -> List[Dict]:
    """Run full analysis with detailed progress."""
    
    games = get_todays_schedule(date_str)
    if not games:
        status_container.error("No games found!")
        return []
    
    # Show games
    status_container.subheader(f"📅 Games: {date_str}")
    game_df = pd.DataFrame([{"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]} for g in games])
    status_container.dataframe(game_df, use_container_width=True, hide_index=True)
    
    # Build team info
    teams_playing = set()
    game_info = {}
    for game in games:
        teams_playing.add(game["away_team"])
        teams_playing.add(game["home_team"])
        game_info[game["away_team"]] = {"opponent": game["home_team"], "home_away": "AWAY", "time": game["time"], "game_id": game["id"]}
        game_info[game["home_team"]] = {"opponent": game["away_team"], "home_away": "HOME", "time": game["time"], "game_id": game["id"]}
    
    progress = status_container.progress(0, text="Starting analysis...")
    
    # Fetch defense stats
    team_defense = {}
    teams_list = list(teams_playing)
    for i, team in enumerate(teams_list):
        progress.progress(0.05 + (i / len(teams_list)) * 0.35, text=f"Fetching {team} defense...")
        team_defense[team] = get_team_defense_stats(team)
        time.sleep(0.05)
    
    # Fetch rosters
    progress.progress(0.45, text="Fetching rosters...")
    all_players = []
    for team in teams_playing:
        roster = get_team_roster(team)
        all_players.extend(roster)
    
    # Analyze players
    plays = []
    total = len(all_players)
    
    for i, player_info in enumerate(all_players):
        progress.progress(0.45 + (i / total) * 0.55, text=f"Analyzing {player_info['name']}...")
        
        stats = fetch_player_stats(player_info)
        if not stats:
            continue
        
        hit_rate = stats.get(f"hit_rate_{threshold}plus", 0)
        if hit_rate < MIN_HIT_RATE:
            continue
        
        info = game_info.get(player_info["team"])
        if not info:
            continue
        
        opp = info["opponent"]
        opp_def = team_defense.get(opp, {"grade": "C", "trend": "stable", "shots_allowed_per_game": 30.0})
        is_home = info["home_away"] == "HOME"
        
        # Trend
        trend_emoji, is_hot, is_cold = get_trend(stats["last_5_avg"], stats["avg_sog"])
        
        # Parlay score
        parlay_score, edges, risks = calculate_parlay_score(stats, opp_def, is_home, threshold, is_hot, is_cold)
        
        # Probability
        prob, capped, cap_reason = calculate_model_probability(stats, opp_def, is_home, threshold)
        
        # Tags
        tags = []
        if stats.get("is_pp1"): tags.append("⚡")
        if stats["floor"] >= 1: tags.append("🛡️")
        if stats["current_streak"] >= 5: tags.append(f"🔥{stats['current_streak']}G")
        if stats.get("is_b2b"): tags.append("B2B")
        
        play = {
            "player": stats,
            "opponent": opp,
            "opponent_defense": opp_def,
            "home_away": info["home_away"],
            "game_time": info["time"],
            "game_id": info["game_id"],
            f"prob_{threshold}plus": round(prob, 1),
            "prob_capped": capped,
            "cap_reason": cap_reason,
            "parlay_score": parlay_score,
            "parlay_grade": get_grade_from_score(parlay_score),
            "tier": get_tier_from_score(parlay_score),
            "trend": trend_emoji,
            "is_hot": is_hot,
            "is_cold": is_cold,
            "is_qualified": hit_rate >= 85 and not is_cold,
            "tags": " ".join(tags),
            "edges": edges,
            "risks": risks,
            "status_icon": "✅" if hit_rate >= 85 and not is_cold else "⚠️",
        }
        plays.append(play)
    
    progress.empty()
    
    if not plays:
        status_container.warning("No qualifying plays found!")
        return []
    
    # Sort by parlay score
    plays.sort(key=lambda x: x["parlay_score"], reverse=True)
    
    # Save picks
    picks_to_save = [{
        "player_id": p["player"]["player_id"],
        "player": p["player"]["name"],
        "team": p["player"]["team"],
        "opponent": p["opponent"],
        "parlay_score": p["parlay_score"],
        "parlay_grade": p["parlay_grade"],
        "model_prob": p[f"prob_{threshold}plus"],
        "hit_rate": p["player"][f"hit_rate_{threshold}plus"],
        "is_qualified": p["is_qualified"],
        "threshold": threshold,
    } for p in plays]
    
    st.session_state.saved_picks[date_str] = picks_to_save
    st.session_state.games = games
    
    return plays

# ============================================================================
# UI COMPONENTS
# ============================================================================
def show_all_results(plays: List[Dict], threshold: int, date_str: str):
    """Tab 1: All results with full details."""
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    cushion_key = f"cushion_{threshold}"
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    locks = len([p for p in plays if "LOCK" in p["tier"]])
    strong = len([p for p in plays if "STRONG" in p["tier"]])
    solid = len([p for p in plays if "SOLID" in p["tier"]])
    capped = len([p for p in plays if p["prob_capped"]])
    qualified = len([p for p in plays if p["is_qualified"]])
    
    col1.metric("🔒 Locks", locks)
    col2.metric("✅ Strong", strong)
    col3.metric("📊 Solid", solid)
    col4.metric("🔻 Capped", capped)
    col5.metric("✅ Qualified", qualified)
    col6.metric("Total", len(plays))
    
    # Results table
    data = []
    for p in plays:
        s = p["player"]
        hit = s.get(hit_key, 0)
        prob = p.get(prob_key, 0)
        cushion = s.get(cushion_key, 0)
        
        opp_def = p["opponent_defense"]
        def_grade = opp_def.get("grade", "C")
        def_trend = opp_def.get("trend", "stable")
        if def_trend == "loosening":
            def_display = f"{def_grade}📈"
        elif def_trend == "tightening":
            def_display = f"{def_grade}📉"
        else:
            def_display = def_grade
        
        data.append({
            "": p["status_icon"],
            "Tier": p["tier"],
            "Player": s["name"],
            "Score": f"{get_score_color(p['parlay_score'])} {p['parlay_score']}",
            "Tags": p["tags"],
            "Team": s["team"],
            "vs": p["opponent"],
            "Model%": f"{prob:.1f}%",
            "Hit%": f"{hit:.0f}%",
            "Cush%": f"{cushion:.0f}%",
            "Shut%": f"{s.get('shutout_rate', 0):.0f}%",
            "Avg": s["avg_sog"],
            "L5": s["last_5_avg"],
            "L10": s["last_10_avg"],
            "SOG/60": s.get("sog_per_60", 0),
            "TOI": f"{s.get('avg_toi', 0):.0f}m",
            "TOI📈": s.get("toi_trend", ""),
            "σ": s["std_dev"],
            "Def": def_display,
            "Cap": "🔻" if p["prob_capped"] else "",
        })
    
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    # Download
    st.download_button("📥 Download CSV", data=pd.DataFrame(data).to_csv(index=False), 
                      file_name=f"nhl_sog_v6_{date_str}.csv", mime="text/csv")

def show_tiered_breakdown(plays: List[Dict], threshold: int):
    """Tab 2: Breakdown by tier."""
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    cushion_key = f"cushion_{threshold}"
    
    locks = [p for p in plays if "LOCK" in p["tier"]]
    strong = [p for p in plays if "STRONG" in p["tier"]]
    solid = [p for p in plays if "SOLID" in p["tier"]]
    risky = [p for p in plays if "RISKY" in p["tier"] or "AVOID" in p["tier"]]
    
    def show_tier_table(tier_plays, title, expanded=True):
        with st.expander(f"{title} ({len(tier_plays)})", expanded=expanded):
            if not tier_plays:
                st.info("None")
                return
            
            data = []
            for p in tier_plays:
                s = p["player"]
                data.append({
                    "Player": s["name"],
                    "Score": f"{get_score_color(p['parlay_score'])} {p['parlay_score']}",
                    "Tags": p["tags"],
                    "vs": p["opponent"],
                    "Model%": f"{p[prob_key]:.1f}%",
                    "Hit%": f"{s[hit_key]:.0f}%",
                    "Cush%": f"{s.get(cushion_key, 0):.0f}%",
                    "Shut%": f"{s.get('shutout_rate', 0):.0f}%",
                    "Avg": s["avg_sog"],
                    "L5": s["last_5_avg"],
                    "Floor": s["floor"],
                })
            
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            
            # Show edges/risks for top picks
            if len(tier_plays) <= 5:
                for p in tier_plays:
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            if p["edges"]:
                                st.success(f"**{p['player']['name']}**: {', '.join(p['edges'][:3])}")
                        with col2:
                            if p["risks"]:
                                st.warning(f"**Risks**: {', '.join(p['risks'][:3])}")
    
    show_tier_table(locks, "🔒 LOCKS - Best for Parlays", True)
    show_tier_table(strong, "✅ STRONG - Core Picks", True)
    show_tier_table(solid, "📊 SOLID - Supporting Picks", False)
    show_tier_table(risky, "⚠️ RISKY/AVOID - Use Caution", False)

def show_parlays_tab(plays: List[Dict], threshold: int, unit_size: float):
    """Tab 3: Parlay builder for all sizes."""
    
    prob_key = f"prob_{threshold}plus"
    sorted_plays = sorted(plays, key=lambda x: x.get("parlay_score", 0), reverse=True)
    
    st.success(f"Building from **{len(sorted_plays)}** players (sorted by Parlay Score)")
    
    # Best parlay table for all sizes
    st.subheader("📊 Best Parlay by Size")
    
    max_legs = min(len(sorted_plays), 15)
    parlay_table = []
    parlays_dict = {}
    
    for num_legs in range(1, max_legs + 1):
        parlay = generate_best_parlay(sorted_plays, num_legs, threshold)
        if parlay:
            players = ", ".join([p["player"]["name"] for p in parlay["legs"]])
            parlay_table.append({
                "Legs": num_legs,
                "Avg Score": f"{parlay['avg_parlay_score']:.0f}",
                "Min Score": f"{parlay['min_parlay_score']:.0f}",
                "Prob%": f"{parlay['combined_prob']*100:.1f}%",
                "Odds": f"{parlay['american_odds']:+d}" if parlay['american_odds'] < 10000 else "—",
                f"${unit_size:.0f}→": f"${parlay['payout_per_100'] * unit_size / 100:.0f}",
                "Players": players[:70] + "..." if len(players) > 70 else players
            })
            parlays_dict[num_legs] = parlay
    
    # MAX parlay
    if len(sorted_plays) > 0:
        max_parlay = generate_best_parlay(sorted_plays, len(sorted_plays), threshold)
        if max_parlay:
            players = ", ".join([p["player"]["name"] for p in max_parlay["legs"]])
            parlay_table.append({
                "Legs": f"MAX ({len(sorted_plays)})",
                "Avg Score": f"{max_parlay['avg_parlay_score']:.0f}",
                "Min Score": f"{max_parlay['min_parlay_score']:.0f}",
                "Prob%": f"{max_parlay['combined_prob']*100:.2f}%",
                "Odds": f"{max_parlay['american_odds']:+d}" if max_parlay['american_odds'] < 100000 else "—",
                f"${unit_size:.0f}→": f"${max_parlay['payout_per_100'] * unit_size / 100:.0f}",
                "Players": players[:70] + "..." if len(players) > 70 else players
            })
            parlays_dict["MAX"] = max_parlay
    
    st.dataframe(pd.DataFrame(parlay_table), use_container_width=True, hide_index=True)
    
    # Copy buttons
    st.subheader("📋 Copy Parlays")
    
    cols = st.columns(4)
    copy_sizes = [2, 3, 5, 10]
    
    for i, size in enumerate(copy_sizes):
        if size in parlays_dict:
            parlay = parlays_dict[size]
            with cols[i]:
                with st.expander(f"**{size}-Leg** ({parlay['combined_prob']*100:.0f}%)"):
                    text = f"🏒 NHL {size}-Leg Parlay\n" + "─" * 30 + "\n"
                    for leg in parlay["legs"]:
                        p = leg["player"]
                        text += f"{get_score_color(leg['parlay_score'])} {p['name']} O{threshold-0.5} SOG\n"
                    text += "─" * 30 + "\n"
                    text += f"Combined: {parlay['combined_prob']*100:.1f}% | {parlay['american_odds']:+d}\n"
                    st.code(text, language=None)
    
    # Recommended parlay
    st.markdown("---")
    st.subheader("🎯 Recommended Parlay")
    
    # Find optimal size (highest prob > 30%)
    best_size = 2
    for size in [3, 4, 5]:
        if size in parlays_dict:
            if parlays_dict[size]["combined_prob"] > 0.30:
                best_size = size
    
    if best_size in parlays_dict:
        rec = parlays_dict[best_size]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Legs", best_size)
        col2.metric("Probability", f"{rec['combined_prob']*100:.1f}%")
        col3.metric("Odds", f"{rec['american_odds']:+d}")
        col4.metric("Payout", f"${rec['payout_per_100'] * unit_size / 100:.0f}")
        
        rec_data = []
        for leg in rec["legs"]:
            p = leg["player"]
            rec_data.append({
                "Player": p["name"],
                "Score": leg["parlay_score"],
                "Tags": leg["tags"],
                "Model%": f"{leg[prob_key]:.1f}%",
                "Hit%": f"{p[f'hit_rate_{threshold}plus']:.0f}%",
            })
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

def show_results_tracker(threshold: int):
    """Results tracking tab."""
    st.header("📈 Results Tracker")
    
    # Date selection
    col1, col2 = st.columns([2, 1])
    with col1:
        check_date = st.date_input("Check date:", value=get_est_datetime().date() - timedelta(days=1))
        check_date_str = check_date.strftime("%Y-%m-%d")
    
    with col2:
        if st.button("🔄 Fetch Results", type="primary"):
            status = st.container()
            fetch_results(check_date_str, threshold, status)
    
    # Show results if available
    if check_date_str in st.session_state.saved_picks:
        picks = st.session_state.saved_picks[check_date_str]
        has_results = any(p.get("actual_sog") is not None for p in picks)
        
        if has_results:
            hits = sum(1 for p in picks if p.get("hit") == 1)
            total = len([p for p in picks if p.get("actual_sog") is not None])
            
            st.subheader(f"Results for {check_date_str}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Hits", hits)
            col2.metric("Total", total)
            col3.metric("Win Rate", f"{hits/total*100:.1f}%" if total > 0 else "N/A")
            
            # Results by grade
            grades = {}
            for p in picks:
                if p.get("actual_sog") is not None:
                    grade = p.get("parlay_grade", "?")
                    if grade not in grades:
                        grades[grade] = {"hits": 0, "total": 0}
                    grades[grade]["total"] += 1
                    if p.get("hit") == 1:
                        grades[grade]["hits"] += 1
            
            grade_data = []
            for grade in ["A+", "A", "B+", "B", "C", "D"]:
                if grade in grades:
                    g = grades[grade]
                    grade_data.append({
                        "Grade": grade,
                        "Hits": g["hits"],
                        "Total": g["total"],
                        "Win%": f"{g['hits']/g['total']*100:.0f}%" if g["total"] > 0 else "N/A"
                    })
            
            if grade_data:
                st.dataframe(pd.DataFrame(grade_data), use_container_width=True, hide_index=True)
            
            # Detailed results
            results_data = []
            for p in picks:
                if p.get("actual_sog") is not None:
                    results_data.append({
                        "Player": p["player"],
                        "Grade": p.get("parlay_grade", "?"),
                        "Model%": f"{p.get('model_prob', 0):.1f}%",
                        "Actual": p.get("actual_sog", "?"),
                        "Hit": "✅" if p.get("hit") == 1 else "❌",
                    })
            
            st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    
    # Historical summary
    st.markdown("---")
    st.subheader("📊 Historical Performance")
    
    history = st.session_state.results_history
    if history:
        total_hits = 0
        total_picks = 0
        by_grade = {}
        
        for date_str, picks in history.items():
            for p in picks:
                if p.get("actual_sog") is not None:
                    total_picks += 1
                    if p.get("hit") == 1:
                        total_hits += 1
                    
                    grade = p.get("parlay_grade", "?")
                    if grade not in by_grade:
                        by_grade[grade] = {"hits": 0, "total": 0}
                    by_grade[grade]["total"] += 1
                    if p.get("hit") == 1:
                        by_grade[grade]["hits"] += 1
        
        if total_picks > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Picks", total_picks)
            col2.metric("Total Hits", total_hits)
            col3.metric("Overall Win%", f"{total_hits/total_picks*100:.1f}%")
            
            # By grade
            hist_data = []
            for grade in ["A+", "A", "B+", "B", "C", "D"]:
                if grade in by_grade:
                    g = by_grade[grade]
                    hist_data.append({
                        "Grade": grade,
                        "Hits": g["hits"],
                        "Total": g["total"],
                        "Win%": f"{g['hits']/g['total']*100:.1f}%" if g["total"] > 0 else "N/A"
                    })
            
            st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)
    else:
        st.info("No historical results yet. Run analysis and fetch results to build history.")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer V6")
    st.caption("Comprehensive analysis with TOI, SOG/60, and historical tracking")
    
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
        
        run_analysis_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"V6.0 | {get_est_datetime().strftime('%I:%M %p EST')}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Results",
        "🎯 Tiered Breakdown", 
        "🎰 Parlays",
        "📈 Results Tracker"
    ])
    
    with tab1:
        if run_analysis_btn:
            status_container = st.container()
            plays = run_analysis(date_str, threshold, status_container)
            st.session_state.plays = plays
        
        if st.session_state.plays:
            show_all_results(st.session_state.plays, threshold, date_str)
        else:
            st.info("👈 Click **Run Analysis** to start")
            games = get_todays_schedule(date_str)
            if games:
                st.subheader(f"📅 Games: {date_str}")
                for g in games:
                    st.write(f"**{g['away_team']}** @ **{g['home_team']}** - {g['time']}")
    
    with tab2:
        if st.session_state.plays:
            show_tiered_breakdown(st.session_state.plays, threshold)
        else:
            st.info("Run analysis first")
    
    with tab3:
        if st.session_state.plays:
            show_parlays_tab(st.session_state.plays, threshold, unit_size)
        else:
            st.info("Run analysis first")
    
    with tab4:
        show_results_tracker(threshold)

if __name__ == "__main__":
    main()
