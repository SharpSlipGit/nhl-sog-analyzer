#!/usr/bin/env python3
"""
NHL Shots on Goal - ULTIMATE ANALYZER V5.1
==========================================
NEW IN V5.1 (Rate-Based Revolution):
- TOI (Time on Ice) tracking from NHL API
- SOG per 60 minutes (the key rate metric)
- SOG per shift  
- Average shifts per game
- Average shift length
- Shutout rate (0 SOG games) - the real risk metric
- Cushion rate (threshold + 1)
- Recent shutout detection (L5/L10)
- Probability caps for high-risk players
- Enhanced variance penalties
- TOI trend detection (role changes)

Based on Day 1 analysis - addresses overconfidence issues.

Usage: streamlit run nhl_sog_ultimate_v5_1.py
"""

import streamlit as st
import requests
import time
import math
import json
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from itertools import combinations
import pytz
import statistics

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NHL SOG Ultimate V5.1",
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
MIN_GAMES = 10
REQUEST_DELAY = 0.08
EST = pytz.timezone('US/Eastern')

# Matchup grades - teams that ALLOW these shots/game
MATCHUP_GRADES = {
    "A+": 33.0, "A": 32.0, "B+": 31.0, "B": 30.0,
    "C+": 29.0, "C": 28.0, "D": 27.0, "F": 0.0,
}

# Parlay config
PARLAY_CONFIG = {
    2: {"name": "Power Pair", "unit_size": 1.0, "expected_hit": 70},
    3: {"name": "Triple Threat", "unit_size": 1.0, "expected_hit": 55},
    4: {"name": "Four Banger", "unit_size": 0.75, "expected_hit": 45},
    5: {"name": "High Five", "unit_size": 0.5, "expected_hit": 35},
}

# V5.1: Probability caps
PROB_CAPS = {
    "max_absolute": 94.0,
    "high_shutout": 82.0,      # shutout_rate > 8%
    "high_variance": 85.0,     # std_dev > 2.0
    "recent_shutout": 80.0,    # shutout in L5
    "toi_dropping": 85.0,      # TOI down >15%
    "small_sample": 88.0,      # <20 games
}

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class TeamDefense:
    team_abbrev: str
    team_name: str
    games_played: int
    shots_allowed_per_game: float
    shots_allowed_L5: float
    pk_shots_allowed: float
    rank: int = 0
    grade: str = ""
    recent_trend: str = ""

@dataclass 
class PlayerStats:
    """V5.1 Enhanced with TOI and rate-based metrics."""
    player_id: int
    name: str
    team: str
    position: str
    games_played: int
    
    # Hit rates
    hit_rate_2plus: float
    hit_rate_3plus: float
    hit_rate_4plus: float
    
    # Volume metrics
    avg_sog: float
    last_5_avg: float
    last_10_avg: float
    std_dev: float
    floor: int
    ceiling: int
    current_streak: int
    
    # Home/away
    home_avg: float
    away_avg: float
    
    # Power play
    pp_toi: float = 0.0
    is_pp1: bool = False
    
    # Raw data
    all_shots: List[int] = field(default_factory=list)
    opponents: List[str] = field(default_factory=list)
    
    # =========================================
    # V5.1 NEW: Time on Ice metrics
    # =========================================
    avg_toi: float = 0.0              # Average TOI in minutes
    l5_toi: float = 0.0               # L5 average TOI
    l10_toi: float = 0.0              # L10 average TOI
    toi_trend: str = ""               # "📈 UP", "📉 DOWN", "➡️ STABLE"
    all_toi: List[float] = field(default_factory=list)
    
    # =========================================
    # V5.1 NEW: Rate-based metrics (THE KEY STUFF)
    # =========================================
    sog_per_60: float = 0.0           # Shots per 60 minutes
    l5_sog_per_60: float = 0.0        # L5 shots per 60
    l10_sog_per_60: float = 0.0       # L10 shots per 60
    
    # =========================================
    # V5.1 NEW: Shift metrics
    # =========================================
    avg_shifts: float = 0.0           # Average shifts per game
    l5_shifts: float = 0.0            # L5 average shifts
    sog_per_shift: float = 0.0        # Shots per shift
    avg_shift_length: float = 0.0     # Average shift length in seconds
    all_shifts: List[int] = field(default_factory=list)
    
    # =========================================
    # V5.1 NEW: Risk metrics
    # =========================================
    shutout_rate: float = 0.0         # % of games with 0 SOG
    cushion_rate_2: float = 0.0       # % with 3+ SOG (cushion for O1.5)
    cushion_rate_3: float = 0.0       # % with 4+ SOG (cushion for O2.5)
    cushion_rate_4: float = 0.0       # % with 5+ SOG (cushion for O3.5)
    l10_shutouts: int = 0
    l5_shutouts: int = 0
    percentile_10: int = 0            # 10th percentile SOG
    percentile_25: int = 0            # 25th percentile SOG

@dataclass
class BettingPlay:
    player: PlayerStats
    opponent: str
    opponent_defense: TeamDefense
    home_away: str
    game_time: str
    game_id: str
    h2h_avg: float
    h2h_games: int
    prob_2plus: float
    prob_3plus: float
    prob_4plus: float
    confidence: float
    tier: str
    trend: str
    implied_odds: int
    edge_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    prob_capped: bool = False
    cap_reason: str = ""
    projected_toi: float = 0.0
    projected_sog_rate: float = 0.0

@dataclass
class ParlayLeg:
    player_name: str
    team: str
    opponent: str
    threshold: float
    our_prob: float
    confidence: int
    tier: str
    game_id: str
    is_pp1: bool

@dataclass
class Parlay:
    legs: List[ParlayLeg]
    combined_prob: float
    combined_odds: int
    unit_size: float
    category: str
    risk_level: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_date():
    return datetime.now(EST).strftime("%Y-%m-%d")

def get_est_datetime():
    return datetime.now(EST)

def parse_toi(toi_str: str) -> float:
    """Parse TOI string (MM:SS) to minutes as float."""
    try:
        if not toi_str or toi_str == "0:00":
            return 0.0
        parts = toi_str.split(":")
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
        return minutes + seconds / 60.0
    except:
        return 0.0

def implied_prob_to_american(prob: float) -> int:
    if prob <= 0:
        return 10000
    if prob >= 1:
        return -10000
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

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

def get_trend_emoji(l5: float, season: float) -> str:
    diff = l5 - season
    if diff >= 0.5:
        return "🔥 HOT"
    elif diff <= -0.5:
        return "❄️ COLD"
    return "➡️ STEADY"

def get_toi_trend(l5_toi: float, avg_toi: float) -> str:
    """Determine if player's ice time is trending."""
    if avg_toi == 0:
        return "❓"
    pct_change = (l5_toi - avg_toi) / avg_toi * 100
    if pct_change >= 8:
        return "📈 UP"
    elif pct_change <= -8:
        return "📉 DOWN"
    return "➡️ STABLE"

def calculate_percentile(data: List[int], percentile: int) -> int:
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * percentile / 100)
    idx = max(0, min(idx, len(sorted_data) - 1))
    return sorted_data[idx]

def calculate_parlay_odds(probs: List[float]) -> Tuple[float, int]:
    combined_prob = 1.0
    for p in probs:
        combined_prob *= p
    american_odds = implied_prob_to_american(combined_prob)
    return combined_prob, american_odds

def calculate_parlay_payout(odds: int, stake: float = 100) -> float:
    if odds > 0:
        return stake + (stake * odds / 100)
    else:
        return stake + (stake * 100 / abs(odds))

# ============================================================================
# V5.1: PROBABILITY CAPPING
# ============================================================================
def apply_probability_cap(raw_prob: float, player: PlayerStats) -> Tuple[float, bool, str]:
    """Cap probabilities based on risk factors."""
    max_prob = PROB_CAPS["max_absolute"]
    reasons = []
    
    if player.shutout_rate > 8:
        max_prob = min(max_prob, PROB_CAPS["high_shutout"])
        reasons.append(f"shutout={player.shutout_rate:.0f}%")
    
    if player.std_dev > 2.0:
        max_prob = min(max_prob, PROB_CAPS["high_variance"])
        reasons.append(f"σ={player.std_dev:.1f}")
    
    if player.l5_shutouts >= 1:
        max_prob = min(max_prob, PROB_CAPS["recent_shutout"])
        reasons.append(f"L5_shutout")
    
    if player.avg_toi > 0 and player.l5_toi < player.avg_toi * 0.85:
        max_prob = min(max_prob, PROB_CAPS["toi_dropping"])
        reasons.append(f"TOI↓")
    
    if player.games_played < 20:
        max_prob = min(max_prob, PROB_CAPS["small_sample"])
        reasons.append(f"n={player.games_played}")
    
    if raw_prob > max_prob:
        return max_prob, True, ", ".join(reasons)
    
    return raw_prob, False, ""

# ============================================================================
# V5.1: TIER SYSTEM
# ============================================================================
def get_tier(confidence: float, hit_rate: float, player: PlayerStats) -> str:
    """Enhanced tier with consistency requirements."""
    # LOCK: Exceptional consistency
    if (confidence >= 75 and 
        hit_rate >= 85 and 
        player.shutout_rate <= 5 and
        player.std_dev <= 1.5 and
        player.l5_shutouts == 0):
        return "🔒 LOCK"
    
    # STRONG: Good consistency
    elif (confidence >= 65 and 
          hit_rate >= 80 and
          player.shutout_rate <= 10 and
          player.l5_shutouts <= 1):
        return "✅ STRONG"
    
    # SOLID: Baseline
    elif (confidence >= 55 and 
          hit_rate >= 75 and
          player.shutout_rate <= 15):
        return "📊 SOLID"
    
    # RISKY
    elif confidence >= 45:
        return "⚠️ RISKY"
    
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
                    })
        return games
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return []

@st.cache_data(ttl=600)
def get_all_teams() -> List[Dict]:
    url = f"{NHL_WEB_API}/standings/now"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [{"abbrev": t.get("teamAbbrev", {}).get("default", ""),
                 "name": t.get("teamName", {}).get("default", "")} 
                for t in data.get("standings", [])]
    except:
        return []

def get_team_roster(team_abbrev: str) -> List[Dict]:
    url = f"{NHL_WEB_API}/roster/{team_abbrev}/current"
    try:
        resp = requests.get(url, timeout=15)
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

def get_player_advanced_stats(player_id: int) -> Dict:
    try:
        url = f"{NHL_WEB_API}/player/{player_id}/landing"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        season_stats = {}
        for season in data.get("seasonTotals", []):
            if str(season.get("season")) == SEASON and season.get("gameTypeId") == GAME_TYPE:
                season_stats = season
                break
        
        pp_toi = season_stats.get("powerPlayToi", "00:00")
        gp = season_stats.get("gamesPlayed", 1)
        
        try:
            parts = pp_toi.split(":")
            total_mins = int(parts[0]) + int(parts[1]) / 60
            pp_toi_per_game = total_mins / gp if gp > 0 else 0
        except:
            pp_toi_per_game = 0
        
        return {"pp_toi_per_game": round(pp_toi_per_game, 2)}
    except:
        return {"pp_toi_per_game": 0}

def get_team_defense_stats(teams_to_fetch: set, progress_callback=None) -> Dict[str, TeamDefense]:
    all_teams = get_all_teams()
    team_defense = {}
    
    for i, team in enumerate(all_teams):
        abbrev = team["abbrev"]
        if progress_callback:
            progress_callback((i + 1) / len(all_teams), f"Fetching {abbrev}...")
        
        try:
            url = f"{NHL_WEB_API}/club-schedule-season/{abbrev}/{SEASON}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            completed = [g for g in data.get("games", []) 
                        if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
            
            if not completed:
                team_defense[abbrev] = TeamDefense(
                    team_abbrev=abbrev, team_name=team["name"],
                    games_played=0, shots_allowed_per_game=30.0,
                    shots_allowed_L5=30.0, pk_shots_allowed=8.0,
                    grade="C", recent_trend="➡️ STABLE"
                )
                continue
            
            recent = completed[-20:]
            recent_5 = completed[-5:] if len(completed) >= 5 else completed
            
            sa_all = []
            sa_L5 = []
            
            for game in recent:
                try:
                    box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
                    box_resp = requests.get(box_url, timeout=10)
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
                        sa = away_sog if abbrev == home_abbrev else home_sog
                        if sa > 0:
                            sa_all.append(sa)
                            if game in recent_5:
                                sa_L5.append(sa)
                    
                    time.sleep(0.03)
                except:
                    continue
            
            if sa_all:
                sa_pg = statistics.mean(sa_all)
                sa_L5_avg = statistics.mean(sa_L5) if sa_L5 else sa_pg
            else:
                sa_pg = 30.0
                sa_L5_avg = 30.0
            
            if sa_L5_avg < sa_pg - 2:
                trend = "🔒 TIGHTENING"
            elif sa_L5_avg > sa_pg + 2:
                trend = "📈 LOOSENING"
            else:
                trend = "➡️ STABLE"
            
            team_defense[abbrev] = TeamDefense(
                team_abbrev=abbrev,
                team_name=team["name"],
                games_played=len(sa_all),
                shots_allowed_per_game=round(sa_pg, 2),
                shots_allowed_L5=round(sa_L5_avg, 2),
                pk_shots_allowed=round(sa_pg * 0.25, 2),
                grade=get_grade(sa_pg),
                recent_trend=trend
            )
            
        except:
            team_defense[abbrev] = TeamDefense(
                team_abbrev=abbrev, team_name=team["name"],
                games_played=0, shots_allowed_per_game=30.0,
                shots_allowed_L5=30.0, pk_shots_allowed=8.0,
                grade="C", recent_trend="➡️ STABLE"
            )
        
        time.sleep(REQUEST_DELAY)
    
    sorted_teams = sorted(team_defense.values(), key=lambda x: x.shots_allowed_per_game, reverse=True)
    for rank, team in enumerate(sorted_teams, 1):
        team_defense[team.team_abbrev].rank = rank
    
    return team_defense

def get_player_stats(player_id: int, name: str, team: str, position: str) -> Optional[PlayerStats]:
    """
    V5.1: Enhanced player stats with TOI and rate-based metrics.
    """
    url = f"{NHL_WEB_API}/player/{player_id}/game-log/{SEASON}/{GAME_TYPE}"
    
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        games = resp.json().get("gameLog", [])
        
        if len(games) < MIN_GAMES:
            return None
        
        # Initialize lists
        all_shots = []
        all_toi = []
        all_shifts = []
        home_shots = []
        away_shots = []
        opponents = []
        
        for game in games:
            # Shots
            shots = game.get("shots", 0)
            if shots < 0:
                shots = 0
            all_shots.append(shots)
            
            # V5.1: TOI - parse from game log
            toi_str = game.get("toi", "0:00")
            toi_minutes = parse_toi(toi_str)
            all_toi.append(toi_minutes)
            
            # V5.1: Shifts
            shifts = game.get("shifts", 0)
            if shifts and shifts > 0:
                all_shifts.append(shifts)
            
            # Opponent
            opponents.append(game.get("opponentAbbrev", ""))
            
            # Home/away
            is_home = game.get("homeRoadFlag", "") == "H"
            if is_home:
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
        
        # Recent averages
        l5 = all_shots[:5] if len(all_shots) >= 5 else all_shots
        l10 = all_shots[:10] if len(all_shots) >= 10 else all_shots
        l5_avg = sum(l5) / len(l5)
        l10_avg = sum(l10) / len(l10)
        
        # Variance
        std = statistics.stdev(all_shots) if len(all_shots) > 1 else 0
        
        # Streak
        streak = 0
        for s in all_shots:
            if s >= 2:
                streak += 1
            else:
                break
        
        # Home/away
        home_avg = sum(home_shots) / len(home_shots) if home_shots else avg
        away_avg = sum(away_shots) / len(away_shots) if away_shots else avg
        
        # =========================================
        # V5.1: TOI calculations
        # =========================================
        avg_toi = sum(all_toi) / len(all_toi) if all_toi else 0
        l5_toi_list = all_toi[:5] if len(all_toi) >= 5 else all_toi
        l10_toi_list = all_toi[:10] if len(all_toi) >= 10 else all_toi
        l5_toi = sum(l5_toi_list) / len(l5_toi_list) if l5_toi_list else 0
        l10_toi = sum(l10_toi_list) / len(l10_toi_list) if l10_toi_list else 0
        toi_trend = get_toi_trend(l5_toi, avg_toi)
        
        # =========================================
        # V5.1: Rate-based metrics (THE KEY)
        # =========================================
        total_shots = sum(all_shots)
        total_toi = sum(all_toi)
        
        # SOG per 60 minutes
        sog_per_60 = (total_shots / total_toi * 60) if total_toi > 0 else 0
        
        # L5 SOG per 60
        l5_shots_sum = sum(l5)
        l5_toi_sum = sum(l5_toi_list)
        l5_sog_per_60 = (l5_shots_sum / l5_toi_sum * 60) if l5_toi_sum > 0 else 0
        
        # L10 SOG per 60
        l10_shots_sum = sum(l10)
        l10_toi_sum = sum(l10_toi_list)
        l10_sog_per_60 = (l10_shots_sum / l10_toi_sum * 60) if l10_toi_sum > 0 else 0
        
        # =========================================
        # V5.1: Shift metrics
        # =========================================
        avg_shifts = sum(all_shifts) / len(all_shifts) if all_shifts else 0
        l5_shifts_list = all_shifts[:5] if len(all_shifts) >= 5 else all_shifts
        l5_shifts = sum(l5_shifts_list) / len(l5_shifts_list) if l5_shifts_list else 0
        
        total_shifts = sum(all_shifts) if all_shifts else 0
        sog_per_shift = total_shots / total_shifts if total_shifts > 0 else 0
        
        # Average shift length in seconds
        avg_shift_length = (total_toi * 60 / total_shifts) if total_shifts > 0 else 0
        
        # =========================================
        # V5.1: Risk metrics
        # =========================================
        # Shutout rate
        shutout_rate = sum(1 for s in all_shots if s == 0) / gp * 100
        
        # Recent shutouts
        l5_shutouts = sum(1 for s in l5 if s == 0)
        l10_shutouts = sum(1 for s in l10 if s == 0)
        
        # Cushion rates
        cushion_rate_2 = sum(1 for s in all_shots if s >= 3) / gp * 100
        cushion_rate_3 = sum(1 for s in all_shots if s >= 4) / gp * 100
        cushion_rate_4 = sum(1 for s in all_shots if s >= 5) / gp * 100
        
        # Percentiles
        percentile_10 = calculate_percentile(all_shots, 10)
        percentile_25 = calculate_percentile(all_shots, 25)
        
        # PP stats
        adv_stats = get_player_advanced_stats(player_id)
        pp_toi = adv_stats.get("pp_toi_per_game", 0)
        is_pp1 = pp_toi >= 2.0
        
        return PlayerStats(
            player_id=player_id, name=name, team=team, position=position,
            games_played=gp,
            hit_rate_2plus=round(hit_2, 1),
            hit_rate_3plus=round(hit_3, 1),
            hit_rate_4plus=round(hit_4, 1),
            avg_sog=round(avg, 2),
            last_5_avg=round(l5_avg, 2),
            last_10_avg=round(l10_avg, 2),
            std_dev=round(std, 2),
            floor=min(all_shots),
            ceiling=max(all_shots),
            current_streak=streak,
            home_avg=round(home_avg, 2),
            away_avg=round(away_avg, 2),
            pp_toi=pp_toi,
            is_pp1=is_pp1,
            all_shots=all_shots,
            opponents=opponents,
            # V5.1 TOI
            avg_toi=round(avg_toi, 1),
            l5_toi=round(l5_toi, 1),
            l10_toi=round(l10_toi, 1),
            toi_trend=toi_trend,
            all_toi=all_toi,
            # V5.1 Rate metrics
            sog_per_60=round(sog_per_60, 2),
            l5_sog_per_60=round(l5_sog_per_60, 2),
            l10_sog_per_60=round(l10_sog_per_60, 2),
            # V5.1 Shift metrics
            avg_shifts=round(avg_shifts, 1),
            l5_shifts=round(l5_shifts, 1),
            sog_per_shift=round(sog_per_shift, 3),
            avg_shift_length=round(avg_shift_length, 1),
            all_shifts=all_shifts,
            # V5.1 Risk metrics
            shutout_rate=round(shutout_rate, 1),
            cushion_rate_2=round(cushion_rate_2, 1),
            cushion_rate_3=round(cushion_rate_3, 1),
            cushion_rate_4=round(cushion_rate_4, 1),
            l10_shutouts=l10_shutouts,
            l5_shutouts=l5_shutouts,
            percentile_10=percentile_10,
            percentile_25=percentile_25,
        )
    except Exception as e:
        return None

# ============================================================================
# V5.1: ENHANCED CONFIDENCE CALCULATIONS
# ============================================================================
def calculate_adjusted_lambda(player: PlayerStats, opp_def: TeamDefense, is_home: bool) -> float:
    """
    V5.1: Rate-based projection using SOG/60 * expected TOI.
    """
    # Rate-based projection if we have TOI data
    if player.avg_toi > 0 and player.sog_per_60 > 0:
        # Weighted rate (recent weighted more)
        weighted_rate = (player.l5_sog_per_60 * 0.4) + (player.l10_sog_per_60 * 0.3) + (player.sog_per_60 * 0.3)
        
        # Project TOI
        if player.toi_trend == "📈 UP":
            expected_toi = player.l5_toi * 1.02
        elif player.toi_trend == "📉 DOWN":
            expected_toi = player.l5_toi * 0.98
        else:
            expected_toi = (player.l5_toi * 0.6) + (player.avg_toi * 0.4)
        
        # Base from rate
        base = (weighted_rate / 60) * expected_toi
    else:
        # Fallback
        base = (player.last_5_avg * 0.4) + (player.last_10_avg * 0.3) + (player.avg_sog * 0.3)
    
    # Home/away
    if is_home and player.home_avg > 0:
        home_factor = player.home_avg / player.avg_sog if player.avg_sog > 0 else 1.0
    elif not is_home and player.away_avg > 0:
        home_factor = player.away_avg / player.avg_sog if player.avg_sog > 0 else 1.0
    else:
        home_factor = 1.02 if is_home else 0.98
    
    # Opponent
    opp_factor = opp_def.shots_allowed_per_game / 30.0 if opp_def.shots_allowed_per_game > 0 else 1.0
    if opp_def.shots_allowed_L5 > 0:
        recent_factor = opp_def.shots_allowed_L5 / 30.0
        opp_factor = (opp_factor * 0.6) + (recent_factor * 0.4)
    
    # PP
    pp_factor = 1.0
    if player.is_pp1:
        pp_factor = 1.15
    elif player.pp_toi > 1.0:
        pp_factor = 1.08
    
    return base * home_factor * opp_factor * pp_factor

def calculate_confidence(player: PlayerStats, opp: TeamDefense, is_home: bool,
                         threshold: int = 2) -> Tuple[float, List[str], List[str]]:
    """V5.1: Enhanced confidence with rate and risk metrics."""
    edges = []
    risks = []
    score = 50
    
    hit_rate = player.hit_rate_2plus if threshold == 2 else player.hit_rate_3plus if threshold == 3 else player.hit_rate_4plus
    cushion = player.cushion_rate_2 if threshold == 2 else player.cushion_rate_3 if threshold == 3 else player.cushion_rate_4
    
    # HIT RATE
    if hit_rate >= 90:
        score += 25
        edges.append(f"Elite {hit_rate}% hit rate")
    elif hit_rate >= 85:
        score += 20
        edges.append(f"Strong {hit_rate}% hit rate")
    elif hit_rate >= 80:
        score += 15
    elif hit_rate >= 75:
        score += 10
    elif hit_rate < 70:
        score -= 15
        risks.append(f"Low {hit_rate}% hit rate")
    
    # CUSHION RATE
    if cushion >= 75:
        score += 10
        edges.append(f"🛡️ {cushion:.0f}% cushion rate")
    elif cushion >= 60:
        score += 5
    elif cushion < 40:
        score -= 5
        risks.append(f"Low cushion ({cushion:.0f}%)")
    
    # SHUTOUT RATE (key V5.1 metric)
    if player.shutout_rate <= 3:
        score += 10
        edges.append(f"Rarely shutout ({player.shutout_rate:.0f}%)")
    elif player.shutout_rate <= 6:
        score += 5
    elif player.shutout_rate > 12:
        score -= 15
        risks.append(f"🚨 High shutout rate ({player.shutout_rate:.0f}%)")
    elif player.shutout_rate > 8:
        score -= 8
        risks.append(f"⚠️ Elevated shutout ({player.shutout_rate:.0f}%)")
    
    # RECENT SHUTOUTS
    if player.l5_shutouts >= 2:
        score -= 20
        risks.append(f"🚨 {player.l5_shutouts} shutouts in L5!")
    elif player.l5_shutouts == 1:
        score -= 10
        risks.append(f"⚠️ Shutout in L5")
    elif player.l10_shutouts >= 2:
        score -= 8
        risks.append(f"⚠️ {player.l10_shutouts} shutouts in L10")
    
    # FORM
    form_diff = player.last_5_avg - player.avg_sog
    if form_diff >= 1.0:
        score += 15
        edges.append(f"🔥 Hot (L5: {player.last_5_avg})")
    elif form_diff >= 0.5:
        score += 10
    elif form_diff <= -1.0:
        score -= 15
        risks.append(f"❄️ Cold (L5: {player.last_5_avg})")
    elif form_diff <= -0.5:
        score -= 10
    
    # TOI TREND
    if player.avg_toi > 0:
        if player.toi_trend == "📈 UP":
            score += 8
            edges.append(f"📈 TOI up (L5: {player.l5_toi:.0f}m)")
        elif player.toi_trend == "📉 DOWN":
            score -= 10
            risks.append(f"📉 TOI down (L5: {player.l5_toi:.0f}m)")
    
    # SOG/60 RATE
    if player.sog_per_60 >= 10.0:
        score += 10
        edges.append(f"⚡ Elite rate ({player.sog_per_60:.1f}/60)")
    elif player.sog_per_60 >= 8.0:
        score += 5
        edges.append(f"High rate ({player.sog_per_60:.1f}/60)")
    elif player.sog_per_60 < 5.0 and player.sog_per_60 > 0:
        score -= 5
        risks.append(f"Low rate ({player.sog_per_60:.1f}/60)")
    
    # Rate trend
    if player.sog_per_60 > 0 and player.l5_sog_per_60 > player.sog_per_60 * 1.15:
        score += 5
        edges.append(f"Rate trending up")
    elif player.sog_per_60 > 0 and player.l5_sog_per_60 < player.sog_per_60 * 0.85:
        score -= 5
        risks.append(f"Rate trending down")
    
    # MATCHUP
    if opp.grade in ["A+", "A"]:
        score += 15
        edges.append(f"Soft matchup ({opp.team_abbrev}: {opp.shots_allowed_per_game} SA/G)")
    elif opp.grade == "B+":
        score += 8
    elif opp.grade in ["D", "F"]:
        score -= 15
        risks.append(f"Tough matchup ({opp.team_abbrev})")
    
    if "LOOSENING" in opp.recent_trend:
        score += 8
        edges.append(f"Defense loosening")
    elif "TIGHTENING" in opp.recent_trend:
        score -= 8
        risks.append("Defense tightening")
    
    # PP
    if player.is_pp1:
        score += 10
        edges.append(f"⚡ PP1 ({player.pp_toi:.1f} min/g)")
    elif player.pp_toi > 1.0:
        score += 5
    
    # VARIANCE
    if player.std_dev < 0.8:
        score += 12
        edges.append(f"Very consistent (σ={player.std_dev})")
    elif player.std_dev < 1.2:
        score += 8
    elif player.std_dev > 2.5:
        score -= 18
        risks.append(f"🎲 Extreme variance (σ={player.std_dev})")
    elif player.std_dev > 2.0:
        score -= 12
        risks.append(f"High variance (σ={player.std_dev})")
    elif player.std_dev > 1.8:
        score -= 6
    
    # FLOOR
    if player.floor >= threshold:
        score += 15
        edges.append(f"🔒 Floor={player.floor} NEVER misses!")
    elif player.floor >= 1:
        score += 8
        edges.append(f"🛡️ Never shutout")
    
    # P10 floor
    if player.percentile_10 >= threshold:
        score += 8
        edges.append(f"P10 clears ({player.percentile_10})")
    elif player.percentile_10 == 0:
        score -= 5
        risks.append(f"P10 shows shutout risk")
    
    # HOME/AWAY
    if is_home and player.home_avg > player.away_avg + 0.3:
        score += 5
        edges.append(f"Home boost")
    elif not is_home and player.away_avg < player.home_avg - 0.3:
        score -= 5
        risks.append("Road penalty")
    
    return max(0, min(100, score)), edges, risks

# ============================================================================
# PARLAY BUILDER
# ============================================================================
def build_parlay_legs(plays: List[BettingPlay], threshold: int) -> List[ParlayLeg]:
    legs = []
    for play in plays:
        prob = play.prob_2plus / 100 if threshold == 2 else play.prob_3plus / 100 if threshold == 3 else play.prob_4plus / 100
        
        legs.append(ParlayLeg(
            player_name=play.player.name,
            team=play.player.team,
            opponent=play.opponent,
            threshold=threshold - 0.5,
            our_prob=prob,
            confidence=int(play.confidence),
            tier=play.tier,
            game_id=play.game_id,
            is_pp1=play.player.is_pp1
        ))
    
    return legs

def generate_parlays(legs: List[ParlayLeg], num_legs: int, max_parlays: int = 5) -> List[Parlay]:
    if len(legs) < num_legs:
        return []
    
    sorted_legs = sorted(legs, key=lambda x: (x.confidence, x.our_prob), reverse=True)
    candidates = sorted_legs[:min(len(sorted_legs), 15)]
    all_combos = list(combinations(candidates, num_legs))
    
    parlays = []
    for combo in all_combos[:200]:
        probs = [leg.our_prob for leg in combo]
        combined_prob, combined_odds = calculate_parlay_odds(probs)
        
        tier_scores = {"🔒 LOCK": 4, "✅ STRONG": 3, "📊 SOLID": 2, "⚠️ RISKY": 1, "❌ AVOID": 0}
        avg_tier = sum(tier_scores.get(leg.tier, 0) for leg in combo) / num_legs
        
        if avg_tier >= 3.5:
            risk_level = "LOW"
        elif avg_tier >= 2.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        config = PARLAY_CONFIG.get(num_legs, PARLAY_CONFIG[3])
        
        parlay = Parlay(
            legs=list(combo),
            combined_prob=combined_prob,
            combined_odds=combined_odds,
            unit_size=config["unit_size"],
            category=config["name"],
            risk_level=risk_level
        )
        parlays.append(parlay)
    
    parlays.sort(key=lambda x: x.combined_prob, reverse=True)
    return parlays[:max_parlays]

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Ultimate V5.1")
    st.caption("Rate-based metrics: TOI, SOG/60, SOG/shift, and enhanced risk detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        today_est = get_est_datetime().date()
        selected_date = st.date_input("📅 Date", value=today_est)
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        bet_type = st.radio(
            "SOG Threshold:",
            ["Over 1.5 (2+ SOG)", "Over 2.5 (3+ SOG)", "Over 3.5 (4+ SOG)"],
            index=0
        )
        threshold = 2 if "1.5" in bet_type else 3 if "2.5" in bet_type else 4
        
        st.markdown("---")
        
        min_hit_rate = st.slider("Min Hit Rate %", 50, 95, 75)
        min_confidence = st.slider("Min Confidence", 0, 100, 40)
        
        st.markdown("---")
        
        unit_size = st.number_input("Unit Size ($)", min_value=1, max_value=1000, value=25)
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"V5.1 | {get_est_datetime().strftime('%I:%M %p EST')}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Results", 
        "🎯 Parlays", 
        "🛡️ Defense", 
        "📈 V5.1 Metrics"
    ])
    
    if 'plays' not in st.session_state:
        st.session_state.plays = []
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 2
    
    with tab1:
        if run_analysis:
            plays = run_full_analysis(date_str, threshold, min_hit_rate, min_confidence)
            st.session_state.plays = plays
            st.session_state.threshold = threshold
        elif st.session_state.plays:
            display_results(st.session_state.plays, st.session_state.threshold, date_str)
        else:
            st.info("👈 Click **Run Analysis**")
            games = get_todays_schedule(date_str)
            if games:
                st.subheader(f"📅 Games: {date_str}")
                for g in games:
                    st.write(f"**{g['away_team']}** @ **{g['home_team']}** - {g['time']}")
    
    with tab2:
        if st.session_state.plays:
            show_parlays(st.session_state.plays, st.session_state.threshold, unit_size)
        else:
            st.info("Run analysis first")
    
    with tab3:
        show_defense()
    
    with tab4:
        show_metrics_guide()

def run_full_analysis(date_str: str, threshold: int, min_hit_rate: float, 
                     min_confidence: float) -> List[BettingPlay]:
    
    games = get_todays_schedule(date_str)
    
    if not games:
        st.error("No games found!")
        return []
    
    st.subheader(f"📅 Games: {date_str}")
    game_df = pd.DataFrame([{"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]} for g in games])
    st.dataframe(game_df, use_container_width=True, hide_index=True)
    
    teams_playing = set()
    game_info = {}
    
    for game in games:
        teams_playing.add(game["away_team"])
        teams_playing.add(game["home_team"])
        game_info[game["away_team"]] = {"opponent": game["home_team"], "home_away": "AWAY", "time": game["time"], "game_id": game["id"]}
        game_info[game["home_team"]] = {"opponent": game["away_team"], "home_away": "HOME", "time": game["time"], "game_id": game["id"]}
    
    progress = st.progress(0)
    progress.progress(0.1, "Fetching defense stats...")
    
    def update_progress(pct, msg):
        progress.progress(0.1 + pct * 0.4, msg)
    
    team_defense = get_team_defense_stats(teams_playing, update_progress)
    
    progress.progress(0.55, "Fetching rosters...")
    
    all_players = []
    for team in teams_playing:
        roster = get_team_roster(team)
        all_players.extend(roster)
        time.sleep(REQUEST_DELAY)
    
    betting_plays = []
    total = len(all_players)
    
    for i, player in enumerate(all_players):
        progress.progress(0.55 + (i / total) * 0.45, f"Analyzing {player['name']}...")
        
        stats = get_player_stats(player["id"], player["name"], player["team"], player["position"])
        
        if not stats:
            time.sleep(REQUEST_DELAY)
            continue
        
        hit_rate = stats.hit_rate_2plus if threshold == 2 else stats.hit_rate_3plus if threshold == 3 else stats.hit_rate_4plus
        if hit_rate < min_hit_rate:
            time.sleep(REQUEST_DELAY)
            continue
        
        info = game_info.get(player["team"])
        if not info:
            continue
        
        opp = info["opponent"]
        opp_def = team_defense.get(opp)
        if not opp_def:
            continue
        
        is_home = info["home_away"] == "HOME"
        
        # H2H
        h2h_shots = [stats.all_shots[j] for j, o in enumerate(stats.opponents) if o == opp]
        h2h_avg = sum(h2h_shots) / len(h2h_shots) if h2h_shots else 0
        
        # Probabilities
        adj_lambda = calculate_adjusted_lambda(stats, opp_def, is_home)
        prob_2 = poisson_prob_at_least(adj_lambda, 2) * 100
        prob_3 = poisson_prob_at_least(adj_lambda, 3) * 100
        prob_4 = poisson_prob_at_least(adj_lambda, 4) * 100
        
        # Apply caps
        main_prob = prob_2 if threshold == 2 else prob_3 if threshold == 3 else prob_4
        capped_prob, was_capped, cap_reason = apply_probability_cap(main_prob, stats)
        
        if threshold == 2:
            prob_2 = capped_prob
        elif threshold == 3:
            prob_3 = capped_prob
        else:
            prob_4 = capped_prob
        
        # Confidence
        conf, edges, risks = calculate_confidence(stats, opp_def, is_home, threshold)
        
        if conf < min_confidence:
            time.sleep(REQUEST_DELAY)
            continue
        
        implied_odds = implied_prob_to_american(capped_prob / 100)
        tier = get_tier(conf, hit_rate, stats)
        
        play = BettingPlay(
            player=stats,
            opponent=opp,
            opponent_defense=opp_def,
            home_away=info["home_away"],
            game_time=info["time"],
            game_id=info["game_id"],
            h2h_avg=round(h2h_avg, 2),
            h2h_games=len(h2h_shots),
            prob_2plus=round(prob_2, 1),
            prob_3plus=round(prob_3, 1),
            prob_4plus=round(prob_4, 1),
            confidence=conf,
            tier=tier,
            trend=get_trend_emoji(stats.last_5_avg, stats.avg_sog),
            implied_odds=implied_odds,
            edge_factors=edges,
            risk_factors=risks,
            prob_capped=was_capped,
            cap_reason=cap_reason
        )
        betting_plays.append(play)
        time.sleep(REQUEST_DELAY)
    
    progress.empty()
    
    if not betting_plays:
        st.warning("No qualifying plays found!")
        return []
    
    betting_plays.sort(key=lambda x: x.confidence, reverse=True)
    display_results(betting_plays, threshold, date_str)
    
    return betting_plays

def display_results(plays: List[BettingPlay], threshold: int, date_str: str):
    st.subheader(f"🎯 Results - O{threshold - 0.5} SOG")
    
    # Summary
    col1, col2, col3, col4, col5 = st.columns(5)
    locks = len([p for p in plays if "LOCK" in p.tier])
    strong = len([p for p in plays if "STRONG" in p.tier])
    solid = len([p for p in plays if "SOLID" in p.tier])
    capped = len([p for p in plays if p.prob_capped])
    
    col1.metric("🔒 Locks", locks)
    col2.metric("✅ Strong", strong)
    col3.metric("📊 Solid", solid)
    col4.metric("🔻 Capped", capped)
    col5.metric("Total", len(plays))
    
    # Table with V5.1 columns
    data = []
    for p in plays:
        s = p.player
        hit = s.hit_rate_2plus if threshold == 2 else s.hit_rate_3plus if threshold == 3 else s.hit_rate_4plus
        prob = p.prob_2plus if threshold == 2 else p.prob_3plus if threshold == 3 else p.prob_4plus
        cush = s.cushion_rate_2 if threshold == 2 else s.cushion_rate_3 if threshold == 3 else s.cushion_rate_4
        
        data.append({
            "Tier": p.tier,
            "Player": s.name,
            "Team": s.team,
            "vs": p.opponent,
            "Hit%": f"{hit:.0f}%",
            "Cush%": f"{cush:.0f}%",
            "Shut%": f"{s.shutout_rate:.0f}%",
            "Avg": s.avg_sog,
            "L5": s.last_5_avg,
            "SOG/60": s.sog_per_60,
            "TOI": f"{s.avg_toi:.0f}m",
            "TOI📈": s.toi_trend,
            "σ": s.std_dev,
            "PP": "⚡" if s.is_pp1 else "",
            "Prob%": f"{prob:.0f}%",
            "Cap": "🔻" if p.prob_capped else "",
            "Conf": int(p.confidence),
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=100, format="%d")})
    
    # Export
    csv = df.to_csv(index=False)
    st.download_button("📥 Download", data=csv, file_name=f"nhl_sog_v51_{date_str}.csv", mime="text/csv")

def show_parlays(plays: List[BettingPlay], threshold: int, unit_size: float):
    st.header("🎯 Parlay Builder")
    
    legs = build_parlay_legs(plays, threshold)
    quality = [l for l in legs if l.confidence >= 60 and l.our_prob >= 0.75]
    
    if not quality:
        st.warning("Not enough quality plays")
        return
    
    st.success(f"**{len(quality)}** quality plays available")
    
    for num_legs, config in PARLAY_CONFIG.items():
        parlays = generate_parlays(quality, num_legs, 3)
        if not parlays:
            continue
        
        with st.expander(f"**{config['name']}** ({num_legs}-leg)", expanded=(num_legs == 2)):
            for i, parlay in enumerate(parlays, 1):
                st.markdown(f"**Option {i}** - {parlay.risk_level} Risk")
                
                col1, col2, col3 = st.columns(3)
                col1.write(f"Prob: **{parlay.combined_prob * 100:.1f}%**")
                col2.write(f"Odds: **{parlay.combined_odds:+d}**")
                col3.write(f"Bet: **${unit_size * parlay.unit_size:.0f}**")
                
                leg_data = [{"Player": l.player_name, "Team": l.team, "Prob": f"{l.our_prob*100:.0f}%", "Conf": l.confidence} for l in parlay.legs]
                st.dataframe(pd.DataFrame(leg_data), hide_index=True, use_container_width=True)
                st.markdown("---")

def show_defense():
    st.header("🛡️ Team Defense")
    
    if st.button("🔄 Refresh"):
        with st.spinner("Loading..."):
            all_teams = get_all_teams()
            team_defense = get_team_defense_stats(set(t["abbrev"] for t in all_teams))
            
            sorted_def = sorted(team_defense.values(), key=lambda x: x.shots_allowed_per_game, reverse=True)
            
            data = [{"Rank": i, "Team": t.team_abbrev, "SA/G": t.shots_allowed_per_game, "L5": t.shots_allowed_L5, "Grade": t.grade, "Trend": t.recent_trend} for i, t in enumerate(sorted_def, 1)]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

def show_metrics_guide():
    st.header("📈 V5.1 Metrics Guide")
    
    st.markdown("""
    ## Rate-Based Metrics (The Key Improvement)
    
    | Metric | Formula | Why It Matters |
    |--------|---------|----------------|
    | **SOG/60** | (Shots / TOI) × 60 | Normalizes for ice time |
    | **SOG/Shift** | Shots / Shifts | Efficiency per deployment |
    | **Avg Shift Length** | (TOI × 60) / Shifts | Star players = longer shifts |
    
    ### Example: Two Players with 3.0 Avg SOG
    
    | Player | TOI | SOG/60 | Better Bet? |
    |--------|-----|--------|-------------|
    | Player A | 22 min | 8.2 | ❌ No (low rate) |
    | Player B | 14 min | 12.9 | ✅ Yes (high rate) |
    
    Player B shoots at a much higher rate - if they get normal ice time, they'll exceed their average.
    
    ---
    
    ## Risk Metrics
    
    | Metric | What It Tells You |
    |--------|-------------------|
    | **Shutout Rate** | % of games with 0 SOG (true danger) |
    | **Cushion Rate** | % clearing threshold+1 (comfortable wins) |
    | **L5 Shutouts** | Recent shutouts (momentum indicator) |
    | **P10 Floor** | 10th percentile (robust floor) |
    
    ---
    
    ## TOI Trend
    
    | Trend | Meaning |
    |-------|---------|
    | 📈 UP | L5 TOI > Season by 8%+ (role expanding) |
    | 📉 DOWN | L5 TOI < Season by 8%+ (role shrinking) |
    | ➡️ STABLE | Within ±8% |
    
    ---
    
    ## Probability Caps
    
    V5.1 caps probability to prevent overconfidence:
    
    | Condition | Max Prob |
    |-----------|----------|
    | Shutout rate > 8% | 82% |
    | Std dev > 2.0 | 85% |
    | Shutout in L5 | 80% |
    | TOI dropping | 85% |
    | < 20 games | 88% |
    | Absolute max | 94% |
    """)

if __name__ == "__main__":
    main()
