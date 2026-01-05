#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer v3.1
===============================
Changes from v3:
- Sorted by hit rate (not confidence)
- Tooltips on all columns
- "How It's Calculated" sections explaining all variables
"""

import streamlit as st
import requests
import time
import math
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from itertools import combinations
import pytz
import statistics

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NHL SOG Analyzer v3.1",
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
REQUEST_DELAY = 0.05
EST = pytz.timezone('US/Eastern')

MATCHUP_GRADES = {
    "A+": 33.0, "A": 32.0, "B+": 31.0, "B": 30.0,
    "C+": 29.0, "C": 28.0, "D": 27.0, "F": 0.0,
}

# Column definitions for tooltips
COLUMN_DEFINITIONS = {
    "Hit%": "Percentage of games this season where player recorded 2+ (or 3+/4+) shots on goal",
    "Avg": "Season average shots on goal per game",
    "L5": "Average shots on goal over last 5 games (recent form)",
    "L10": "Average shots on goal over last 10 games",
    "Floor": "Minimum SOG in any game this season (0 = has been shut out)",
    "Ceil": "Maximum SOG in any game this season",
    "Prob%": "Model-calculated probability of hitting the threshold tonight",
    "Conf": "Confidence score (0-100) based on multiple factors - see 'How It's Calculated'",
    "Trend": "🔥 HOT = L5 > Avg + 0.5 | ❄️ COLD = L5 < Avg - 0.5 | ➡️ STEADY = within range",
    "Matchup": "Opponent defense grade: A+ (allows most shots) to F (allows fewest)",
    "PP": "Power Play indicator: ⚡PP1 = first unit (2+ min/game), ✓ = some PP time",
    "Status": "✅ QUALIFIED = 85%+ hit rate & not cold | ⚠️ RISK = 80-84% or cold streak",
    "Tags": "🛡️ HIGH FLOOR = never 0 SOG | ⚡ PP1 = power play 1 | 🔥 STREAK = 5+ game streak",
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
    american_odds = implied_prob_to_american(combined_prob)
    return combined_prob, american_odds

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

def get_trend(l5: float, season: float) -> str:
    diff = l5 - season
    if diff >= 0.5:
        return "🔥 HOT"
    elif diff <= -0.5:
        return "❄️ COLD"
    return "➡️ STEADY"

def is_cold(l5: float, season: float) -> bool:
    return (l5 - season) <= -0.5

def get_tags(player: Dict) -> List[str]:
    tags = []
    if player["floor"] >= 1:
        tags.append("🛡️ HIGH FLOOR")
    if player["is_pp1"]:
        tags.append("⚡ PP1")
    elif player["pp_toi"] > 0.5:
        tags.append("PP")
    if player["current_streak"] >= 5:
        tags.append(f"🔥 {player['current_streak']}G STREAK")
    return tags

def get_qualification_status(player: Dict, threshold: int = 2) -> Tuple[bool, str]:
    hit_rate = player["hit_rate_2plus"] if threshold == 2 else player["hit_rate_3plus"] if threshold == 3 else player["hit_rate_4plus"]
    cold = is_cold(player["last_5_avg"], player["avg_sog"])
    
    if hit_rate >= 85 and not cold:
        return True, "✅ QUALIFIED"
    elif hit_rate >= 80:
        reasons = []
        if hit_rate < 85:
            reasons.append(f"{hit_rate:.0f}%")
        if cold:
            reasons.append("cold")
        return False, f"⚠️ {', '.join(reasons)}"
    return False, "❌ <80%"

def calculate_confidence(player: Dict, opp_def: Dict, is_home: bool, threshold: int = 2) -> Tuple[int, Dict[str, int]]:
    """
    Calculate confidence score with breakdown of factors.
    Returns (total_score, breakdown_dict)
    """
    breakdown = {}
    score = 50  # Base score
    breakdown["Base"] = 50
    
    hit_rate = player["hit_rate_2plus"] if threshold == 2 else player["hit_rate_3plus"] if threshold == 3 else player["hit_rate_4plus"]
    
    # Hit rate factor (most important)
    if hit_rate >= 95:
        hr_score = 30
    elif hit_rate >= 90:
        hr_score = 25
    elif hit_rate >= 85:
        hr_score = 18
    elif hit_rate >= 80:
        hr_score = 10
    else:
        hr_score = 0
    score += hr_score
    breakdown["Hit Rate"] = hr_score
    
    # Form factor (L5 vs season avg)
    form_diff = player["last_5_avg"] - player["avg_sog"]
    if form_diff >= 1.0:
        form_score = 12
    elif form_diff >= 0.5:
        form_score = 6
    elif form_diff <= -1.0:
        form_score = -12
    elif form_diff <= -0.5:
        form_score = -6
    else:
        form_score = 0
    score += form_score
    breakdown["Recent Form"] = form_score
    
    # Floor factor
    if player["floor"] >= 2:
        floor_score = 8
    elif player["floor"] >= 1:
        floor_score = 4
    else:
        floor_score = -4
    score += floor_score
    breakdown["Floor"] = floor_score
    
    # Matchup factor (reduced weight)
    grade = opp_def.get("grade", "C")
    if grade in ["A+", "A"]:
        matchup_score = 6
    elif grade == "B+":
        matchup_score = 3
    elif grade in ["D", "F"]:
        matchup_score = -6
    else:
        matchup_score = 0
    score += matchup_score
    breakdown["Matchup"] = matchup_score
    
    # PP factor
    if player["is_pp1"]:
        pp_score = 6
    elif player["pp_toi"] > 1.0:
        pp_score = 3
    else:
        pp_score = 0
    score += pp_score
    breakdown["Power Play"] = pp_score
    
    # Consistency factor
    if player["std_dev"] < 1.0:
        cons_score = 4
    elif player["std_dev"] > 2.0:
        cons_score = -4
    else:
        cons_score = 0
    score += cons_score
    breakdown["Consistency"] = cons_score
    
    # Home/away
    if is_home:
        ha_score = 2
    else:
        ha_score = -2
    score += ha_score
    breakdown["Home/Away"] = ha_score
    
    return max(0, min(100, score)), breakdown

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

def get_team_defense_stats(teams_playing: set, progress_bar, status_text) -> Dict[str, Dict]:
    team_defense = {}
    all_teams = get_all_teams()
    
    teams_to_fetch = [t for t in all_teams if t["abbrev"] in teams_playing]
    total_teams = len(teams_to_fetch)
    
    for i, team in enumerate(teams_to_fetch):
        abbrev = team["abbrev"]
        status_text.text(f"🛡️ Defense: {abbrev} ({i+1}/{total_teams})")
        progress_bar.progress(int((i / total_teams) * 25))
        
        try:
            url = f"{NHL_WEB_API}/club-schedule-season/{abbrev}/{SEASON}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            completed = [g for g in data.get("games", []) 
                        if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
            
            if not completed:
                team_defense[abbrev] = {
                    "team_abbrev": abbrev, "shots_allowed_per_game": 30.0,
                    "shots_allowed_L5": 30.0, "grade": "C"
                }
                continue
            
            recent = completed[-10:]
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
                    
                    time.sleep(0.02)
                except:
                    continue
            
            sa_pg = statistics.mean(sa_all) if sa_all else 30.0
            sa_L5_avg = statistics.mean(sa_L5) if sa_L5 else sa_pg
            
            team_defense[abbrev] = {
                "team_abbrev": abbrev,
                "shots_allowed_per_game": round(sa_pg, 2),
                "shots_allowed_L5": round(sa_L5_avg, 2),
                "grade": get_grade(sa_pg)
            }
            
        except:
            team_defense[abbrev] = {
                "team_abbrev": abbrev, "shots_allowed_per_game": 30.0,
                "shots_allowed_L5": 30.0, "grade": "C"
            }
        
        time.sleep(REQUEST_DELAY)
    
    return team_defense

def get_player_stats(player_id: int, name: str, team: str, position: str) -> Optional[Dict]:
    url = f"{NHL_WEB_API}/player/{player_id}/game-log/{SEASON}/{GAME_TYPE}"
    
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        games = resp.json().get("gameLog", [])
        
        if len(games) < MIN_GAMES:
            return None
        
        all_shots = []
        home_shots = []
        away_shots = []
        opponents = []
        
        for game in games:
            shots = game.get("shots", 0)
            if shots < 0:
                shots = 0
            all_shots.append(shots)
            opponents.append(game.get("opponentAbbrev", ""))
            
            is_home = game.get("homeRoadFlag", "") == "H"
            if is_home:
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
        
        adv_stats = get_player_advanced_stats(player_id)
        pp_toi = adv_stats.get("pp_toi_per_game", 0)
        is_pp1 = pp_toi >= 2.0
        
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
            "pp_toi": pp_toi,
            "is_pp1": is_pp1,
            "all_shots": all_shots,
            "opponents": opponents
        }
    except:
        return None

# ============================================================================
# PARLAY GENERATION
# ============================================================================
def generate_best_parlay(plays: List[Dict], num_legs: int, threshold: int) -> Optional[Dict]:
    if len(plays) < num_legs:
        return None
    
    # Sort by hit rate (primary sort)
    hit_key = f"hit_rate_{threshold}plus"
    sorted_plays = sorted(plays, key=lambda x: x["player"][hit_key], reverse=True)
    
    best_legs = sorted_plays[:num_legs]
    
    probs = []
    for play in best_legs:
        prob = play[f"prob_{threshold}plus"] / 100
        probs.append(prob)
    
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
    
    hit_key = f"hit_rate_{threshold}plus"
    sorted_plays = sorted(game_plays, key=lambda x: x["player"][hit_key], reverse=True)
    
    for num_legs in range(min_legs, min(len(sorted_plays) + 1, 8)):
        legs = sorted_plays[:num_legs]
        probs = [p[f"prob_{threshold}plus"] / 100 for p in legs]
        combined_prob, american_odds = calculate_parlay_odds(probs)
        
        if american_odds >= min_odds:
            return {
                "legs": legs,
                "num_legs": num_legs,
                "combined_prob": combined_prob,
                "american_odds": american_odds,
                "payout_per_100": calculate_parlay_payout(american_odds, 100),
                "game_id": game_id
            }
    
    return None

# ============================================================================
# UI HELPER - DEFINITIONS EXPANDER
# ============================================================================
def show_definitions():
    """Show expandable definitions section."""
    with st.expander("📖 Column Definitions & How It's Calculated", expanded=False):
        
        st.markdown("### 📊 Column Definitions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            | Column | Definition |
            |--------|------------|
            | **Hit%** | % of games hitting threshold this season |
            | **Avg** | Season average SOG per game |
            | **L5** | Average SOG last 5 games |
            | **L10** | Average SOG last 10 games |
            | **Floor** | Lowest SOG in any game |
            | **Ceil** | Highest SOG in any game |
            """)
        
        with col2:
            st.markdown("""
            | Column | Definition |
            |--------|------------|
            | **Prob%** | Model probability for tonight |
            | **Conf** | Confidence score (0-100) |
            | **Trend** | 🔥 HOT / ❄️ COLD / ➡️ STEADY |
            | **Matchup** | Opponent defense grade (A+ to F) |
            | **Status** | ✅ Qualified / ⚠️ Risk |
            """)
        
        st.markdown("---")
        st.markdown("### 🧮 How Confidence Score is Calculated")
        
        st.markdown("""
        **Base Score: 50 points**
        
        | Factor | Points | Logic |
        |--------|--------|-------|
        | **Hit Rate** | -0 to +30 | 95%+ = +30, 90%+ = +25, 85%+ = +18, 80%+ = +10 |
        | **Recent Form** | -12 to +12 | L5 vs Avg: +1.0 diff = +12, +0.5 = +6, -0.5 = -6, -1.0 = -12 |
        | **Floor** | -4 to +8 | Floor ≥2 = +8, Floor ≥1 = +4, Floor 0 = -4 |
        | **Matchup** | -6 to +6 | A+/A = +6, B+ = +3, D/F = -6 |
        | **Power Play** | 0 to +6 | PP1 (2+ min) = +6, PP time = +3 |
        | **Consistency** | -4 to +4 | StdDev <1.0 = +4, >2.0 = -4 |
        | **Home/Away** | -2 to +2 | Home = +2, Away = -2 |
        
        **Final Score = Base + All Factors (capped 0-100)**
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 How Probability is Calculated")
        
        st.markdown("""
        Uses **Poisson distribution** with adjusted lambda (expected SOG):
        
        ```
        Base Lambda = (L5 × 0.4) + (L10 × 0.3) + (Season Avg × 0.3)
        
        Adjustments:
        × Home Factor (1.02 home, 0.98 away)
        × Opponent Factor (Opp SA/G ÷ 30)
        × PP Factor (1.15 for PP1, 1.08 for PP time)
        
        Final Prob = Poisson P(X ≥ threshold) using adjusted lambda
        ```
        """)
        
        st.markdown("---")
        st.markdown("### 🏷️ Tags Explained")
        
        st.markdown("""
        | Tag | Meaning |
        |-----|---------|
        | 🛡️ **HIGH FLOOR** | Player has never recorded 0 SOG this season |
        | ⚡ **PP1** | Averages 2+ minutes on Power Play per game |
        | 🔥 **X G STREAK** | Has hit 2+ SOG in X consecutive games |
        """)
        
        st.markdown("---")
        st.markdown("### ✅ Qualification Criteria")
        
        st.markdown("""
        **Qualified** (safest plays):
        - 85%+ hit rate for selected threshold
        - NOT in cold streak (L5 not more than 0.5 below season avg)
        
        **Higher Risk** (still shown, but flagged):
        - 80-84% hit rate, OR
        - Currently in cold streak
        """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer v3.1")
    st.caption("Sorted by Hit Rate | Full Definitions | All Players Shown")
    
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
            index=0,
            help="The minimum shots on goal to win the bet"
        )
        threshold = 2 if "1.5" in bet_type else 3 if "2.5" in bet_type else 4
        
        st.markdown("---")
        
        unit_size = st.number_input(
            "Unit Size ($)", 
            min_value=1, 
            max_value=1000, 
            value=25,
            help="Your standard bet amount for calculating payouts"
        )
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"Time: {get_est_datetime().strftime('%I:%M %p EST')}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Results", 
        "🎯 Best Bets", 
        "🎰 Parlays",
        "❓ Help"
    ])
    
    # Session state
    if 'all_plays' not in st.session_state:
        st.session_state.all_plays = []
    if 'games' not in st.session_state:
        st.session_state.games = []
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 2
    
    with tab1:
        if run_analysis:
            plays, games = run_full_analysis(date_str, threshold)
            st.session_state.all_plays = plays
            st.session_state.games = games
            st.session_state.threshold = threshold
        elif st.session_state.all_plays:
            display_all_results(st.session_state.all_plays, st.session_state.threshold, date_str)
        else:
            st.info("👈 Click **Run Analysis** to start")
            games = get_todays_schedule(date_str)
            if games:
                st.subheader(f"📅 Games on {date_str}")
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

def run_full_analysis(date_str: str, threshold: int) -> Tuple[List[Dict], List[Dict]]:
    
    st.subheader(f"📅 Games on {date_str}")
    games = get_todays_schedule(date_str)
    
    if not games:
        st.error("No games found!")
        return [], []
    
    game_df = pd.DataFrame([
        {"Away": g["away_team"], "Home": g["home_team"], "Time": g["time"]}
        for g in games
    ])
    st.dataframe(game_df, use_container_width=True, hide_index=True)
    
    teams_playing = set()
    game_info = {}
    
    for game in games:
        teams_playing.add(game["away_team"])
        teams_playing.add(game["home_team"])
        game_info[game["away_team"]] = {
            "opponent": game["home_team"], 
            "home_away": "AWAY", 
            "time": game["time"],
            "game_id": game["id"],
            "matchup": game["matchup"]
        }
        game_info[game["home_team"]] = {
            "opponent": game["away_team"], 
            "home_away": "HOME", 
            "time": game["time"],
            "game_id": game["id"],
            "matchup": game["matchup"]
        }
    
    st.markdown("---")
    st.subheader("📊 Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_display = st.empty()
    
    team_defense = get_team_defense_stats(teams_playing, progress_bar, status_text)
    
    progress_bar.progress(25)
    status_text.text(f"📋 Fetching rosters...")
    
    all_players = []
    for team in teams_playing:
        roster = get_team_roster(team)
        all_players.extend(roster)
        time.sleep(REQUEST_DELAY)
    
    progress_bar.progress(30)
    
    all_plays = []
    total = len(all_players)
    players_checked = 0
    players_found = 0
    
    for i, player in enumerate(all_players):
        players_checked += 1
        pct = 30 + int((i / total) * 70)
        progress_bar.progress(pct)
        status_text.text(f"🔍 {player['name']} ({player['team']})")
        stats_display.text(f"Checked: {players_checked}/{total} | Found: {players_found}")
        
        stats = get_player_stats(player["id"], player["name"], player["team"], player["position"])
        
        if not stats:
            time.sleep(REQUEST_DELAY)
            continue
        
        # Only filter: 80%+ hit rate
        hit_rate = stats["hit_rate_2plus"] if threshold == 2 else stats["hit_rate_3plus"] if threshold == 3 else stats["hit_rate_4plus"]
        
        if hit_rate < 80:
            time.sleep(REQUEST_DELAY)
            continue
        
        info = game_info.get(player["team"])
        if not info:
            continue
        
        opp = info["opponent"]
        opp_def = team_defense.get(opp, {"shots_allowed_per_game": 30.0, "grade": "C"})
        is_home = info["home_away"] == "HOME"
        
        # Calculate probability
        base = (stats["last_5_avg"] * 0.4) + (stats["last_10_avg"] * 0.3) + (stats["avg_sog"] * 0.3)
        home_factor = 1.02 if is_home else 0.98
        opp_factor = opp_def["shots_allowed_per_game"] / 30.0
        pp_factor = 1.15 if stats["is_pp1"] else (1.08 if stats["pp_toi"] > 1.0 else 1.0)
        adj_lambda = base * home_factor * opp_factor * pp_factor
        
        prob_2 = poisson_prob_at_least(adj_lambda, 2) * 100
        prob_3 = poisson_prob_at_least(adj_lambda, 3) * 100
        prob_4 = poisson_prob_at_least(adj_lambda, 4) * 100
        
        # Confidence with breakdown
        conf, conf_breakdown = calculate_confidence(stats, opp_def, is_home, threshold)
        
        # Qualification
        is_qualified, status = get_qualification_status(stats, threshold)
        
        players_found += 1
        
        play = {
            "player": stats,
            "opponent": opp,
            "opponent_defense": opp_def,
            "home_away": info["home_away"],
            "game_time": info["time"],
            "game_id": info["game_id"],
            "matchup": info["matchup"],
            "prob_2plus": round(prob_2, 1),
            "prob_3plus": round(prob_3, 1),
            "prob_4plus": round(prob_4, 1),
            "confidence": conf,
            "conf_breakdown": conf_breakdown,
            "is_qualified": is_qualified,
            "status": status,
            "trend": get_trend(stats["last_5_avg"], stats["avg_sog"]),
            "tags": get_tags(stats)
        }
        all_plays.append(play)
        time.sleep(REQUEST_DELAY)
    
    progress_bar.progress(100)
    status_text.text("✅ Complete!")
    stats_display.text(f"Total: {players_checked} checked, {players_found} with 80%+ hit rate")
    time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    stats_display.empty()
    
    # SORT BY HIT RATE (not confidence)
    hit_key = f"hit_rate_{threshold}plus"
    all_plays.sort(key=lambda x: x["player"][hit_key], reverse=True)
    
    st.success(f"Found **{len(all_plays)}** players with 80%+ hit rate!")
    display_all_results(all_plays, threshold, date_str)
    
    return all_plays, games

def display_all_results(plays: List[Dict], threshold: int, date_str: str):
    
    st.subheader(f"🎯 All Players - Over {threshold - 0.5} SOG")
    st.caption("Sorted by Hit Rate (highest first) | All players with 80%+ shown")
    
    # Show definitions
    show_definitions()
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    # Summary
    qualified = len([p for p in plays if p["is_qualified"]])
    high_floor = len([p for p in plays if p["player"]["floor"] >= 1])
    pp1_count = len([p for p in plays if p["player"]["is_pp1"]])
    hot_count = len([p for p in plays if "HOT" in p["trend"]])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", len(plays), help="All players with 80%+ hit rate")
    col2.metric("✅ Qualified", qualified, help="85%+ hit rate and not cold")
    col3.metric("🛡️ High Floor", high_floor, help="Never had 0 SOG this season")
    col4.metric("⚡ PP1", pp1_count, help="Power Play 1 (2+ min/game)")
    col5.metric("🔥 Hot", hot_count, help="L5 average above season average")
    
    # Build table
    results_data = []
    for play in plays:
        p = play["player"]
        tags = " ".join(play["tags"]) if play["tags"] else ""
        
        row = {
            "Status": play["status"],
            "Player": p["name"],
            "Tags": tags,
            "Team": p["team"],
            "vs": play["opponent"],
            "H/A": play["home_away"],
            "Hit%": f"{p[hit_key]:.1f}%",
            "Avg": p["avg_sog"],
            "L5": p["last_5_avg"],
            "Floor": p["floor"],
            "Ceil": p["ceiling"],
            "Prob%": f"{play[prob_key]:.1f}%",
            "Conf": play["confidence"],
            "Trend": play["trend"],
            "Matchup": play["opponent_defense"]["grade"],
            "Time": play["game_time"]
        }
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    
    # Display with column config for tooltips
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", help=COLUMN_DEFINITIONS["Status"]),
            "Hit%": st.column_config.TextColumn("Hit%", help=COLUMN_DEFINITIONS["Hit%"]),
            "Avg": st.column_config.NumberColumn("Avg", help=COLUMN_DEFINITIONS["Avg"]),
            "L5": st.column_config.NumberColumn("L5", help=COLUMN_DEFINITIONS["L5"]),
            "Floor": st.column_config.NumberColumn("Floor", help=COLUMN_DEFINITIONS["Floor"]),
            "Ceil": st.column_config.NumberColumn("Ceil", help=COLUMN_DEFINITIONS["Ceil"]),
            "Prob%": st.column_config.TextColumn("Prob%", help=COLUMN_DEFINITIONS["Prob%"]),
            "Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=100, help=COLUMN_DEFINITIONS["Conf"]),
            "Trend": st.column_config.TextColumn("Trend", help=COLUMN_DEFINITIONS["Trend"]),
            "Matchup": st.column_config.TextColumn("Matchup", help=COLUMN_DEFINITIONS["Matchup"]),
            "Tags": st.column_config.TextColumn("Tags", help=COLUMN_DEFINITIONS["Tags"]),
        }
    )
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, 
                       file_name=f"nhl_sog_all_{date_str}.csv", mime="text/csv")

def show_best_bets(plays: List[Dict], threshold: int):
    
    st.header("🎯 Best Bets")
    
    # Show definitions
    show_definitions()
    
    qualified = [p for p in plays if p["is_qualified"]]
    risky = [p for p in plays if not p["is_qualified"]]
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    # Qualified section
    st.subheader(f"✅ Qualified Plays ({len(qualified)})")
    st.caption("85%+ hit rate AND not in cold streak")
    
    if qualified:
        qual_data = []
        for play in qualified:
            p = play["player"]
            tags = " ".join(play["tags"]) if play["tags"] else ""
            qual_data.append({
                "Player": p["name"],
                "Tags": tags,
                "Team": p["team"],
                "vs": play["opponent"],
                "Hit%": f"{p[hit_key]:.1f}%",
                "Avg": p["avg_sog"],
                "L5": p["last_5_avg"],
                "Prob%": f"{play[prob_key]:.1f}%",
                "Conf": play["confidence"],
                "Trend": play["trend"],
            })
        st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No qualified plays. Check Higher Risk section.")
    
    st.markdown("---")
    
    # Higher risk section
    st.subheader(f"⚠️ Higher Risk Plays ({len(risky)})")
    st.caption("80-84% hit rate OR currently in cold streak")
    
    if risky:
        risk_data = []
        for play in risky:
            p = play["player"]
            tags = " ".join(play["tags"]) if play["tags"] else ""
            risk_data.append({
                "Player": p["name"],
                "Risk Factor": play["status"].replace("⚠️ ", ""),
                "Tags": tags,
                "Team": p["team"],
                "vs": play["opponent"],
                "Hit%": f"{p[hit_key]:.1f}%",
                "L5": p["last_5_avg"],
                "Avg": p["avg_sog"],
                "Prob%": f"{play[prob_key]:.1f}%",
            })
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

def show_parlays(plays: List[Dict], games: List[Dict], threshold: int, unit_size: float):
    
    st.header("🎰 Optimal Parlays")
    
    # Show definitions
    show_definitions()
    
    qualified = [p for p in plays if p["is_qualified"]]
    
    if len(qualified) < 2:
        st.warning("Need 2+ qualified plays for parlays. Using all plays.")
        qualified = plays
    
    st.success(f"Building parlays from **{len(qualified)}** players (sorted by hit rate)")
    
    # Best parlay for each leg count
    st.subheader("📊 Best Parlay by Leg Count")
    st.caption("Optimal combination for each number of legs")
    
    parlay_results = []
    max_legs = min(12, len(qualified))
    
    for num_legs in range(1, max_legs + 1):
        parlay = generate_best_parlay(qualified, num_legs, threshold)
        if parlay:
            players = ", ".join([p["player"]["name"] for p in parlay["legs"]])
            parlay_results.append({
                "Legs": num_legs,
                "Prob": f"{parlay['combined_prob'] * 100:.1f}%",
                "Odds": f"{parlay['american_odds']:+d}" if parlay['american_odds'] < 10000 else "N/A",
                f"${unit_size:.0f} Wins": f"${parlay['payout_per_100'] * unit_size / 100:.0f}",
                "Players": players[:100] + "..." if len(players) > 100 else players
            })
    
    if parlay_results:
        st.dataframe(pd.DataFrame(parlay_results), use_container_width=True, hide_index=True)
    
    # Copyable parlay text for top options
    st.markdown("### 📋 Copy-Paste Parlays")
    
    for num_legs in [2, 3, 5]:
        if num_legs <= len(qualified):
            parlay = generate_best_parlay(qualified, num_legs, threshold)
            if parlay:
                with st.expander(f"**{num_legs}-Leg Parlay** ({parlay['combined_prob']*100:.0f}% | {parlay['american_odds']:+d})"):
                    copy_text = f"🏒 NHL SOG {num_legs}-Leg Parlay:\n"
                    for p in parlay["legs"]:
                        copy_text += f"• {p['player']['name']} O{threshold-0.5} SOG ({p['player'][f'hit_rate_{threshold}plus']:.0f}%)\n"
                    copy_text += f"\nProb: {parlay['combined_prob']*100:.0f}% | Odds: {parlay['american_odds']:+d}"
                    st.code(copy_text, language=None)
    
    st.markdown("---")
    
    # Ultimate parlay
    st.subheader("🏆 Ultimate Parlay (Max Legs)")
    
    ultimate_legs = min(20, len(qualified))
    ultimate = generate_best_parlay(qualified, ultimate_legs, threshold)
    
    if ultimate:
        col1, col2, col3 = st.columns(3)
        col1.metric("Legs", ultimate["num_legs"])
        col2.metric("Prob", f"{ultimate['combined_prob'] * 100:.2f}%")
        col3.metric("Odds", f"{ultimate['american_odds']:+d}" if ultimate['american_odds'] < 100000 else "Long shot!")
        
        ult_data = [{
            "Player": p["player"]["name"],
            "Team": p["player"]["team"],
            "Hit%": f"{p['player'][f'hit_rate_{threshold}plus']:.1f}%"
        } for p in ultimate["legs"]]
        st.dataframe(pd.DataFrame(ult_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # SGPs
    st.subheader("🎮 Single Game Parlays (SGP)")
    st.caption("For profit boosts: 3+ legs, same game, +300 minimum")
    
    sgp_found = False
    for game in games:
        sgp = generate_sgp_for_game(qualified, game["id"], threshold, min_legs=3, min_odds=300)
        
        if sgp:
            sgp_found = True
            st.markdown(f"### {game['matchup']} ({game['time']})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Legs", sgp["num_legs"])
            col2.metric("Prob", f"{sgp['combined_prob'] * 100:.1f}%")
            col3.metric("Odds", f"+{sgp['american_odds']}")
            col4.metric(f"${unit_size:.0f} Wins", f"${sgp['payout_per_100'] * unit_size / 100:.0f}")
            
            sgp_data = [{
                "Player": p["player"]["name"],
                "Team": p["player"]["team"],
                "Hit%": f"{p['player'][f'hit_rate_{threshold}plus']:.1f}%"
            } for p in sgp["legs"]]
            st.dataframe(pd.DataFrame(sgp_data), use_container_width=True, hide_index=True)
            
            sgp_text = f"🎮 SGP - {game['matchup']}:\n"
            for p in sgp["legs"]:
                sgp_text += f"• {p['player']['name']} O{threshold-0.5} SOG\n"
            sgp_text += f"Odds: +{sgp['american_odds']} | Use profit boost!"
            st.code(sgp_text, language=None)
            st.markdown("---")
    
    if not sgp_found:
        st.info("No SGPs with 3+ qualified players at +300 odds.")
        
        # Try all plays
        st.markdown("### SGPs Including Higher Risk Players")
        for game in games:
            sgp = generate_sgp_for_game(plays, game["id"], threshold, min_legs=3, min_odds=300)
            if sgp:
                st.markdown(f"**{game['matchup']}** ⚠️")
                sgp_data = [{
                    "Player": p["player"]["name"],
                    "Hit%": f"{p['player'][f'hit_rate_{threshold}plus']:.1f}%",
                    "Status": "✅" if p["is_qualified"] else "⚠️"
                } for p in sgp["legs"]]
                st.dataframe(pd.DataFrame(sgp_data), use_container_width=True, hide_index=True)

def show_help():
    st.header("❓ Help")
    
    # Full definitions
    show_definitions()
    
    st.markdown("""
    ## Quick Start
    
    1. **Run Analysis** - Fetches all players with 80%+ hit rate
    2. **All Results** - See everyone, sorted by hit rate
    3. **Best Bets** - Qualified (safest) vs Higher Risk
    4. **Parlays** - Best combo for each leg count + SGPs
    
    ## What Should I Bet?
    
    | Risk Level | What to Pick |
    |------------|--------------|
    | **Safest** | 2-leg parlay of top Qualified players |
    | **Moderate** | 3-leg parlay of Qualified players |
    | **Aggressive** | 5+ leg parlay |
    | **Lottery** | Ultimate parlay (all legs) |
    | **Profit Boost** | SGP with +300 odds |
    
    ## Why Hit Rate > Confidence?
    
    **Hit rate is historical fact.** If a player has hit 2+ SOG in 90% of games, that's real data.
    
    **Confidence is a model estimate.** It tries to predict tonight, but it's just math.
    
    For parlays, prioritize high hit rate players because:
    - More consistent = fewer busted legs
    - Historical performance > model predictions
    """)

if __name__ == "__main__":
    main()
