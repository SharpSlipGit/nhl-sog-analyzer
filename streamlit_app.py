#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer v3.3
===============================
Changes from v3.2:
- Player name is first column
- Status column is just icons (✅/⚠️)
- Parlay table is clean, with expandable copy sections
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
REQUEST_DELAY = 0.05
EST = pytz.timezone('US/Eastern')

MATCHUP_GRADES = {
    "A+": 33.0, "A": 32.0, "B+": 31.0, "B": 30.0,
    "C+": 29.0, "C": 28.0, "D": 27.0, "F": 0.0,
}

MODEL_WEIGHTS = {
    "l5_weight": 0.40,
    "l10_weight": 0.30,
    "season_weight": 0.30,
    "hit_rate_90_boost": 1.12,
    "hit_rate_85_boost": 1.08,
    "hit_rate_80_boost": 1.04,
    "pp1_boost": 1.18,
    "pp2_boost": 1.08,
    "home_boost": 1.03,
    "away_penalty": 0.97,
    "hot_streak_boost": 1.05,
    "cold_streak_penalty": 0.92,
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
    elif player["pp_toi"] > 0.5:
        tags.append("PP")
    if player["current_streak"] >= 5:
        tags.append(f"🔥{player['current_streak']}G")
    return " ".join(tags)

def get_status_icon(hit_rate: float, is_cold: bool) -> Tuple[bool, str]:
    """Returns (is_qualified, icon)"""
    if hit_rate >= 85 and not is_cold:
        return True, "✅"
    else:
        return False, "⚠️"

def format_parlay_text(legs: List[Dict], threshold: int, parlay_name: str, prob: float, odds: int) -> str:
    """Format parlay for copy-paste."""
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
# PROBABILITY MODEL
# ============================================================================
def calculate_model_probability(player: Dict, opp_def: Dict, is_home: bool, threshold: int) -> float:
    base_lambda = (
        player["last_5_avg"] * MODEL_WEIGHTS["l5_weight"] +
        player["last_10_avg"] * MODEL_WEIGHTS["l10_weight"] +
        player["avg_sog"] * MODEL_WEIGHTS["season_weight"]
    )
    
    hit_rate = player[f"hit_rate_{threshold}plus"]
    if hit_rate >= 90:
        hr_factor = MODEL_WEIGHTS["hit_rate_90_boost"]
    elif hit_rate >= 85:
        hr_factor = MODEL_WEIGHTS["hit_rate_85_boost"]
    elif hit_rate >= 80:
        hr_factor = MODEL_WEIGHTS["hit_rate_80_boost"]
    else:
        hr_factor = 1.0
    
    if player["is_pp1"]:
        pp_factor = MODEL_WEIGHTS["pp1_boost"]
    elif player["pp_toi"] > 1.0:
        pp_factor = MODEL_WEIGHTS["pp2_boost"]
    else:
        pp_factor = 1.0
    
    ha_factor = MODEL_WEIGHTS["home_boost"] if is_home else MODEL_WEIGHTS["away_penalty"]
    
    opp_sa = opp_def.get("shots_allowed_per_game", 30.0)
    opp_factor = opp_sa / 30.0
    
    trend, is_hot, is_cold = get_trend(player["last_5_avg"], player["avg_sog"])
    if is_hot:
        streak_factor = MODEL_WEIGHTS["hot_streak_boost"]
    elif is_cold:
        streak_factor = MODEL_WEIGHTS["cold_streak_penalty"]
    else:
        streak_factor = 1.0
    
    adj_lambda = base_lambda * hr_factor * pp_factor * ha_factor * opp_factor * streak_factor
    prob = poisson_prob_at_least(adj_lambda, threshold)
    
    return prob

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
                "legs": legs,
                "num_legs": num_legs,
                "combined_prob": combined_prob,
                "american_odds": american_odds,
                "payout_per_100": calculate_parlay_payout(american_odds, 100),
                "game_id": game_id,
                "qualified_count": qualified_count,
                "risky_count": risky_count,
                "risk_level": "🟢" if risky_count == 0 else ("🟡" if risky_count <= 1 else "🔴")
            }
    
    if len(sorted_plays) >= min_legs:
        legs = sorted_plays[:min_legs]
        probs = [p[prob_key] / 100 for p in legs]
        combined_prob, american_odds = calculate_parlay_odds(probs)
        qualified_count = sum(1 for p in legs if p["is_qualified"])
        risky_count = min_legs - qualified_count
        
        return {
            "legs": legs,
            "num_legs": min_legs,
            "combined_prob": combined_prob,
            "american_odds": american_odds,
            "payout_per_100": calculate_parlay_payout(american_odds, 100),
            "game_id": game_id,
            "qualified_count": qualified_count,
            "risky_count": risky_count,
            "risk_level": "⚪" if american_odds < min_odds else ("🟢" if risky_count == 0 else "🔴")
        }
    
    return None

# ============================================================================
# UI COMPONENTS
# ============================================================================
def show_model_explanation():
    with st.expander("📖 How It Works", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Model Weights
            
            **Base** = L5 (40%) + L10 (30%) + Avg (30%)
            
            | Factor | Multiplier |
            |--------|------------|
            | Hit Rate 90%+ | ×1.12 |
            | Hit Rate 85%+ | ×1.08 |
            | Hit Rate 80%+ | ×1.04 |
            | PP1 | ×1.18 |
            | PP2 | ×1.08 |
            | Home | ×1.03 |
            | Away | ×0.97 |
            | 🔥 Hot | ×1.05 |
            | ❄️ Cold | ×0.92 |
            """)
        
        with col2:
            st.markdown("""
            ### Status Icons
            
            | Icon | Meaning |
            |------|---------|
            | ✅ | Qualified (85%+, not cold) |
            | ⚠️ | Higher risk |
            | 🛡️ | High floor (never 0 SOG) |
            | ⚡ | PP1 player |
            | 🔥 | Hot streak |
            | ❄️ | Cold streak |
            
            ### Matchup Grades
            
            **A+/A** = Soft defense (good)  
            **D/F** = Tough defense (bad)
            """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer")
    st.caption("v3.3 | Sorted by Model Probability")
    
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
            plays, games = run_full_analysis(date_str, threshold)
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

def run_full_analysis(date_str: str, threshold: int) -> Tuple[List[Dict], List[Dict]]:
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
    
    team_defense = get_team_defense_stats(teams_playing, progress_bar, status_text)
    
    progress_bar.progress(25)
    status_text.text("📋 Fetching rosters...")
    
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
        
        hit_rate = stats["hit_rate_2plus"] if threshold == 2 else stats["hit_rate_3plus"] if threshold == 3 else stats["hit_rate_4plus"]
        
        if hit_rate < MIN_HIT_RATE:
            time.sleep(REQUEST_DELAY)
            continue
        
        info = game_info.get(player["team"])
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
        
        players_found += 1
        
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
        time.sleep(REQUEST_DELAY)
    
    progress_bar.progress(100)
    status_text.text("✅ Complete!")
    stats_display.text(f"Total: {players_checked} checked, {players_found} with {MIN_HIT_RATE}%+ hit rate")
    time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    stats_display.empty()
    
    prob_key = f"prob_{threshold}plus"
    all_plays.sort(key=lambda x: x[prob_key], reverse=True)
    
    st.success(f"Found **{len(all_plays)}** players with {MIN_HIT_RATE}%+ hit rate!")
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
    
    # Player name FIRST, status as icon only
    results_data = []
    for play in plays:
        p = play["player"]
        
        row = {
            "": play["status_icon"],  # Icon column (no header)
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
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Prob%": st.column_config.ProgressColumn("Prob%", min_value=0, max_value=100, format="%.1f%%"),
            "Hit%": st.column_config.NumberColumn("Hit%", format="%.1f%%"),
        }
    )
    
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name=f"nhl_sog_{date_str}.csv", mime="text/csv")

def show_best_bets(plays: List[Dict], threshold: int):
    st.header("🎯 Best Bets")
    show_model_explanation()
    
    qualified = [p for p in plays if p["is_qualified"]]
    risky = [p for p in plays if not p["is_qualified"]]
    
    hit_key = f"hit_rate_{threshold}plus"
    prob_key = f"prob_{threshold}plus"
    
    st.subheader(f"✅ Qualified ({len(qualified)})")
    st.caption("85%+ hit rate, not cold")
    
    if qualified:
        qual_data = []
        for play in qualified:
            p = play["player"]
            qual_data.append({
                "Player": p["name"],
                "Tags": play["tags"],
                "Team": p["team"],
                "vs": play["opponent"],
                "Prob%": play[prob_key],
                "Hit%": p[hit_key],
                "L5": p["last_5_avg"],
                "Trend": play["trend"],
            })
        st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No qualified plays")
    
    st.markdown("---")
    
    st.subheader(f"⚠️ Higher Risk ({len(risky)})")
    st.caption("75-84% hit rate or cold streak")
    
    if risky:
        risk_data = []
        for play in risky:
            p = play["player"]
            risk_data.append({
                "Player": p["name"],
                "Tags": play["tags"],
                "Team": p["team"],
                "vs": play["opponent"],
                "Prob%": play[prob_key],
                "Hit%": p[hit_key],
                "L5": p["last_5_avg"],
                "Trend": play["trend"],
            })
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

def show_parlays(plays: List[Dict], games: List[Dict], threshold: int, unit_size: float):
    st.header("🎰 Parlays")
    show_model_explanation()
    
    prob_key = f"prob_{threshold}plus"
    hit_key = f"hit_rate_{threshold}plus"
    
    sorted_plays = sorted(plays, key=lambda x: x[prob_key], reverse=True)
    
    st.success(f"Building from **{len(sorted_plays)}** players")
    
    # CLEAN PARLAY TABLE
    st.subheader("📊 Best Parlay by Legs")
    
    max_legs = min(12, len(sorted_plays))
    
    # Summary table
    parlay_table = []
    parlays_dict = {}  # Store for expanders
    
    for num_legs in range(1, max_legs + 1):
        parlay = generate_best_parlay(sorted_plays, num_legs, threshold)
        if parlay:
            prob_pct = parlay['combined_prob'] * 100
            odds = parlay['american_odds']
            payout = parlay['payout_per_100'] * unit_size / 100
            players = ", ".join([p["player"]["name"] for p in parlay["legs"]])
            
            parlay_table.append({
                "Legs": num_legs,
                "Prob%": f"{prob_pct:.1f}%",
                "Odds": f"{odds:+d}" if odds < 10000 else "—",
                f"${unit_size:.0f} Wins": f"${payout:.0f}",
                "Players": players[:60] + "..." if len(players) > 60 else players
            })
            parlays_dict[num_legs] = parlay
    
    if parlay_table:
        st.dataframe(pd.DataFrame(parlay_table), use_container_width=True, hide_index=True)
    
    # EXPANDABLE COPY SECTIONS
    st.markdown("### 📋 Click to Copy")
    
    cols = st.columns(4)
    for i, num_legs in enumerate([2, 3, 5, 10]):
        if num_legs in parlays_dict:
            parlay = parlays_dict[num_legs]
            with cols[i % 4]:
                with st.expander(f"**{num_legs}-Leg** ({parlay['combined_prob']*100:.0f}%)"):
                    copy_text = format_parlay_text(
                        parlay["legs"], 
                        threshold, 
                        f"{num_legs}-Leg Parlay",
                        parlay['combined_prob'],
                        parlay['american_odds']
                    )
                    st.code(copy_text, language=None)
    
    st.markdown("---")
    
    # ULTIMATE PARLAY
    st.subheader("🏆 Ultimate Parlay")
    
    ultimate_legs = min(20, len(sorted_plays))
    ultimate = generate_best_parlay(sorted_plays, ultimate_legs, threshold)
    
    if ultimate:
        col1, col2, col3 = st.columns(3)
        col1.metric("Legs", ultimate["num_legs"])
        col2.metric("Prob", f"{ultimate['combined_prob'] * 100:.2f}%")
        col3.metric("Odds", f"{ultimate['american_odds']:+d}" if ultimate['american_odds'] < 100000 else "Long shot")
        
        with st.expander("📋 View Players & Copy"):
            ult_data = [{"Player": p["player"]["name"], "Team": p["player"]["team"], "Prob%": f"{p[prob_key]:.0f}%"} for p in ultimate["legs"]]
            st.dataframe(pd.DataFrame(ult_data), use_container_width=True, hide_index=True)
            
            copy_text = format_parlay_text(ultimate["legs"], threshold, f"Ultimate {ultimate['num_legs']}-Leg", ultimate['combined_prob'], ultimate['american_odds'])
            st.code(copy_text, language=None)
    
    st.markdown("---")
    
    # SGPs
    st.subheader("🎮 Single Game Parlays")
    st.caption("For profit boosts | 3+ legs | +300 target")
    
    for game in games:
        sgp = generate_sgp_for_game(sorted_plays, game["id"], threshold, min_legs=3, min_odds=300)
        
        if sgp:
            risk_text = f"{sgp['risk_level']} {sgp['qualified_count']}✅ {sgp['risky_count']}⚠️"
            odds_text = f"{sgp['american_odds']:+d}"
            
            with st.expander(f"**{game['matchup']}** | {odds_text} | {risk_text}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Legs", sgp["num_legs"])
                col2.metric("Prob", f"{sgp['combined_prob'] * 100:.1f}%")
                col3.metric("Odds", odds_text)
                col4.metric(f"${unit_size:.0f} Wins", f"${sgp['payout_per_100'] * unit_size / 100:.0f}")
                
                sgp_data = [{
                    "": "✅" if p["is_qualified"] else "⚠️",
                    "Player": p["player"]["name"],
                    "Prob%": f"{p[prob_key]:.0f}%",
                    "Hit%": f"{p['player'][hit_key]:.0f}%",
                } for p in sgp["legs"]]
                st.dataframe(pd.DataFrame(sgp_data), use_container_width=True, hide_index=True)
                
                copy_text = f"🎮 SGP - {game['matchup']}\n"
                copy_text += "─" * 30 + "\n"
                for p in sgp["legs"]:
                    status = "✅" if p["is_qualified"] else "⚠️"
                    copy_text += f"{status} {p['player']['name']} O{threshold-0.5} SOG\n"
                copy_text += "─" * 30 + "\n"
                copy_text += f"Odds: {sgp['american_odds']:+d} | Risk: {sgp['risk_level']}\n"
                st.code(copy_text, language=None)

def show_help():
    st.header("❓ Help")
    show_model_explanation()
    
    st.markdown("""
    ## Quick Start
    
    1. **Run Analysis** → Fetches players with 75%+ hit rate
    2. **All Results** → Everyone sorted by model probability  
    3. **Best Bets** → ✅ Qualified vs ⚠️ Higher Risk
    4. **Parlays** → Best combos + SGPs for profit boosts
    
    ## Status Icons
    
    | Icon | Meaning |
    |------|---------|
    | ✅ | Qualified (85%+, not cold) |
    | ⚠️ | Higher risk (75-84% or cold) |
    
    ## Risk Levels (SGPs)
    
    | Level | Meaning |
    |-------|---------|
    | 🟢 | All qualified players |
    | 🟡 | 1 risky player |
    | 🔴 | 2+ risky players |
    | ⚪ | Below +300 odds |
    
    ## Copying Parlays
    
    Click the expander for any parlay to see the formatted text.
    Select all and copy to share!
    """)

if __name__ == "__main__":
    main()
