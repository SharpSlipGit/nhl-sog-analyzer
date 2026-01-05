#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer - Cloud Version v2
==============================================
Fixed: Caching issues, added detailed progress tracking
"""

import streamlit as st
import requests
import time
import math
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
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
MIN_GAMES = 10
REQUEST_DELAY = 0.05
EST = pytz.timezone('US/Eastern')

MATCHUP_GRADES = {
    "A+": 33.0, "A": 32.0, "B+": 31.0, "B": 30.0,
    "C+": 29.0, "C": 28.0, "D": 27.0, "F": 0.0,
}

PARLAY_CONFIG = {
    2: {"name": "Power Pair", "unit_size": 1.0, "target_odds": "+100 to +150", "expected_hit": 70},
    3: {"name": "Triple Threat", "unit_size": 1.0, "target_odds": "+180 to +280", "expected_hit": 55},
    4: {"name": "Four Banger", "unit_size": 0.75, "target_odds": "+300 to +450", "expected_hit": 45},
    5: {"name": "High Five", "unit_size": 0.5, "target_odds": "+450 to +700", "expected_hit": 35},
    10: {"name": "Lottery Ticket", "unit_size": 0.1, "target_odds": "+2000 to +5000", "expected_hit": 12},
}

# ============================================================================
# SIMPLE DATA STORAGE (no dataclasses for caching compatibility)
# ============================================================================
def make_team_defense(abbrev, name, gp, sa_pg, sa_l5, pk_sa, rank=0, grade="", trend=""):
    return {
        "team_abbrev": abbrev,
        "team_name": name,
        "games_played": gp,
        "shots_allowed_per_game": sa_pg,
        "shots_allowed_L5": sa_l5,
        "pk_shots_allowed": pk_sa,
        "rank": rank,
        "grade": grade,
        "recent_trend": trend
    }

def make_player_stats(player_id, name, team, position, gp, hit2, hit3, hit4, avg, l5, l10, 
                      std, floor, ceiling, streak, home, away, pp_toi, is_pp1, shots, opps):
    return {
        "player_id": player_id,
        "name": name,
        "team": team,
        "position": position,
        "games_played": gp,
        "hit_rate_2plus": hit2,
        "hit_rate_3plus": hit3,
        "hit_rate_4plus": hit4,
        "avg_sog": avg,
        "last_5_avg": l5,
        "last_10_avg": l10,
        "std_dev": std,
        "floor": floor,
        "ceiling": ceiling,
        "current_streak": streak,
        "home_avg": home,
        "away_avg": away,
        "pp_toi": pp_toi,
        "is_pp1": is_pp1,
        "all_shots": shots,
        "opponents": opps
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_date():
    return datetime.now(EST).strftime("%Y-%m-%d")

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

def get_trend_emoji(l5: float, season: float) -> str:
    diff = l5 - season
    if diff >= 0.5:
        return "🔥 HOT"
    elif diff <= -0.5:
        return "❄️ COLD"
    return "➡️ STEADY"

def get_tier(confidence: float, hit_rate: float) -> str:
    if confidence >= 75 and hit_rate >= 85:
        return "🔒 LOCK"
    elif confidence >= 65 and hit_rate >= 80:
        return "✅ STRONG"
    elif confidence >= 55 and hit_rate >= 75:
        return "📊 SOLID"
    elif confidence >= 45:
        return "⚠️ RISKY"
    return "❌ AVOID"

def detect_correlation(leg1: Dict, leg2: Dict) -> Tuple[float, str]:
    if leg1["team"] == leg2["team"]:
        if leg1["is_pp1"] and leg2["is_pp1"]:
            return (0.3, f"Same team PP1 ({leg1['team']})")
        return (0.15, f"Teammates ({leg1['team']})")
    if leg1["opponent"] == leg2["team"] or leg2["opponent"] == leg1["team"]:
        return (-0.1, f"Opponents")
    if leg1["game_id"] == leg2["game_id"]:
        return (0.05, f"Same game")
    return (0.0, "Independent")

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
    """Fetch defense stats for teams playing today only (faster)."""
    team_defense = {}
    all_teams = get_all_teams()
    
    # Filter to only teams we need
    teams_to_fetch = [t for t in all_teams if t["abbrev"] in teams_playing]
    total_teams = len(teams_to_fetch)
    
    for i, team in enumerate(teams_to_fetch):
        abbrev = team["abbrev"]
        status_text.text(f"📊 Fetching defense stats: {abbrev} ({i+1}/{total_teams})")
        progress_bar.progress(int((i / total_teams) * 30))
        
        try:
            url = f"{NHL_WEB_API}/club-schedule-season/{abbrev}/{SEASON}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            completed = [g for g in data.get("games", []) 
                        if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
            
            if not completed:
                team_defense[abbrev] = make_team_defense(
                    abbrev, team["name"], 0, 30.0, 30.0, 8.0, 0, "C", "➡️ STABLE"
                )
                continue
            
            # Only fetch last 10 games for speed
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
            
            team_defense[abbrev] = make_team_defense(
                abbrev, team["name"], len(sa_all),
                round(sa_pg, 2), round(sa_L5_avg, 2), round(sa_pg * 0.25, 2),
                0, get_grade(sa_pg), trend
            )
            
        except Exception as e:
            team_defense[abbrev] = make_team_defense(
                abbrev, team["name"], 0, 30.0, 30.0, 8.0, 0, "C", "➡️ STABLE"
            )
        
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
        
        return make_player_stats(
            player_id, name, team, position, gp,
            round(hit_2, 1), round(hit_3, 1), round(hit_4, 1),
            round(avg, 2), round(l5_avg, 2), round(l10_avg, 2),
            round(std, 2), min(all_shots), max(all_shots), streak,
            round(home_avg, 2), round(away_avg, 2),
            pp_toi, is_pp1, all_shots, opponents
        )
    except:
        return None

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def calculate_adjusted_lambda(player: Dict, opp_def: Dict, is_home: bool) -> float:
    base = (player["last_5_avg"] * 0.4) + (player["last_10_avg"] * 0.3) + (player["avg_sog"] * 0.3)
    
    if is_home and player["home_avg"] > 0:
        home_factor = player["home_avg"] / player["avg_sog"] if player["avg_sog"] > 0 else 1.0
    elif not is_home and player["away_avg"] > 0:
        home_factor = player["away_avg"] / player["avg_sog"] if player["avg_sog"] > 0 else 1.0
    else:
        home_factor = 1.02 if is_home else 0.98
    
    opp_factor = opp_def["shots_allowed_per_game"] / 30.0 if opp_def["shots_allowed_per_game"] > 0 else 1.0
    
    if opp_def["shots_allowed_L5"] > 0:
        recent_factor = opp_def["shots_allowed_L5"] / 30.0
        opp_factor = (opp_factor * 0.6) + (recent_factor * 0.4)
    
    pp_factor = 1.0
    if player["is_pp1"]:
        pp_factor = 1.15
    elif player["pp_toi"] > 1.0:
        pp_factor = 1.08
    
    return base * home_factor * opp_factor * pp_factor

def calculate_confidence(player: Dict, opp: Dict, is_home: bool, threshold: int = 2) -> Tuple[float, List[str], List[str]]:
    edges = []
    risks = []
    score = 50
    
    hit_rate = player["hit_rate_2plus"] if threshold == 2 else player["hit_rate_3plus"] if threshold == 3 else player["hit_rate_4plus"]
    
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
    
    form_diff = player["last_5_avg"] - player["avg_sog"]
    if form_diff >= 1.0:
        score += 15
        edges.append(f"🔥 Hot streak (L5: {player['last_5_avg']})")
    elif form_diff >= 0.5:
        score += 10
    elif form_diff <= -1.0:
        score -= 15
        risks.append(f"❄️ Cold streak (L5: {player['last_5_avg']})")
    elif form_diff <= -0.5:
        score -= 10
    
    if opp["grade"] in ["A+", "A"]:
        score += 15
        edges.append(f"Soft matchup ({opp['team_abbrev']}: {opp['shots_allowed_per_game']} SA/G)")
    elif opp["grade"] == "B+":
        score += 8
    elif opp["grade"] in ["D", "F"]:
        score -= 15
        risks.append(f"Tough matchup ({opp['team_abbrev']}: {opp['shots_allowed_per_game']} SA/G)")
    
    if "LOOSENING" in opp["recent_trend"]:
        score += 8
        edges.append(f"Defense loosening (L5: {opp['shots_allowed_L5']})")
    elif "TIGHTENING" in opp["recent_trend"]:
        score -= 8
        risks.append("Defense tightening")
    
    if player["is_pp1"]:
        score += 10
        edges.append(f"⚡ PP1 ({player['pp_toi']:.1f} min/g)")
    elif player["pp_toi"] > 1.0:
        score += 5
    
    if player["std_dev"] < 1.0:
        score += 8
        edges.append(f"Very consistent (σ={player['std_dev']})")
    elif player["std_dev"] > 2.0:
        score -= 8
        risks.append(f"High variance (σ={player['std_dev']})")
    
    if player["floor"] >= threshold:
        score += 8
        edges.append(f"Floor of {player['floor']} SOG")
    elif player["floor"] == 0:
        score -= 5
        risks.append("Has 0 SOG games")
    
    if is_home and player["home_avg"] > player["away_avg"] + 0.3:
        score += 5
        edges.append(f"Home boost ({player['home_avg']} vs {player['away_avg']})")
    elif not is_home and player["away_avg"] < player["home_avg"] - 0.3:
        score -= 5
        risks.append("Road penalty")
    
    return max(0, min(100, score)), edges, risks

# ============================================================================
# PARLAY BUILDER
# ============================================================================
def build_parlay_legs(plays: List[Dict], threshold: int) -> List[Dict]:
    legs = []
    for play in plays:
        prob = play["prob_2plus"] / 100 if threshold == 2 else play["prob_3plus"] / 100 if threshold == 3 else play["prob_4plus"] / 100
        
        legs.append({
            "player_name": play["player"]["name"],
            "team": play["player"]["team"],
            "opponent": play["opponent"],
            "threshold": threshold - 0.5,
            "our_prob": prob,
            "confidence": int(play["confidence"]),
            "tier": play["tier"],
            "game_id": play["game_id"],
            "is_pp1": play["player"]["is_pp1"],
            "trend": play["trend"]
        })
    
    return legs

def generate_parlays(legs: List[Dict], num_legs: int, max_parlays: int = 10) -> List[Dict]:
    if len(legs) < num_legs:
        return []
    
    sorted_legs = sorted(legs, key=lambda x: (x["confidence"], x["our_prob"]), reverse=True)
    candidates = sorted_legs[:min(len(sorted_legs), 15)]
    all_combos = list(combinations(candidates, num_legs))
    
    parlays = []
    for combo in all_combos[:200]:
        probs = [leg["our_prob"] for leg in combo]
        combined_prob, combined_odds = calculate_parlay_odds(probs)
        
        total_corr = 0
        corr_notes = []
        pairs = list(combinations(combo, 2))
        for leg1, leg2 in pairs:
            corr, note = detect_correlation(leg1, leg2)
            total_corr += corr
            if abs(corr) > 0.05:
                corr_notes.append(note)
        
        avg_corr = total_corr / len(pairs) if pairs else 0
        adjusted_prob = combined_prob * (1 + avg_corr * 0.1)
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        tier_scores = {"🔒 LOCK": 4, "✅ STRONG": 3, "📊 SOLID": 2, "⚠️ RISKY": 1, "❌ AVOID": 0}
        avg_tier = sum(tier_scores.get(leg["tier"], 0) for leg in combo) / num_legs
        
        if avg_tier >= 3.5:
            risk_level = "LOW"
        elif avg_tier >= 2.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        config = PARLAY_CONFIG.get(num_legs, PARLAY_CONFIG[5])
        
        parlay = {
            "legs": list(combo),
            "combined_prob": adjusted_prob,
            "combined_odds": combined_odds,
            "expected_payout": calculate_parlay_payout(combined_odds),
            "unit_size": config["unit_size"],
            "category": config["name"],
            "correlation_score": avg_corr,
            "correlation_notes": corr_notes[:3],
            "risk_level": risk_level
        }
        parlays.append(parlay)
    
    parlays.sort(key=lambda x: x["combined_prob"], reverse=True)
    return parlays[:max_parlays]

def get_best_parlays_by_category(legs: List[Dict]) -> Dict[str, List[Dict]]:
    return {
        "Power Pairs (2-leg)": generate_parlays(legs, 2, 5),
        "Triple Threats (3-leg)": generate_parlays(legs, 3, 5),
        "High Fives (5-leg)": generate_parlays(legs, 5, 3),
    }

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer")
    st.caption("Shots on Goal Betting Analysis | SharpSlip")
    
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
        
        st.subheader("🔍 Filters")
        min_hit_rate = st.slider("Min Hit Rate %", 50, 95, 75)
        min_confidence = st.slider("Min Confidence", 0, 100, 40)
        pp_only = st.checkbox("PP Players Only", value=False)
        
        st.markdown("---")
        
        st.subheader("💰 Bankroll")
        unit_size = st.number_input("Unit Size ($)", min_value=1, max_value=1000, value=25)
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"Current: {get_est_datetime().strftime('%I:%M %p EST')}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 All Results", 
        "🎯 Best Bets & Parlays", 
        "❓ Help"
    ])
    
    # Session state
    if 'betting_plays' not in st.session_state:
        st.session_state.betting_plays = []
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 2
    
    with tab1:
        if run_analysis:
            plays = run_full_analysis(date_str, threshold, min_hit_rate, min_confidence, pp_only)
            st.session_state.betting_plays = plays
            st.session_state.threshold = threshold
        elif st.session_state.betting_plays:
            display_all_results(st.session_state.betting_plays, st.session_state.threshold, date_str)
        else:
            st.info("👈 Click **Run Analysis** to fetch today's plays")
            games = get_todays_schedule(date_str)
            if games:
                st.subheader(f"📅 Games on {date_str}")
                for game in games:
                    st.write(f"**{game['away_team']}** @ **{game['home_team']}** - {game['time']}")
            else:
                st.warning("No games scheduled for this date.")
    
    with tab2:
        if st.session_state.betting_plays:
            show_betting_strategy(st.session_state.betting_plays, st.session_state.threshold, unit_size, date_str)
        else:
            st.info("Run analysis first to see betting strategy")
    
    with tab3:
        show_help()

def run_full_analysis(date_str: str, threshold: int, min_hit_rate: float, 
                     min_confidence: float, pp_only: bool) -> List[Dict]:
    
    st.subheader(f"📅 Games on {date_str}")
    games = get_todays_schedule(date_str)
    
    if not games:
        st.error("No games found for this date!")
        return []
    
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
            "game_id": game["id"]
        }
        game_info[game["home_team"]] = {
            "opponent": game["away_team"], 
            "home_away": "HOME", 
            "time": game["time"],
            "game_id": game["id"]
        }
    
    st.markdown("---")
    st.subheader("📊 Analysis Progress")
    
    # Progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.empty()
    
    # Counters for live stats
    players_checked = 0
    players_qualified = 0
    
    # Fetch defense stats
    status_text.text(f"🛡️ Fetching defense stats for {len(teams_playing)} teams...")
    team_defense = get_team_defense_stats(teams_playing, progress_bar, status_text)
    
    # Fetch rosters
    progress_bar.progress(30)
    status_text.text(f"📋 Fetching rosters for {len(teams_playing)} teams...")
    
    all_players = []
    for team in teams_playing:
        roster = get_team_roster(team)
        all_players.extend(roster)
        time.sleep(REQUEST_DELAY)
    
    progress_bar.progress(35)
    status_text.text(f"👥 Found {len(all_players)} players to analyze")
    time.sleep(0.5)
    
    # Analyze players
    betting_plays = []
    total = len(all_players)
    
    for i, player in enumerate(all_players):
        players_checked += 1
        
        # Update progress
        pct = 35 + int((i / total) * 65)
        progress_bar.progress(pct)
        status_text.text(f"🔍 Analyzing: {player['name']} ({player['team']})")
        stats_container.text(f"✅ Checked: {players_checked}/{total} | 🎯 Qualified: {players_qualified}")
        
        stats = get_player_stats(player["id"], player["name"], player["team"], player["position"])
        
        if not stats:
            time.sleep(REQUEST_DELAY)
            continue
        
        if pp_only and not stats["is_pp1"] and stats["pp_toi"] < 1.0:
            continue
        
        hit_rate = stats["hit_rate_2plus"] if threshold == 2 else stats["hit_rate_3plus"] if threshold == 3 else stats["hit_rate_4plus"]
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
        h2h_shots = [stats["all_shots"][j] for j, o in enumerate(stats["opponents"]) if o == opp]
        h2h_avg = sum(h2h_shots) / len(h2h_shots) if h2h_shots else 0
        h2h_games = len(h2h_shots)
        
        # Probabilities
        adj_lambda = calculate_adjusted_lambda(stats, opp_def, is_home)
        prob_2 = poisson_prob_at_least(adj_lambda, 2)
        prob_3 = poisson_prob_at_least(adj_lambda, 3)
        prob_4 = poisson_prob_at_least(adj_lambda, 4)
        
        # Confidence
        conf, edges, risks = calculate_confidence(stats, opp_def, is_home, threshold)
        
        if conf < min_confidence:
            time.sleep(REQUEST_DELAY)
            continue
        
        main_prob = prob_2 if threshold == 2 else prob_3 if threshold == 3 else prob_4
        implied_odds = implied_prob_to_american(main_prob)
        
        players_qualified += 1
        
        play = {
            "player": stats,
            "opponent": opp,
            "opponent_defense": opp_def,
            "home_away": info["home_away"],
            "game_time": info["time"],
            "game_id": info["game_id"],
            "h2h_avg": round(h2h_avg, 2),
            "h2h_games": h2h_games,
            "prob_2plus": round(prob_2 * 100, 1),
            "prob_3plus": round(prob_3 * 100, 1),
            "prob_4plus": round(prob_4 * 100, 1),
            "confidence": conf,
            "tier": get_tier(conf, hit_rate),
            "trend": get_trend_emoji(stats["last_5_avg"], stats["avg_sog"]),
            "implied_odds": implied_odds,
            "edge_factors": edges,
            "risk_factors": risks
        }
        betting_plays.append(play)
        time.sleep(REQUEST_DELAY)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    stats_container.text(f"📊 Final: {players_checked} players checked, {players_qualified} qualified")
    time.sleep(1)
    
    # Clear progress elements
    progress_bar.empty()
    status_text.empty()
    stats_container.empty()
    
    if not betting_plays:
        st.warning("No qualifying plays found with current filters! Try lowering the minimum hit rate or confidence.")
        return []
    
    betting_plays.sort(key=lambda x: x["confidence"], reverse=True)
    
    st.success(f"🎯 Found {len(betting_plays)} qualifying plays!")
    display_all_results(betting_plays, threshold, date_str)
    
    return betting_plays

def display_all_results(plays: List[Dict], threshold: int, date_str: str):
    st.subheader(f"🎯 All Plays - Over {threshold - 0.5} SOG")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    locks = len([p for p in plays if "LOCK" in p["tier"]])
    strong = len([p for p in plays if "STRONG" in p["tier"]])
    solid = len([p for p in plays if "SOLID" in p["tier"]])
    pp_players = len([p for p in plays if p["player"]["is_pp1"]])
    
    col1.metric("🔒 Locks", locks)
    col2.metric("✅ Strong", strong)
    col3.metric("📊 Solid", solid)
    col4.metric("⚡ PP1 Players", pp_players)
    col5.metric("Total", len(plays))
    
    results_data = []
    for play in plays:
        p = play["player"]
        hit_rate = p["hit_rate_2plus"] if threshold == 2 else p["hit_rate_3plus"] if threshold == 3 else p["hit_rate_4plus"]
        prob = play["prob_2plus"] if threshold == 2 else play["prob_3plus"] if threshold == 3 else play["prob_4plus"]
        
        row = {
            "Tier": play["tier"],
            "Player": p["name"],
            "Team": p["team"],
            "Pos": p["position"],
            "vs": play["opponent"],
            "H/A": play["home_away"],
            "Hit%": f"{hit_rate:.1f}%",
            "Avg": p["avg_sog"],
            "L5": p["last_5_avg"],
            "Floor": p["floor"],
            "PP": "⚡" if p["is_pp1"] else ("✓" if p["pp_toi"] > 1 else ""),
            "Prob%": f"{prob:.1f}%",
            "Conf": int(play["confidence"]),
            "Trend": play["trend"],
            "Matchup": play["opponent_defense"]["grade"],
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=100, format="%d")
        }
    )
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"nhl_sog_{threshold-0.5}_{date_str}.csv",
        mime="text/csv"
    )

def show_betting_strategy(plays: List[Dict], threshold: int, unit_size: float, date_str: str):
    st.header("🎯 Betting Strategy")
    
    legs = build_parlay_legs(plays, threshold)
    quality_legs = [l for l in legs if l["confidence"] >= 60 and l["our_prob"] >= 0.75]
    
    if not quality_legs:
        st.warning("Not enough high-quality plays. Try lowering filters.")
        return
    
    st.success(f"Found **{len(quality_legs)}** quality plays (60+ confidence, 75%+ probability)")
    
    # Unit sizing guide
    with st.expander("💰 Unit Sizing Guide", expanded=False):
        st.markdown(f"""
        **Your Unit: ${unit_size}**
        
        | Type | Legs | Bet | Target Odds | Hit Rate |
        |------|------|-----|-------------|----------|
        | Power Pair | 2 | ${unit_size:.0f} | +100 to +150 | ~70% |
        | Triple Threat | 3 | ${unit_size:.0f} | +180 to +280 | ~55% |
        | High Five | 5 | ${unit_size * 0.5:.0f} | +450 to +700 | ~35% |
        """)
    
    parlays_by_category = get_best_parlays_by_category(quality_legs)
    
    # Best parlays
    st.subheader("🔥 Recommended Parlays")
    
    for category, parlays in parlays_by_category.items():
        if not parlays:
            continue
        
        best = parlays[0]
        
        st.markdown(f"### {category}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Probability", f"{best['combined_prob'] * 100:.1f}%")
        col2.metric("Odds", f"{best['combined_odds']:+d}")
        col3.metric("Bet", f"${unit_size * best['unit_size']:.0f}")
        col4.metric("To Win", f"${best['expected_payout'] * unit_size * best['unit_size'] / 100:.0f}")
        
        leg_data = [{
            "Player": leg["player_name"],
            "Team": leg["team"],
            "Line": f"O{leg['threshold']} SOG",
            "Prob": f"{leg['our_prob'] * 100:.0f}%",
            "Conf": leg["confidence"],
            "PP1": "⚡" if leg["is_pp1"] else ""
        } for leg in best["legs"]]
        
        st.dataframe(pd.DataFrame(leg_data), hide_index=True, use_container_width=True)
        
        # Copy-paste format
        copy_text = f"NHL SOG Parlay ({category}):\n"
        for leg in best["legs"]:
            copy_text += f"• {leg['player_name']} O{leg['threshold']} SOG\n"
        copy_text += f"Combined: {best['combined_prob']*100:.0f}% | Odds: {best['combined_odds']:+d}"
        
        st.code(copy_text, language=None)
        st.markdown("---")

def show_help():
    st.header("❓ Help")
    
    st.markdown("""
    ### Quick Start
    1. Select date and threshold (Over 1.5 is most common)
    2. Click **Run Analysis** (takes 2-4 minutes)
    3. Check **All Results** for full data
    4. Go to **Best Bets & Parlays** for recommended plays
    
    ### Tiers Explained
    
    | Tier | Meaning | Action |
    |------|---------|--------|
    | 🔒 LOCK | 75+ confidence, 85%+ hit rate | Best plays |
    | ✅ STRONG | 65+ confidence, 80%+ hit rate | Good plays |
    | 📊 SOLID | 55+ confidence, 75%+ hit rate | Decent plays |
    | ⚠️ RISKY | Lower confidence | Avoid |
    
    ### Key Metrics
    
    - **Hit%**: How often player hits this threshold historically
    - **Prob%**: Model-calculated probability for tonight
    - **Conf**: Overall confidence score (0-100)
    - **PP1**: ⚡ = Power Play Unit 1 (more ice time)
    - **Matchup**: Opponent defense grade (A+ = allows most shots)
    
    ### What to Bet
    
    | Priority | Parlay | Unit Size |
    |----------|--------|-----------|
    | 1st | Power Pair (2-leg) | 1.0x your unit |
    | 2nd | Triple Threat (3-leg) | 1.0x your unit |
    | 3rd | High Five (5-leg) | 0.5x your unit |
    
    ### Parlay Strategy
    
    Heavy favorites (-400 to -600) are bad as singles but great in 2-3 leg parlays 
    where combined probability stays above 50% while getting plus-money odds.
    """)

if __name__ == "__main__":
    main()
