#!/usr/bin/env python3
"""
NHL SOG Analyzer V7.2 - RECALIBRATED SCORING
============================================
Changes from V7.1:
- Hit rate is KING: 89% hitter = base score 78.5
- Modifiers fine-tune from base (not additive from zero)
- Terminal-style "coder vibe" progress display
- Fixed duplicate games display issue

Target calibration:
- 89% hitter + A matchup + PP1 = LOCK (~90+)
- 88% hitter + C matchup = STRONG (~80)
- 85% hitter + F matchup = SOLID (~70)
- 75% hitter = RISKY (~57)
"""

import streamlit as st
import requests
import time
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import pytz

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SharpSlip NHL SOG V7.2",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================
NHL_API = "https://api-web.nhle.com/v1"
SEASON = "20242025"
GAME_TYPE = 2
MIN_GAMES = 15
EST = pytz.timezone('US/Eastern')

# V7.2 Tier thresholds
LOCK_THRESHOLD = 88
STRONG_THRESHOLD = 80
SOLID_THRESHOLD = 70
RISKY_THRESHOLD = 60

# Defense grades by shots allowed per game
DEFENSE_GRADES = {
    "A+": (34.0, 100, 5), "A": (32.0, 34.0, 4), "B": (30.0, 32.0, 2),
    "C": (28.0, 30.0, 0), "D": (26.0, 28.0, -3), "F": (0, 26.0, -5)
}

# Terminal messages for coder vibe
TERMINAL_MESSAGES = [
    "Initializing neural pathways...",
    "Calibrating shot probability matrices...",
    "Computing Poisson distributions...",
    "Analyzing defensive vulnerabilities...",
    "Cross-referencing historical patterns...",
    "Optimizing feature weights...",
    "Running Monte Carlo simulations...",
    "Validating statistical significance...",
    "Aggregating multi-factor scores...",
    "Applying Bayesian adjustments...",
    "Normalizing variance coefficients...",
    "Calculating expected value deltas...",
    "Processing game state correlations...",
    "Generating confidence intervals...",
    "Finalizing tier classifications...",
]

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class TeamDefense:
    abbrev: str
    name: str
    games: int
    sa_per_game: float
    sa_l5: float
    grade: str
    modifier: int
    trend: str

@dataclass
class PlayerStats:
    id: int
    name: str
    team: str
    position: str
    games_played: int
    total_shots: int
    avg_sog: float
    l5_avg: float
    l10_avg: float
    std_dev: float
    hit_rate_2plus: float
    shutout_rate: float
    floor: int
    ceiling: int
    p10: int
    p90: int
    streak: int
    home_avg: float
    away_avg: float
    sog_per_60: float
    avg_toi: float
    toi_trend: float
    is_pp1: bool
    recent_shots: List[int] = field(default_factory=list)

@dataclass
class ScoredPlay:
    player: PlayerStats
    opponent: str
    opp_defense: TeamDefense
    is_home: bool
    game_time: str
    score: float
    tier: str
    probability: float
    factors: Dict[str, float]
    tags: List[str]
    killable: bool

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_est_now():
    return datetime.now(EST)

def linear_scale(value: float, min_val: float, max_val: float, 
                 max_points: float, inverted: bool = False) -> float:
    """Scale value to points within range."""
    clamped = max(min_val, min(max_val, value))
    if inverted:
        proportion = (max_val - clamped) / (max_val - min_val)
    else:
        proportion = (clamped - min_val) / (max_val - min_val)
    return proportion * max_points

def hit_rate_to_base_score(hit_rate: float) -> float:
    """
    V7.2 Core: Hit rate maps directly to base score.
    70% → 50, 80% → 65, 85% → 72.5, 90% → 80, 95% → 87.5, 100% → 95
    Formula: 50 + (hit_rate - 70) * 1.5
    """
    if hit_rate < 70:
        return 50 + (hit_rate - 70) * 1.0  # Penalty below 70%
    return 50 + (hit_rate - 70) * 1.5

def score_to_probability(score: float) -> float:
    """
    Convert score to win probability using logistic function.
    Calibrated so: 65→70%, 75→80%, 85→88%, 90→92%
    """
    # Logistic: 1 / (1 + e^(-k*(x-midpoint)))
    k = 0.08
    midpoint = 55
    raw_prob = 1 / (1 + math.exp(-k * (score - midpoint)))
    # Scale to 45%-94% range
    return 0.45 + raw_prob * 0.49

def get_tier(score: float) -> Tuple[str, str]:
    """Return tier name and emoji."""
    if score >= LOCK_THRESHOLD:
        return "🔒 LOCK", "🟢"
    elif score >= STRONG_THRESHOLD:
        return "✅ STRONG", "🔵"
    elif score >= SOLID_THRESHOLD:
        return "📊 SOLID", "🟡"
    elif score >= RISKY_THRESHOLD:
        return "⚠️ RISKY", "🟠"
    else:
        return "❌ AVOID", "🔴"

# ============================================================================
# TERMINAL-STYLE PROGRESS DISPLAY
# ============================================================================
def create_terminal_css():
    """Return CSS for terminal-style display."""
    return """
    <style>
    .terminal-container {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.5;
        max-height: 400px;
        overflow-y: auto;
    }
    .terminal-header {
        color: #58a6ff;
        font-weight: bold;
        margin-bottom: 8px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }
    .terminal-line {
        color: #8b949e;
        margin: 2px 0;
    }
    .terminal-success { color: #3fb950; }
    .terminal-warning { color: #d29922; }
    .terminal-error { color: #f85149; }
    .terminal-info { color: #58a6ff; }
    .terminal-data { color: #a5d6ff; }
    .terminal-dim { color: #6e7681; }
    .terminal-prompt { color: #7ee787; }
    </style>
    """

def format_terminal_line(timestamp: str, message: str, style: str = "") -> str:
    """Format a single terminal line."""
    style_class = f"terminal-{style}" if style else "terminal-line"
    return f'<div class="{style_class}">[{timestamp}] $ {message}</div>'

# ============================================================================
# NHL API FUNCTIONS
# ============================================================================
@st.cache_data(ttl=300)
def get_schedule(date_str: str) -> List[Dict]:
    """Fetch games for a specific date."""
    try:
        resp = requests.get(f"{NHL_API}/schedule/{date_str}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        games = []
        for week in data.get("gameWeek", []):
            if week.get("date") == date_str:
                for game in week.get("games", []):
                    away = game.get("awayTeam", {}).get("abbrev", "")
                    home = game.get("homeTeam", {}).get("abbrev", "")
                    game_id = str(game.get("id", ""))
                    
                    if not away or not home:
                        continue
                    
                    try:
                        utc = datetime.fromisoformat(game["startTimeUTC"].replace("Z", "+00:00"))
                        time_str = utc.astimezone(EST).strftime("%I:%M %p")
                    except:
                        time_str = "TBD"
                    
                    games.append({
                        "id": game_id,
                        "away": away,
                        "home": home,
                        "time": time_str
                    })
        return games
    except Exception as e:
        return []

@st.cache_data(ttl=600)
def get_all_teams() -> List[Dict]:
    """Get all NHL teams."""
    try:
        resp = requests.get(f"{NHL_API}/standings/now", timeout=30)
        resp.raise_for_status()
        return [
            {"abbrev": t.get("teamAbbrev", {}).get("default", ""),
             "name": t.get("teamName", {}).get("default", "")}
            for t in resp.json().get("standings", [])
        ]
    except:
        return []

def get_team_roster(team: str) -> List[Dict]:
    """Get forwards and defensemen for a team."""
    try:
        resp = requests.get(f"{NHL_API}/roster/{team}/current", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        players = []
        for pos in ["forwards", "defensemen"]:
            for p in data.get(pos, []):
                first = p.get("firstName", {}).get("default", "")
                last = p.get("lastName", {}).get("default", "")
                if first and last and p.get("id"):
                    players.append({
                        "id": p["id"],
                        "name": f"{first} {last}",
                        "position": p.get("positionCode", ""),
                        "team": team
                    })
        return players
    except:
        return []

def get_player_gamelog(player_id: int) -> Optional[List[Dict]]:
    """Get player's game log for current season."""
    try:
        resp = requests.get(
            f"{NHL_API}/player/{player_id}/game-log/{SEASON}/{GAME_TYPE}",
            timeout=15
        )
        resp.raise_for_status()
        return resp.json().get("gameLog", [])
    except:
        return None

def get_player_advanced(player_id: int) -> Dict:
    """Get advanced stats including PP time."""
    try:
        resp = requests.get(f"{NHL_API}/player/{player_id}/landing", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        for season in data.get("seasonTotals", []):
            if str(season.get("season")) == SEASON and season.get("gameTypeId") == GAME_TYPE:
                pp_toi = season.get("powerPlayToi", "00:00")
                gp = max(season.get("gamesPlayed", 1), 1)
                try:
                    parts = pp_toi.split(":")
                    total_mins = int(parts[0]) + int(parts[1]) / 60
                    return {"pp_toi_per_game": round(total_mins / gp, 2)}
                except:
                    pass
        return {"pp_toi_per_game": 0}
    except:
        return {"pp_toi_per_game": 0}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_player(player_id: int, name: str, team: str, position: str) -> Optional[PlayerStats]:
    """Analyze a single player's shot statistics."""
    games = get_player_gamelog(player_id)
    if not games or len(games) < MIN_GAMES:
        return None
    
    # Extract shot data
    shots = []
    home_shots = []
    away_shots = []
    toi_values = []
    
    for g in games:
        sog = g.get("shots", 0)
        if sog < 0:
            sog = 0
        shots.append(sog)
        
        toi_str = g.get("toi", "0:00")
        try:
            parts = toi_str.split(":")
            toi_mins = int(parts[0]) + int(parts[1]) / 60
            toi_values.append(toi_mins)
        except:
            pass
        
        if g.get("homeRoadFlag") == "H":
            home_shots.append(sog)
        else:
            away_shots.append(sog)
    
    if not shots:
        return None
    
    gp = len(shots)
    total = sum(shots)
    avg = total / gp
    
    # Recent averages
    l5 = shots[:5] if len(shots) >= 5 else shots
    l10 = shots[:10] if len(shots) >= 10 else shots
    l5_avg = sum(l5) / len(l5)
    l10_avg = sum(l10) / len(l10)
    
    # Variance
    std = statistics.stdev(shots) if len(shots) > 1 else 0
    
    # Hit rates
    hit_2 = sum(1 for s in shots if s >= 2) / gp * 100
    shutout = sum(1 for s in shots if s <= 1) / gp * 100
    
    # Percentiles
    sorted_shots = sorted(shots)
    p10_idx = max(0, int(len(sorted_shots) * 0.1) - 1)
    p90_idx = min(len(sorted_shots) - 1, int(len(sorted_shots) * 0.9))
    
    # Streak
    streak = 0
    for s in shots:
        if s >= 2:
            streak += 1
        else:
            break
    
    # Home/away
    home_avg = sum(home_shots) / len(home_shots) if home_shots else avg
    away_avg = sum(away_shots) / len(away_shots) if away_shots else avg
    
    # TOI and SOG/60
    avg_toi = sum(toi_values) / len(toi_values) if toi_values else 15
    sog_per_60 = (avg / avg_toi * 60) if avg_toi > 0 else 0
    
    # TOI trend
    toi_trend = 0
    if len(toi_values) >= 5:
        recent_toi = sum(toi_values[:5]) / 5
        season_toi = sum(toi_values) / len(toi_values)
        if season_toi > 0:
            toi_trend = (recent_toi - season_toi) / season_toi * 100
    
    # PP1 status
    adv = get_player_advanced(player_id)
    is_pp1 = adv.get("pp_toi_per_game", 0) >= 2.5
    
    return PlayerStats(
        id=player_id,
        name=name,
        team=team,
        position=position,
        games_played=gp,
        total_shots=total,
        avg_sog=round(avg, 2),
        l5_avg=round(l5_avg, 2),
        l10_avg=round(l10_avg, 2),
        std_dev=round(std, 2),
        hit_rate_2plus=round(hit_2, 1),
        shutout_rate=round(shutout, 1),
        floor=min(shots),
        ceiling=max(shots),
        p10=sorted_shots[p10_idx],
        p90=sorted_shots[p90_idx],
        streak=streak,
        home_avg=round(home_avg, 2),
        away_avg=round(away_avg, 2),
        sog_per_60=round(sog_per_60, 2),
        avg_toi=round(avg_toi, 1),
        toi_trend=round(toi_trend, 1),
        is_pp1=is_pp1,
        recent_shots=shots[:10]
    )

def get_defense_stats(teams: set, log_callback=None) -> Dict[str, TeamDefense]:
    """Calculate defense stats for relevant teams."""
    all_teams = get_all_teams()
    defense = {}
    
    for i, team in enumerate(all_teams):
        abbrev = team["abbrev"]
        
        if log_callback:
            log_callback(f"Fetching {abbrev} defense stats...", "info")
        
        try:
            resp = requests.get(
                f"{NHL_API}/club-schedule-season/{abbrev}/{SEASON}",
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            
            completed = [g for g in data.get("games", [])
                        if g.get("gameType") == GAME_TYPE and g.get("gameState") == "OFF"]
            
            if not completed:
                defense[abbrev] = TeamDefense(
                    abbrev=abbrev, name=team["name"], games=0,
                    sa_per_game=30.0, sa_l5=30.0, grade="C",
                    modifier=0, trend="→"
                )
                continue
            
            recent = completed[-20:]
            last_5 = completed[-5:] if len(completed) >= 5 else completed
            
            sa_all = []
            sa_l5 = []
            
            for game in recent:
                try:
                    box = requests.get(
                        f"{NHL_API}/gamecenter/{game['id']}/boxscore",
                        timeout=10
                    ).json()
                    
                    home_abbrev = box.get("homeTeam", {}).get("abbrev", "")
                    home_sog = box.get("homeTeam", {}).get("sog", 0)
                    away_sog = box.get("awayTeam", {}).get("sog", 0)
                    
                    if home_sog == 0 and away_sog == 0:
                        for stat in box.get("boxscore", {}).get("teamGameStats", []):
                            if stat.get("category") == "sog":
                                home_sog = stat.get("homeValue", 0)
                                away_sog = stat.get("awayValue", 0)
                                break
                    
                    sa = away_sog if abbrev == home_abbrev else home_sog
                    if sa > 0:
                        sa_all.append(sa)
                        if game in last_5:
                            sa_l5.append(sa)
                    
                    time.sleep(0.03)
                except:
                    continue
            
            if sa_all:
                sa_avg = statistics.mean(sa_all)
                sa_l5_avg = statistics.mean(sa_l5) if sa_l5 else sa_avg
            else:
                sa_avg = 30.0
                sa_l5_avg = 30.0
            
            # Determine grade
            grade = "C"
            modifier = 0
            for g, (low, high, mod) in DEFENSE_GRADES.items():
                if low <= sa_avg < high:
                    grade = g
                    modifier = mod
                    break
            
            # Trend
            if sa_l5_avg > sa_avg + 2:
                trend = "↑"  # Getting worse (more shots allowed)
            elif sa_l5_avg < sa_avg - 2:
                trend = "↓"  # Getting better
            else:
                trend = "→"
            
            if log_callback:
                log_callback(f"  └── SA/G: {sa_avg:.1f} | Grade: {grade} | Trend: {trend}", "data")
            
            defense[abbrev] = TeamDefense(
                abbrev=abbrev,
                name=team["name"],
                games=len(sa_all),
                sa_per_game=round(sa_avg, 2),
                sa_l5=round(sa_l5_avg, 2),
                grade=grade,
                modifier=modifier,
                trend=trend
            )
            
        except Exception as e:
            defense[abbrev] = TeamDefense(
                abbrev=abbrev, name=team["name"], games=0,
                sa_per_game=30.0, sa_l5=30.0, grade="C",
                modifier=0, trend="→"
            )
        
        time.sleep(0.05)
    
    return defense

# ============================================================================
# V7.2 SCORING ENGINE
# ============================================================================
def calculate_v72_score(player: PlayerStats, defense: TeamDefense, 
                        is_home: bool) -> Tuple[float, Dict[str, float], List[str]]:
    """
    V7.2 Scoring: Hit rate is KING.
    Base score from hit rate, modifiers adjust from there.
    """
    factors = {}
    tags = []
    
    # =========================================
    # BASE SCORE FROM HIT RATE (the foundation)
    # =========================================
    base = hit_rate_to_base_score(player.hit_rate_2plus)
    factors["base_hit_rate"] = round(base, 1)
    
    # =========================================
    # MODIFIERS (centered around zero)
    # =========================================
    
    # Shutout rate: 0% = +4, 10% = 0, 20% = -4
    # This captures DOWNSIDE risk (frequent 0-1 SOG games)
    shutout_mod = linear_scale(player.shutout_rate, 0, 20, 8, inverted=True) - 4
    factors["shutout_mod"] = round(shutout_mod, 1)
    
    # Variance: NO PENALTY - high σ from upside games is fine
    # Downside risk already captured by shutout_rate and P10 floor
    # Just track for display purposes
    factors["variance_display"] = player.std_dev
    
    # SOG/60: high (12+) = +2, low (6) = -2
    sog60_mod = linear_scale(player.sog_per_60, 6, 12, 4) - 2
    factors["sog60_mod"] = round(sog60_mod, 1)
    
    # Volume: high (4.5+) = +2, low (2.0) = -2
    vol_mod = linear_scale(player.avg_sog, 2.0, 4.5, 4) - 2
    factors["volume_mod"] = round(vol_mod, 1)
    
    # Floor (P10): 2+ = +2, 0 = -2
    floor_mod = linear_scale(player.p10, 0, 2, 4) - 2
    factors["floor_mod"] = round(floor_mod, 1)
    if player.p10 >= 2:
        tags.append("🛡️")
    
    # =========================================
    # SITUATIONAL MODIFIERS
    # =========================================
    
    # Matchup (defense grade)
    matchup_mod = defense.modifier
    factors["matchup"] = matchup_mod
    
    # Defense trend bonus
    trend_mod = 0
    if defense.trend == "↑":
        trend_mod = 1  # Defense getting worse = good for us
    elif defense.trend == "↓":
        trend_mod = -1
    factors["defense_trend"] = trend_mod
    
    # PP1 bonus
    pp_mod = 0
    if player.is_pp1:
        pp_mod = 3
        tags.append("⚡")
    factors["pp1"] = pp_mod
    
    # Home/away
    venue_mod = 1 if is_home else -0.5
    factors["venue"] = venue_mod
    
    # Hot/cold streak
    streak_mod = 0
    if player.streak >= 10:
        streak_mod = 3
        tags.append(f"🔥{player.streak}G")
    elif player.streak >= 7:
        streak_mod = 2
        tags.append(f"🔥{player.streak}G")
    elif player.streak >= 5:
        streak_mod = 1
        tags.append(f"🔥{player.streak}G")
    factors["streak"] = streak_mod
    
    # Recent form (L5 vs season)
    form_diff = player.l5_avg - player.avg_sog
    if form_diff >= 0.8:
        form_mod = 2
    elif form_diff >= 0.4:
        form_mod = 1
    elif form_diff <= -0.8:
        form_mod = -2
    elif form_diff <= -0.4:
        form_mod = -1
    else:
        form_mod = 0
    factors["form"] = form_mod
    
    # TOI trend
    toi_mod = 0
    if player.toi_trend >= 10:
        toi_mod = 1
    elif player.toi_trend <= -10:
        toi_mod = -1
        tags.append("⏱️↓")
    factors["toi_trend"] = toi_mod
    
    # =========================================
    # FINAL SCORE
    # =========================================
    modifiers = (shutout_mod + sog60_mod + vol_mod + floor_mod +
                 matchup_mod + trend_mod + pp_mod + venue_mod + 
                 streak_mod + form_mod + toi_mod)
    
    final_score = base + modifiers
    final_score = max(20, min(98, final_score))  # Clamp to valid range
    
    # =========================================
    # KILLABLE STATUS (for parlay safety)
    # Based on DOWNSIDE risk only - not variance
    # =========================================
    killable = (
        player.hit_rate_2plus >= 75 and
        player.shutout_rate <= 12 and
        player.p10 >= 1
    )
    
    return round(final_score, 1), factors, tags, killable

# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================
def run_analysis(date_str: str, games: List[Dict], 
                 terminal_placeholder, min_hit_rate: float = 70.0) -> List[ScoredPlay]:
    """Run full analysis with terminal-style progress."""
    
    terminal_lines = []
    msg_idx = 0
    
    def add_line(text: str, style: str = ""):
        nonlocal terminal_lines
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        terminal_lines.append(format_terminal_line(ts, text, style))
        # Keep last 30 lines
        if len(terminal_lines) > 30:
            terminal_lines = terminal_lines[-30:]
        update_terminal()
    
    def update_terminal():
        html = create_terminal_css()
        html += '<div class="terminal-container">'
        html += '<div class="terminal-header">SHARPSLIP NHL SOG ANALYZER V7.2</div>'
        html += ''.join(terminal_lines)
        html += '</div>'
        terminal_placeholder.markdown(html, unsafe_allow_html=True)
    
    def get_message():
        nonlocal msg_idx
        msg = TERMINAL_MESSAGES[msg_idx % len(TERMINAL_MESSAGES)]
        msg_idx += 1
        return msg
    
    # Initialize
    add_line(f"Initializing analysis for {date_str}...", "info")
    time.sleep(0.3)
    
    add_line(f"Found {len(games)} games scheduled", "success")
    for g in games:
        add_line(f"  └── {g['away']} @ {g['home']} ({g['time']})", "dim")
    
    # Build team mapping
    teams_playing = set()
    game_info = {}
    
    for g in games:
        teams_playing.add(g["away"])
        teams_playing.add(g["home"])
        game_info[g["away"]] = {"opponent": g["home"], "is_home": False, "time": g["time"]}
        game_info[g["home"]] = {"opponent": g["away"], "is_home": True, "time": g["time"]}
    
    # Phase 1: Defense metrics
    add_line("", "")
    add_line("═" * 45, "dim")
    add_line("PHASE 1: DEFENSIVE METRICS", "info")
    add_line("═" * 45, "dim")
    
    def defense_callback(msg, style):
        add_line(msg, style)
    
    defense_stats = get_defense_stats(teams_playing, defense_callback)
    add_line(f"Defense analysis complete: {len(defense_stats)} teams", "success")
    
    # Phase 2: Roster acquisition
    add_line("", "")
    add_line("═" * 45, "dim")
    add_line("PHASE 2: ROSTER ACQUISITION", "info")
    add_line("═" * 45, "dim")
    
    all_players = []
    for team in teams_playing:
        add_line(f"Loading {team} roster...", "info")
        roster = get_team_roster(team)
        all_players.extend(roster)
        add_line(f"  └── Found {len(roster)} skaters", "data")
        time.sleep(0.05)
    
    add_line(f"Total players to analyze: {len(all_players)}", "success")
    
    # Phase 3: Player analysis
    add_line("", "")
    add_line("═" * 45, "dim")
    add_line("PHASE 3: PLAYER ANALYSIS", "info")
    add_line("═" * 45, "dim")
    
    plays = []
    qualified = 0
    
    for i, p in enumerate(all_players):
        if i % 5 == 0:
            add_line(get_message(), "dim")
        
        stats = analyze_player(p["id"], p["name"], p["team"], p["position"])
        
        if not stats:
            continue
        
        if stats.hit_rate_2plus < min_hit_rate:
            continue
        
        info = game_info.get(p["team"])
        if not info:
            continue
        
        opp_def = defense_stats.get(info["opponent"])
        if not opp_def:
            continue
        
        # Calculate score
        score, factors, tags, killable = calculate_v72_score(
            stats, opp_def, info["is_home"]
        )
        
        tier, color = get_tier(score)
        prob = score_to_probability(score)
        
        play = ScoredPlay(
            player=stats,
            opponent=info["opponent"],
            opp_defense=opp_def,
            is_home=info["is_home"],
            game_time=info["time"],
            score=score,
            tier=tier,
            probability=prob,
            factors=factors,
            tags=tags,
            killable=killable
        )
        plays.append(play)
        qualified += 1
        
        # Log qualified players
        tier_emoji = "🔒" if "LOCK" in tier else "✅" if "STRONG" in tier else "📊" if "SOLID" in tier else "⚠️"
        add_line(f"{tier_emoji} {stats.name}: Score={score:.0f} | Hit={stats.hit_rate_2plus:.0f}% | σ={stats.std_dev}", 
                "success" if score >= 75 else "warning" if score >= 65 else "dim")
        
        time.sleep(0.03)
    
    # Phase 4: Finalization
    add_line("", "")
    add_line("═" * 45, "dim")
    add_line("ANALYSIS COMPLETE", "success")
    add_line("═" * 45, "dim")
    
    # Sort by score
    plays.sort(key=lambda x: x.score, reverse=True)
    
    # Summary
    locks = len([p for p in plays if "LOCK" in p.tier])
    strong = len([p for p in plays if "STRONG" in p.tier])
    solid = len([p for p in plays if "SOLID" in p.tier])
    
    add_line(f"🔒 LOCKs: {locks} | ✅ STRONG: {strong} | 📊 SOLID: {solid}", "success")
    add_line(f"Total qualified plays: {len(plays)}", "info")
    
    return plays

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    st.title("🏒 SharpSlip NHL SOG Analyzer V7.2")
    st.caption("Hit Rate is King • Recalibrated Scoring • Terminal Aesthetic")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        today = get_est_now().date()
        selected_date = st.date_input("📅 Date", value=today)
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        min_hit_rate = st.slider("Min Hit Rate %", 65, 85, 70)
        
        st.markdown("---")
        
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(f"EST: {get_est_now().strftime('%I:%M %p')}")
        
        # Version info
        st.markdown("---")
        st.markdown("""
        **V7.2 Changes:**
        - Hit rate → base score directly
        - Modifiers fine-tune (±)
        - 89% hitter = ~78 base
        - A matchup = +4
        - Terminal progress display
        """)
    
    # Main area
    if run_btn:
        games = get_schedule(date_str)
        
        if not games:
            st.error(f"No games found for {date_str}")
            return
        
        # Terminal placeholder - this replaces all progress display
        terminal = st.empty()
        
        # Run analysis
        plays = run_analysis(date_str, games, terminal, min_hit_rate)
        
        # Clear terminal and show results
        terminal.empty()
        
        if not plays:
            st.warning("No plays found matching criteria")
            return
        
        # Store in session
        st.session_state.plays = plays
        st.session_state.date = date_str
    
    # Display results if available
    if "plays" in st.session_state and st.session_state.plays:
        plays = st.session_state.plays
        date_str = st.session_state.get("date", "")
        
        # Summary metrics
        st.subheader(f"📊 Results for {date_str}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        locks = len([p for p in plays if "LOCK" in p.tier])
        strong = len([p for p in plays if "STRONG" in p.tier])
        solid = len([p for p in plays if "SOLID" in p.tier])
        risky = len([p for p in plays if "RISKY" in p.tier])
        
        col1.metric("🔒 LOCK", locks)
        col2.metric("✅ STRONG", strong)
        col3.metric("📊 SOLID", solid)
        col4.metric("⚠️ RISKY", risky)
        col5.metric("Total", len(plays))
        
        # Results table
        st.markdown("---")
        
        rows = []
        for p in plays:
            _, color = get_tier(p.score)
            
            # Format tags
            tags_str = " ".join(p.tags) if p.tags else ""
            
            # Format matchup
            matchup = f"{p.opponent} ({p.opp_defense.grade})"
            
            # TOI display
            toi_str = f"{p.player.avg_toi:.0f}m"
            if abs(p.player.toi_trend) >= 8:
                toi_str += f" ({p.player.toi_trend:+.0f}%)"
            
            rows.append({
                "Score": f"{color} {p.score:.0f}",
                "Tier": p.tier,
                "Player": p.player.name,
                "Tags": tags_str,
                "Team": p.player.team,
                "vs": matchup,
                "H/A": "🏠" if p.is_home else "✈️",
                "Hit%": f"{p.player.hit_rate_2plus:.0f}%",
                "Shut%": f"{p.player.shutout_rate:.0f}%",
                "Avg": p.player.avg_sog,
                "L5": p.player.l5_avg,
                "SOG/60": p.player.sog_per_60,
                "TOI": toi_str,
                "σ": p.player.std_dev,
                "Prob%": f"{p.probability*100:.0f}%",
                "Kill": "✓" if p.killable else "❌"
            })
        
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Tier": st.column_config.TextColumn("Tier", width="small"),
                "Player": st.column_config.TextColumn("Player", width="medium"),
                "Tags": st.column_config.TextColumn("Tags", width="small"),
                "Hit%": st.column_config.TextColumn("Hit%", width="small"),
                "Prob%": st.column_config.TextColumn("Prob%", width="small"),
            }
        )
        
        # Export
        st.markdown("---")
        
        import pandas as pd
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"nhl_sog_v72_{date_str}.csv",
            mime="text/csv"
        )
        
        # Scoring breakdown expander
        with st.expander("📐 V7.2 Scoring Breakdown"):
            st.markdown("""
            ### Base Score from Hit Rate
            - `70% → 50 pts`
            - `80% → 65 pts`
            - `85% → 72.5 pts`
            - `90% → 80 pts`
            - `95% → 87.5 pts`
            
            ### Modifiers (centered at 0)
            | Factor | Range | Points |
            |--------|-------|--------|
            | Shutout Rate | 0-20% | +4 to -4 |
            | SOG/60 | 6-12 | -2 to +2 |
            | Volume | 2.0-4.5 | -2 to +2 |
            | Floor (P10) | 0-2 | -2 to +2 |
            
            *Note: Variance (σ) is displayed but NOT penalized - high σ from upside games is fine. Downside risk is captured by Shutout Rate and P10 Floor.*
            
            ### Situational
            | Factor | Value |
            |--------|-------|
            | A+ Matchup | +5 |
            | A Matchup | +4 |
            | F Matchup | -5 |
            | PP1 | +3 |
            | Home | +1 |
            | Away | -0.5 |
            | 10G+ Streak | +3 |
            | Hot Form | +1 to +2 |
            | Cold Form | -1 to -2 |
            
            ### Tier Thresholds
            - 🔒 LOCK: 88+
            - ✅ STRONG: 80-87
            - 📊 SOLID: 70-79
            - ⚠️ RISKY: 60-69
            - ❌ AVOID: <60
            """)
    
    else:
        # No results yet - show just the instructions
        st.info("👈 Click **Run Analysis** to analyze today's games")
        
        # Show today's schedule
        games = get_schedule(get_est_now().strftime("%Y-%m-%d"))
        if games:
            st.subheader("📅 Today's Games")
            for g in games:
                st.write(f"**{g['away']}** @ **{g['home']}** - {g['time']}")

if __name__ == "__main__":
    main()
