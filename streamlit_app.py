#!/usr/bin/env python3
"""
NHL Shots on Goal Analyzer V8.3
===============================
CALIBRATION UPDATE - Based on 24 parlays, 58 legs analysis

KEY CHANGES (V8.3):
- GAME DIVERSIFICATION: Max 1 player per game prevents correlated failures
- PROBLEM PLAYER EXCLUSION: Dorofeyev, Bennett excluded (shutout risk)
- FLOOR PROTECTION: P10=0 with HR<90% auto-killed
- RAISED THRESHOLDS: LOCK=90, min parlay score=75
- LOGISTIC CALIBRATION: A=0.10, B=6.0, cap=92%

CALIBRATION DATA (Dec 9 - Jan 6):
- Parlay Record: 15-9 (62.5%) 
- Leg Hit Rate: 50/58 (86.2%) - excellent!
- Focus: Improve leg selection quality to boost parlay win rate

PARLAY STRATEGY:
- Keep 3-4 leg parlays for ~-150 odds (better value than 2-leg at -600)
- Game diversification prevents correlated failures
- Problem player exclusion removes known shutout risks

v8.3 - January 2026 - Calibration update (diversification, exclusions, floor protection)
v8.2 - January 2026 - Fixed prune function (save_history typo)
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

# JSONBin.io for free cloud storage (no credit card required)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NHL SOG Analyzer V8.2",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="auto"  # Auto-collapses on mobile, expands on desktop
)

# ============================================================================
# MOBILE-RESPONSIVE CSS (does not affect desktop)
# ============================================================================
st.markdown("""
<style>
/* Mobile optimizations - only applies to screens < 768px */
@media (max-width: 768px) {
    /* Smaller fonts throughout */
    .stDataFrame { font-size: 11px !important; }
    .stDataFrame td, .stDataFrame th { 
        padding: 4px 6px !important; 
        font-size: 11px !important;
    }
    
    /* Compact headers */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    
    /* Tighter metrics */
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
    
    /* Reduce padding */
    .block-container { padding: 1rem 0.5rem !important; }
    
    /* Make tabs smaller */
    .stTabs [data-baseweb="tab"] { 
        font-size: 0.8rem !important; 
        padding: 8px 12px !important;
    }
    
    /* Sidebar auto-collapse hint */
    section[data-testid="stSidebar"] { min-width: 0px !important; }
}

/* Ensure horizontal scroll on tables for all screen sizes */
.stDataFrame > div { overflow-x: auto !important; }
</style>
""", unsafe_allow_html=True)

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
PARLAY_HISTORY_FILE = f"{DATA_DIR}/parlay_history.json"

# League averages for baseline
LEAGUE_AVG_SAG = 30.0  # Shots allowed per game
LEAGUE_AVG_SOG = 2.8   # Shots on goal per player

# Shot suppression/friendly teams (research-backed)
# Elite suppressors: Limit shots via possession/structure (reduce projections)
ELITE_SUPPRESSORS = {"CAR", "FLA", "LAK", "VGK", "COL"}
# Shot-friendly: Allow high volume (boost projections)  
SHOT_FRIENDLY = {"SJS", "ANA", "CHI", "CBJ", "PIT"}

# ==============================================================================
# V8.3 CALIBRATION CONSTANTS (Based on Dec 9 - Jan 6 analysis: 24 parlays, 58 legs)
# ==============================================================================

# Logistic mapping coefficients - CALIBRATED
# Old: A=0.09, B=4.8 | New: A=0.10, B=6.0 (reduces overconfidence at top)
LOGISTIC_A = 0.10      # Steepness (was 0.09)
LOGISTIC_B = 6.0       # Midpoint shift (was 4.8)
PROB_CAP = 0.92        # Max probability cap (was 0.94, reduced for realism)

# Tier thresholds - RAISED based on calibration
TIERS = {
    "🔒 LOCK": 90,      # Raised from 88 (user specified 90)
    "✅ STRONG": 80,    # Keep at 80
    "📊 SOLID": 70,     # Keep at 70
    "⚠️ RISKY": 60,     # Keep at 60
    "❌ AVOID": 0
}

# Kill switch thresholds - UPDATED
KILL_SWITCHES = {
    "min_score": 75,           # Raised from 60 - be more selective
    "max_l5_shutouts": 1,      # Keep strict
    "max_variance": 3.0,       # Relaxed from 2.8 (backtest showed variance doesn't hurt!)
    "max_toi_drop_pct": 20,    # Tightened from 25
    "min_l5_hit_rate": 60,     # Keep at 60%
}

# Problem players - BACKTEST IDENTIFIED (hit rate < 70%)
# NOTE: Only Dorofeyev and Bennett excluded per user request
# Celebrini and Kyle Connor remain eligible
PROBLEM_PLAYERS = {
    8481604: {"name": "Pavel Dorofeyev", "rate": "50%", "issue": "shutout risk"},
    8477935: {"name": "Sam Bennett", "rate": "67%", "issue": "shutout risk"},
}

# Parlay configuration - Keep 3+ legs for decent odds (~-150)
# User prefers 3-4 leg parlays for better odds, not 2-leg (-600)
PARLAY_CONFIG = {
    "default_legs": 3,              # Keep at 3 for better odds
    "max_legs": 5,                  # Allow up to 5 legs
    "min_score_for_parlay": 75,     # Raised from 60 - be more selective
    "max_players_same_game": 1,     # NEW: Game diversification prevents correlated failures
    "exclude_problem_players": True, # Use exclusion list
}

# Correlation penalties (unchanged - working well)
CORRELATION_PENALTIES = {
    "same_team": 0.94,
    "same_line": 0.90,
    "same_pp_unit": 0.92,
    "same_game": 0.98,
}

# ============================================================================
# DATA PERSISTENCE (JSONBin.io + Local JSON Fallback)
# ============================================================================
# JSONBin.io is 100% FREE - no credit card required!
# Setup: 1) Create account at jsonbin.io  2) Get API key  3) Add to Streamlit secrets

JSONBIN_API_URL = "https://api.jsonbin.io/v3"

def get_jsonbin_headers():
    """Get JSONBin API headers from Streamlit secrets."""
    try:
        if "jsonbin" in st.secrets:
            return {
                "X-Master-Key": st.secrets["jsonbin"]["api_key"],
                "Content-Type": "application/json"
            }
    except:
        pass
    return None

def jsonbin_load_data(bin_key: str) -> Dict:
    """Load data from JSONBin.io."""
    headers = get_jsonbin_headers()
    if not headers:
        return None
    
    try:
        bin_id = st.secrets["jsonbin"].get(bin_key)
        if not bin_id:
            return None
        
        response = requests.get(
            f"{JSONBIN_API_URL}/b/{bin_id}/latest",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("record", {})
        
    except Exception as e:
        pass
    
    return None

def jsonbin_save_data(bin_key: str, data: Dict) -> bool:
    """Save data to JSONBin.io."""
    # Initialize debug log if needed
    if "save_debug" not in st.session_state:
        st.session_state.save_debug = []
    
    headers = get_jsonbin_headers()
    if not headers:
        st.session_state.save_debug.append(f"❌ JSONBin headers not configured")
        return False
    
    try:
        bin_id = st.secrets["jsonbin"].get(bin_key)
        if not bin_id:
            st.session_state.save_debug.append(f"❌ JSONBin bin_id not found for {bin_key}")
            return False
        
        response = requests.put(
            f"{JSONBIN_API_URL}/b/{bin_id}",
            headers=headers,
            json=data,
            timeout=15
        )
        
        if response.status_code == 200:
            st.session_state.save_debug.append(f"✅ JSONBin API returned 200 OK for {bin_key}")
            return True
        else:
            st.session_state.save_debug.append(f"❌ JSONBin save failed: HTTP {response.status_code}")
            try:
                st.session_state.save_debug.append(f"   Response: {response.text[:200]}")
            except:
                pass
            return False
        
    except Exception as e:
        st.session_state.save_debug.append(f"❌ JSONBin save exception: {str(e)}")
        return False

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_history() -> Dict:
    """Load results history from JSONBin (or local JSON fallback)."""
    # Try JSONBin first
    cloud_data = jsonbin_load_data("results_bin_id")
    if cloud_data is not None:
        return cloud_data
    
    # Fall back to local JSON
    ensure_data_dir()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history: Dict):
    """Save results history to JSONBin (and local JSON as backup)."""
    # Always save to local JSON as backup
    ensure_data_dir()
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Also save to JSONBin if available
    jsonbin_save_data("results_bin_id", history)

def load_parlay_history() -> Dict:
    """Load parlay history from JSONBin (or local JSON fallback)."""
    # Try JSONBin first
    cloud_data = jsonbin_load_data("parlay_bin_id")
    if cloud_data is not None:
        return cloud_data
    
    # Fall back to local JSON
    ensure_data_dir()
    if os.path.exists(PARLAY_HISTORY_FILE):
        try:
            with open(PARLAY_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_parlay_history(history: Dict, verify_date: str = None):
    """Save parlay history to JSONBin (and local JSON as backup).
    
    Args:
        history: Full parlay history dict
        verify_date: Specific date to verify after save (optional but recommended)
    """
    # Initialize debug log if needed
    if "save_debug" not in st.session_state:
        st.session_state.save_debug = []
    
    # SAFEGUARD: Never overwrite a resolved parlay with an unresolved one
    # This prevents the bug where auto_save_parlay() overwrites fetch results
    if verify_date and verify_date in history:
        new_parlay = history[verify_date]
        new_result = new_parlay.get("result") if isinstance(new_parlay, dict) else None
        
        # If we're trying to save result=None, check if cloud already has a result
        if new_result is None:
            cloud_data = jsonbin_load_data("parlay_bin_id")
            if cloud_data and verify_date in cloud_data:
                existing = cloud_data[verify_date]
                existing_result = existing.get("result") if isinstance(existing, dict) else None
                if existing_result in ["WIN", "LOSS"]:
                    # DON'T overwrite! Preserve the existing resolved parlay
                    st.session_state.save_debug.append(f"🛡️ BLOCKED: Won't overwrite {verify_date} (existing result={existing_result}) with result=None")
                    # Update our local history with the cloud data instead
                    history[verify_date] = existing
                    st.session_state.parlay_history[verify_date] = existing
                    return  # Exit without saving
    
    # Log what we're saving for the specific date (if provided)
    if verify_date and verify_date in history:
        parlay = history[verify_date]
        if isinstance(parlay, dict):
            result = parlay.get("result")
            legs_hit = parlay.get("legs_hit")
            st.session_state.save_debug.append(f"📤 Saving {verify_date}: result={result}, legs_hit={legs_hit}")
    
    # Always save to local JSON as backup
    ensure_data_dir()
    with open(PARLAY_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Also save to JSONBin if available
    success = jsonbin_save_data("parlay_bin_id", history)
    if success:
        # VERIFY by reading back immediately - check the SPECIFIC date we just saved
        time.sleep(0.5)  # Small delay to ensure write completes
        verify_data = jsonbin_load_data("parlay_bin_id")
        
        # Use provided verify_date, or fall back to most recent
        check_date = verify_date
        if not check_date:
            recent_dates = sorted(history.keys(), reverse=True)[:1]
            check_date = recent_dates[0] if recent_dates else None
        
        if verify_data and check_date:
            saved_result = history.get(check_date, {}).get("result") if isinstance(history.get(check_date), dict) else None
            loaded_result = verify_data.get(check_date, {}).get("result") if isinstance(verify_data.get(check_date), dict) else None
            saved_legs = history.get(check_date, {}).get("legs_hit") if isinstance(history.get(check_date), dict) else None
            loaded_legs = verify_data.get(check_date, {}).get("legs_hit") if isinstance(verify_data.get(check_date), dict) else None
            
            if saved_result == loaded_result and saved_legs == loaded_legs:
                st.session_state.save_debug.append(f"✅ Verified {check_date}: result={loaded_result}, legs_hit={loaded_legs}")
            else:
                st.session_state.save_debug.append(f"⚠️ MISMATCH {check_date}: sent result={saved_result}/legs={saved_legs}, got result={loaded_result}/legs={loaded_legs}")
                # RETRY: Try saving again
                st.session_state.save_debug.append(f"🔄 Retrying save for {check_date}...")
                time.sleep(0.3)
                retry_success = jsonbin_save_data("parlay_bin_id", history)
                if retry_success:
                    st.session_state.save_debug.append(f"🔄 Retry complete")
                else:
                    st.session_state.save_debug.append(f"❌ Retry failed")
        else:
            st.session_state.save_debug.append("⚠️ Could not verify save (no data)")
    else:
        st.session_state.save_debug.append("❌ JSONBin save FAILED")

def is_cloud_connected() -> bool:
    """Check if JSONBin is properly configured."""
    try:
        return "jsonbin" in st.secrets and "api_key" in st.secrets["jsonbin"]
    except:
        return False

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
if 'parlay_history' not in st.session_state:
    st.session_state.parlay_history = load_parlay_history()
if 'analysis_date' not in st.session_state:
    st.session_state.analysis_date = datetime.now(EST).strftime("%Y-%m-%d")

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

def negative_binomial_prob_at_least(mu: float, variance: float, k: int) -> float:
    """
    Calculate P(X >= k) using Negative Binomial distribution.
    
    Used when variance > mean (overdispersion), common in SOG data.
    Falls back to Poisson if variance <= mean.
    
    Parameters:
        mu: Expected value (projection)
        variance: Variance of player's SOG (std_dev²)
        k: Threshold (e.g., 2 for O1.5)
    """
    if mu <= 0:
        return 0.0
    
    # If not overdispersed, use Poisson
    if variance <= mu:
        return poisson_prob_at_least(mu, k)
    
    # Negative Binomial parameterization
    # r = μ² / (σ² - μ)  [shape parameter]
    # p = μ / σ²         [success probability]
    r = (mu * mu) / (variance - mu)
    p = mu / variance
    
    # P(X >= k) = 1 - P(X < k) = 1 - CDF(k-1)
    # Using scipy-style calculation without scipy
    prob_less_than_k = 0.0
    for i in range(k):
        # PMF of negative binomial: C(i+r-1, i) * p^r * (1-p)^i
        # Using log-space to avoid overflow
        log_coef = math.lgamma(i + r) - math.lgamma(i + 1) - math.lgamma(r)
        log_pmf = log_coef + r * math.log(p) + i * math.log(1 - p)
        prob_less_than_k += math.exp(log_pmf)
    
    return max(0.0, min(1.0, 1.0 - prob_less_than_k))

def calculate_statistical_probability(projection: float, std_dev: float, threshold: int) -> float:
    """
    Calculate TRUE probability of hitting SOG threshold using statistical model.
    
    Uses Negative Binomial if data is overdispersed (common for SOG),
    otherwise falls back to Poisson.
    
    This is a REAL probability, not a heuristic score.
    """
    if projection <= 0:
        return 0.0
    
    variance = std_dev * std_dev
    
    # Add small buffer to variance to account for game-to-game uncertainty
    # This prevents overconfidence for "consistent" players
    variance = max(variance, projection * 0.8)  # Minimum variance = 80% of mean
    
    prob = negative_binomial_prob_at_least(projection, variance, threshold)
    
    # Cap at 96% to prevent overconfidence (even elite players miss sometimes)
    # Floor at 25% (if projection is near threshold, still some chance)
    return min(0.96, max(0.25, prob))

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
    V7.4: HIT RATE IS KING scoring model.
    
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
# V8.3: CALIBRATED LOGISTIC PROBABILITY MAPPING
# ============================================================================
def score_to_probability(score: float) -> float:
    """
    Convert parlay score to true probability using CALIBRATED logistic function.
    
    V8.3 Calibration Update:
    - A: 0.09 → 0.10 (steeper curve)
    - B: 4.8 → 6.0 (shift to reduce overconfidence)
    - Cap: 0.94 → 0.92 (backtest showed 90-95 range hit only 77.8%)
    
    P(hit) = 1 / (1 + e^(-(a × score - b)))
    """
    raw_prob = 1 / (1 + math.exp(-(LOGISTIC_A * score - LOGISTIC_B)))
    
    # Cap at 92% to prevent overconfidence (was 94%), floor at 45%
    return min(PROB_CAP, max(0.45, raw_prob))

# ============================================================================
# V8.3: PROBLEM PLAYER CHECK
# ============================================================================
def is_problem_player(player_id) -> Tuple[bool, Optional[str]]:
    """
    Check if player is on the exclusion list based on backtest data.
    
    Problem players identified from Dec 9 - Jan 6 analysis:
    - Hit rate significantly below 86.2% average
    - Shutout risk or 1-SOG miss pattern
    
    Returns: (is_problem: bool, reason: str or None)
    """
    player_id_int = int(player_id) if isinstance(player_id, str) else player_id
    
    if player_id_int in PROBLEM_PLAYERS:
        player_info = PROBLEM_PLAYERS[player_id_int]
        return True, f"Excluded: {player_info['name']} ({player_info['rate']} rate, {player_info['issue']})"
    return False, None

# ============================================================================
# V8.3: ENHANCED KILL SWITCH CHECK
# ============================================================================
def check_kill_switches(player: Dict, score: float, threshold: int) -> Tuple[bool, str]:
    """
    Enhanced kill switch logic with V8.3 calibration improvements:
    - Problem player exclusion (NEW)
    - Floor protection for P10=0 players (NEW)
    - Raised min score to 75 (was 60)
    - Relaxed variance check (backtest showed it doesn't hurt)
    
    Returns (should_kill, reason).
    """
    player_id = player.get("player_id")
    
    # Kill Switch 1: Problem player exclusion (NEW in V8.3)
    if PARLAY_CONFIG.get("exclude_problem_players", True):
        is_problem, reason = is_problem_player(player_id)
        if is_problem:
            return True, reason
    
    # Kill Switch 2: Score too low (RAISED threshold)
    min_score = PARLAY_CONFIG.get("min_score_for_parlay", KILL_SWITCHES["min_score"])
    if score < min_score:
        return True, f"Score {score:.1f} < {min_score} min"
    
    # Kill Switch 3: Too many recent shutouts
    l5_shutouts = player.get("l5_shutouts", 0)
    if l5_shutouts > KILL_SWITCHES["max_l5_shutouts"]:
        return True, f"{l5_shutouts} shutouts in L5"
    
    # Kill Switch 4: Extreme variance (RELAXED based on backtest)
    # Note: Backtest showed high variance players actually performed BETTER
    std_dev = player.get("std_dev", 1.5)
    if std_dev > KILL_SWITCHES["max_variance"]:
        return True, f"σ = {std_dev:.1f}"
    
    # Kill Switch 5: TOI dropping significantly (TIGHTENED)
    l5_toi = player.get("l5_toi", 15)
    avg_toi = player.get("avg_toi", 15)
    if avg_toi > 0:
        toi_drop = (avg_toi - l5_toi) / avg_toi * 100
        if toi_drop > KILL_SWITCHES["max_toi_drop_pct"]:
            return True, f"TOI -{toi_drop:.0f}%"
    
    # Kill Switch 6: L5 hit rate too low
    l5_hit = calculate_l5_hit_rate(player, threshold)
    if l5_hit < KILL_SWITCHES["min_l5_hit_rate"]:
        return True, f"L5 hit = {l5_hit:.0f}%"
    
    # Kill Switch 7: Floor protection (NEW in V8.3)
    # 2 of 8 misses were shutouts - P10=0 with low hit rate is risky
    p10 = player.get("p10", 1)
    hit_rate_key = f"hit_rate_{threshold}plus"
    hit_rate = player.get(hit_rate_key, player.get("hit_rate_2plus", 80))
    if p10 == 0 and hit_rate < 90:
        return True, f"Floor risk (P10=0, HR={hit_rate:.0f}%)"
    
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

def get_teams_playing_on_date(date_str: str) -> set:
    """Get set of team abbreviations playing on a given date."""
    games = get_todays_schedule(date_str)
    teams = set()
    for game in games:
        teams.add(game.get("away_team", ""))
        teams.add(game.get("home_team", ""))
    teams.discard("")  # Remove empty string if present
    return teams

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
def fetch_parlay_results_direct(check_date: str, status_container):
    """Directly fetch parlay leg results from boxscores with debug output."""
    # Store debug info persistently
    debug_log = []
    
    if check_date not in st.session_state.parlay_history:
        status_container.error(f"❌ No parlay found for {check_date}")
        return False
    
    parlay = st.session_state.parlay_history[check_date]
    if not isinstance(parlay, dict) or not parlay.get("legs"):
        status_container.error(f"❌ Invalid parlay data for {check_date}")
        return False
    
    parlay_threshold = parlay.get("threshold", 2)
    legs = parlay.get("legs", [])
    
    # Build lookup for parlay legs - show what we're looking for
    leg_by_id = {}
    leg_by_name = {}
    debug_info = []
    for i, leg in enumerate(legs):
        pid = str(leg.get("player_id", ""))
        pname = leg.get("player_name", "").lower().strip()
        if pid:
            leg_by_id[pid] = i
        if pname:
            leg_by_name[pname] = i
        debug_info.append(f"{leg.get('player_name', '?')} (ID: {pid})")
    
    debug_log.append(f"🔍 Looking for: {', '.join(debug_info)}")
    status_container.info(f"🔍 Looking for: {', '.join(debug_info)}")
    
    games = get_todays_schedule(check_date)
    if not games:
        debug_log.append(f"❌ No games found for {check_date}")
        status_container.error(f"❌ No games found for {check_date} - NHL schedule API may not have this date")
        st.session_state.parlay_debug = debug_log
        return False
    
    debug_log.append(f"📅 Found {len(games)} games on {check_date}")
    status_container.write(f"📅 Found {len(games)} games on {check_date}")
    
    legs_found = [None] * len(legs)
    all_players_seen = []
    finished_games = 0
    
    for game in games:
        try:
            box_url = f"{NHL_WEB_API}/gamecenter/{game['id']}/boxscore"
            resp = requests.get(box_url, timeout=15)
            if resp.status_code != 200:
                status_container.write(f"⚠️ Game {game['id']}: HTTP {resp.status_code}")
                continue
            
            box_data = resp.json()
            game_state = box_data.get("gameState", "")
            
            if game_state not in ["OFF", "FINAL"]:
                status_container.write(f"⏳ Game {game['away_team']}@{game['home_team']}: {game_state}")
                continue
            
            finished_games += 1
            
            # Try multiple paths for player stats
            player_locations = []
            
            # Path 1: boxscore.playerByGameStats
            if "boxscore" in box_data:
                boxscore = box_data["boxscore"]
                if "playerByGameStats" in boxscore:
                    pbgs = boxscore["playerByGameStats"]
                    for team_key in ["homeTeam", "awayTeam"]:
                        if team_key in pbgs:
                            for pos in ["forwards", "defense"]:
                                if pos in pbgs[team_key]:
                                    player_locations.extend(pbgs[team_key][pos])
            
            # Path 2: playerByGameStats at root
            if "playerByGameStats" in box_data:
                pbgs = box_data["playerByGameStats"]
                for team_key in ["homeTeam", "awayTeam"]:
                    if team_key in pbgs:
                        for pos in ["forwards", "defense"]:
                            if pos in pbgs[team_key]:
                                player_locations.extend(pbgs[team_key][pos])
            
            # Check each player against parlay legs
            for player in player_locations:
                pid = player.get("playerId") or player.get("id")
                pid_str = str(pid) if pid else ""
                
                name_data = player.get("name", {})
                if isinstance(name_data, dict):
                    player_name = name_data.get("default", "")
                else:
                    player_name = str(name_data) if name_data else ""
                player_name_lower = player_name.lower().strip()
                
                actual_sog = player.get("sog", 0) or player.get("shots", 0) or 0
                
                # Track all players for debugging
                all_players_seen.append(f"{player_name}:{pid_str}")
                
                # Check if this player is a parlay leg
                leg_idx = None
                if pid_str and pid_str in leg_by_id:
                    leg_idx = leg_by_id[pid_str]
                elif player_name_lower and player_name_lower in leg_by_name:
                    leg_idx = leg_by_name[player_name_lower]
                
                if leg_idx is not None and legs_found[leg_idx] is None:
                    legs_found[leg_idx] = actual_sog
                    hit = "✅" if actual_sog >= parlay_threshold else "❌"
                    status_container.success(f"{hit} {player_name}: {actual_sog} SOG")
            
            time.sleep(0.05)
        except Exception as e:
            status_container.write(f"⚠️ Error: {str(e)[:50]}")
            continue
    
    status_container.write(f"📊 Checked {finished_games} finished games, saw {len(all_players_seen)} players")
    debug_log.append(f"📊 Checked {finished_games} finished games, saw {len(all_players_seen)} players")
    
    # Debug: show first few players seen if we didn't find our targets
    found_count = sum(1 for x in legs_found if x is not None)
    if found_count < len(legs):
        # Show sample of players we saw
        sample = all_players_seen[:10]
        status_container.write(f"🔎 Sample players in boxscores: {', '.join(sample)}")
        debug_log.append(f"🔎 Sample: {', '.join(sample)}")
    
    # Update parlay legs with results
    legs_hit = 0
    legs_checked = 0
    for i, leg in enumerate(legs):
        if legs_found[i] is not None:
            legs_checked += 1
            actual = legs_found[i]
            leg["actual_sog"] = actual
            leg["hit"] = 1 if actual >= parlay_threshold else 0
            if leg["hit"]:
                legs_hit += 1
            debug_log.append(f"{'✅' if leg['hit'] else '❌'} {leg.get('player_name','?')}: {actual} SOG")
    
    if legs_checked > 0:
        parlay["legs_hit"] = legs_hit
        parlay["legs_checked"] = legs_checked
        
        if legs_checked == len(legs):
            parlay["result"] = "WIN" if legs_hit == len(legs) else "LOSS"
            status_container.success(f"🎯 Parlay Result: **{parlay['result']}** ({legs_hit}/{len(legs)} legs hit)")
            debug_log.append(f"🎯 Result: {parlay['result']} ({legs_hit}/{len(legs)})")
        else:
            parlay["result"] = f"PARTIAL ({legs_checked}/{len(legs)} legs checked)"
            status_container.warning(f"⚠️ Only {legs_checked}/{len(legs)} legs could be verified")
            debug_log.append(f"⚠️ PARTIAL: {legs_checked}/{len(legs)}")
        
        st.session_state.parlay_history[check_date] = parlay
        save_parlay_history(st.session_state.parlay_history, verify_date=check_date)
        st.session_state.parlay_debug = debug_log  # Save debug log
        return True
    
    debug_log.append(f"❌ Could not find any parlay players in boxscores!")
    st.session_state.parlay_debug = debug_log  # Save debug log
    status_container.error(f"❌ Could not find any of the {len(legs)} parlay leg players in boxscores")
    return False


def fetch_results(check_date: str, threshold: int, status_container):
    """Fetch results for saved picks."""
    if check_date not in st.session_state.saved_picks:
        # No picks saved - but maybe we have a parlay to fetch
        if check_date in st.session_state.parlay_history:
            parlay = st.session_state.parlay_history[check_date]
            if isinstance(parlay, dict) and parlay.get("result") is None:
                status_container.info(f"No picks for {check_date}, but found parlay - fetching...")
                result = fetch_parlay_results_direct(check_date, status_container)
                if result:
                    updated_parlay = st.session_state.parlay_history.get(check_date, {})
                    status_container.success(f"✅ Parlay result: {updated_parlay.get('result', 'Unknown')}")
                return
        status_container.warning(f"No picks saved for {check_date}")
        return
    
    picks = st.session_state.saved_picks[check_date]
    # Skip non-list values
    if not isinstance(picks, list):
        status_container.warning(f"Invalid picks data for {check_date}")
        return
    
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
        # NOTE: Parlay updates are handled by fetch_parlay_results_direct()
        # which is always called after fetch_results() from the Fetch button
    
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
    """
    Generate best parlay with V8.3 improvements:
    - Game diversification (max 1 player per game)
    - Problem player exclusion
    - Higher score threshold
    """
    
    # Filter to parlay-eligible plays (not killed)
    eligible = [p for p in plays if not p.get("killed", False)]
    
    if len(eligible) < num_legs:
        return None
    
    # Sort by score
    sorted_plays = sorted(eligible, key=lambda x: x.get("parlay_score", 0), reverse=True)
    
    # V8.3: Apply game diversification - max 1 player per game
    max_per_game = PARLAY_CONFIG.get("max_players_same_game", 1)
    games_used = {}  # game_id -> count
    best_legs = []
    
    for play in sorted_plays:
        if len(best_legs) >= num_legs:
            break
        
        # Get game_id - construct from opponent if not available
        game_id = play.get("game_id", "")
        if not game_id:
            # Construct a pseudo game_id from teams
            team = play.get("player", {}).get("team", play.get("team", ""))
            opp = play.get("opponent", "")
            game_id = f"{min(team, opp)}-{max(team, opp)}" if team and opp else ""
        
        # Check diversification
        if game_id:
            current_count = games_used.get(game_id, 0)
            if current_count >= max_per_game:
                continue  # Skip - already have player from this game
            games_used[game_id] = current_count + 1
        
        best_legs.append(play)
    
    if len(best_legs) < num_legs:
        # Not enough diversified legs - fall back to top N
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
    
    # V8.3: Check if all legs are from different games
    unique_games = len(set(games_used.keys())) if games_used else len(best_legs)
    is_diversified = unique_games >= len(best_legs)
    
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
        "is_diversified": is_diversified,
        "unique_games": unique_games,
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
    
    # Fetch defense stats (checks last 20 games per team)
    team_defense = {}
    teams_list = list(teams_playing)
    for i, team in enumerate(teams_list):
        pct = 0.05 + (i / len(teams_list)) * 0.35
        progress_bar.progress(pct)
        status_text.text(f"🛡️ {team} defense (L20)... ({i+1}/{len(teams_list)} teams)")
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
    error_count = 0
    no_games_count = 0
    low_hit_count = 0
    
    for i, player_info in enumerate(all_players):
        pct = 0.45 + (i / total) * 0.55
        progress_bar.progress(pct)
        status_text.text(f"🔍 {player_info['name'][:18]}... ({i+1}/{total}) | ✅ {qualified_count} found | ❌ {error_count} errors")
        
        # Change quote every ~40 players
        if i % 40 == 0 and i > 0:
            quote_text.caption(f"💬 *\"{random.choice(QUOTES)}\"*")
        
        try:
            stats = fetch_player_stats(player_info)
            if not stats:
                no_games_count += 1
                continue
            
            hit_rate = stats.get(f"hit_rate_{threshold}plus", 0)
            if hit_rate < MIN_HIT_RATE:
                low_hit_count += 1
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
            
            # SOG Projection (adjusted for matchup, venue, PP) - used as λ in probability model
            base_proj = (stats["last_5_avg"] * 0.4) + (stats["last_10_avg"] * 0.3) + (stats["avg_sog"] * 0.3)
            opp_factor = opp_def.get("shots_allowed_per_game", 30.0) / LEAGUE_AVG_SAG
            venue_factor = 1.03 if is_home else 0.97
            pp_factor = 1.10 if stats.get("is_pp1") else 1.0
            projection = base_proj * opp_factor * venue_factor * pp_factor
            
            # STATISTICAL PROBABILITY (Negative Binomial / Poisson model)
            # Uses projection as expected value (λ) and player's std_dev for dispersion
            # This is a TRUE probability, not a heuristic
            std_dev = stats.get("std_dev", 1.5)
            model_prob = calculate_statistical_probability(projection, std_dev, threshold) * 100
            
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
            # Opponent-based tags
            if opp in ELITE_SUPPRESSORS: tags.append("🧱")  # Tough matchup - suppressor
            if opp in SHOT_FRIENDLY: tags.append("🧀")  # Easy matchup - swiss cheese defense
            
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
                "projection": round(projection, 2),
                "tier": tier,
                "edges": edges,
                "risks": risks,
                "highlights": highlights,
                "tags": " ".join(tags),
                "killed": killed,
                "kill_reason": kill_reason,
            }
            plays.append(play)
        
        except Exception as e:
            error_count += 1
            # Log first few errors for debugging
            if error_count <= 3:
                st.warning(f"⚠️ Error on {player_info.get('name', '?')}: {str(e)[:50]}")
    
    # Clear loading UI
    progress_bar.empty()
    status_text.empty()
    quote_text.empty()
    gif_container.empty()
    
    # Show analysis summary
    if len(plays) == 0:
        st.error(f"❌ No plays found! Debug: {total} players checked, {no_games_count} insufficient games, {low_hit_count} below {MIN_HIT_RATE}% hit rate, {error_count} errors")
    
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
        "projection": p.get("projection", 0),
        "std_dev": p["player"].get("std_dev", 1.5),
        "hit_rate": p["player"].get(f"hit_rate_{threshold}plus", 0),
        "threshold": threshold,
    } for p in plays]
    
    st.session_state.saved_picks[date_str] = picks_to_save
    
    # Auto-save recommended parlay for tracking
    auto_save_parlay(plays, date_str, threshold)
    
    return plays

def auto_save_parlay(plays: List[Dict], date_str: str, threshold: int):
    """Automatically generate and save the recommended parlay when analysis runs."""
    if not plays:
        return
    
    # Initialize debug log
    if "save_debug" not in st.session_state:
        st.session_state.save_debug = []
    
    # CRITICAL: Check BOTH session state AND cloud for existing resolved parlay
    existing = st.session_state.parlay_history.get(date_str)
    existing_result = existing.get("result") if isinstance(existing, dict) else None
    
    # Also check cloud data (in case session state is stale)
    if existing_result is None:
        cloud_data = jsonbin_load_data("parlay_bin_id")
        if cloud_data and date_str in cloud_data:
            cloud_parlay = cloud_data[date_str]
            cloud_result = cloud_parlay.get("result") if isinstance(cloud_parlay, dict) else None
            if cloud_result in ["WIN", "LOSS"]:
                # Cloud has resolved result - sync to session state and skip
                st.session_state.parlay_history[date_str] = cloud_parlay
                st.session_state.save_debug.append(f"⏭️ Skipping auto_save_parlay for {date_str} - cloud has result: {cloud_result}")
                return
    
    if existing_result is not None:
        # Parlay already has result (WIN/LOSS/PARTIAL) - don't overwrite
        st.session_state.save_debug.append(f"⏭️ Skipping auto_save_parlay for {date_str} - already has result: {existing_result}")
        return
    
    # VALIDATION: Get teams playing on this date and filter plays
    teams_playing = get_teams_playing_on_date(date_str)
    if teams_playing:
        # Only include players whose teams are playing on this date
        validated_plays = [p for p in plays if p.get("team") in teams_playing]
        if len(validated_plays) < len(plays):
            excluded = len(plays) - len(validated_plays)
            # Log excluded players for debugging
            if "save_debug" not in st.session_state:
                st.session_state.save_debug = []
            st.session_state.save_debug.append(f"⚠️ Parlay validation: excluded {excluded} players (teams not playing on {date_str})")
        plays = validated_plays
    
    # Filter to eligible plays (LOCK, STRONG, SOLID with score >= 65)
    eligible = [p for p in plays if p["parlay_score"] >= 65 and p["tier"] in ["🔒 LOCK", "✅ STRONG", "📊 SOLID"]]
    
    if len(eligible) < 2:
        # Fall back to top plays by score
        eligible = sorted(plays, key=lambda x: x["parlay_score"], reverse=True)[:10]
    
    if len(eligible) < 2:
        return
    
    # Find best parlay with prob > 30%
    best_parlay = None
    for size in [3, 2, 4, 5]:
        if size > len(eligible):
            continue
        parlay = generate_best_parlay_v7(eligible, size, threshold)
        if parlay and parlay["adjusted_prob"] >= 0.30:
            best_parlay = parlay
            break
    
    if not best_parlay:
        best_parlay = generate_best_parlay_v7(eligible, 2, threshold)
    
    if best_parlay:
        parlay_to_save = {
            "date": date_str,
            "threshold": threshold,
            "num_legs": best_parlay["num_legs"],
            "probability": best_parlay["adjusted_prob"],
            "odds": best_parlay["adjusted_odds"],
            "legs": [{
                "player_id": leg["player"]["player_id"],
                "player_name": leg["player"]["name"],
                "team": leg["player"]["team"],
                "score": leg["parlay_score"],
                "projection": leg.get("projection", 0),
            } for leg in best_parlay["legs"]],
            "result": None,
            "legs_hit": None,
        }
        st.session_state.parlay_history[date_str] = parlay_to_save
        save_parlay_history(st.session_state.parlay_history, verify_date=date_str)

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_all_results(plays: List[Dict], threshold: int):
    """Display all results with V7.4 highlighting."""
    
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
            "Proj": p.get("projection", 0),
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
            "Tags": st.column_config.TextColumn("Tags", width="small", help="⚡=PP1 | 🛡️=Floor≥1 | 🔥=Streak | B2B=Back-to-back | 🧱=vs Suppressor | 🧀=vs Swiss Cheese"),
            "Hit%": st.column_config.TextColumn("Hit%", width="small"),
            "Cush%": st.column_config.TextColumn("Cush%", width="small"),
            "Shut%": st.column_config.TextColumn("Shut%", width="small"),
            "Proj": st.column_config.NumberColumn("Proj", format="%.1f", width="small", help="Tonight's projected SOG"),
            "Avg": st.column_config.NumberColumn("Avg", format="%.2f", width="small"),
            "L5": st.column_config.NumberColumn("L5", format="%.1f", width="small"),
            "L10": st.column_config.NumberColumn("L10", format="%.1f", width="small"),
            "SOG/60": st.column_config.NumberColumn("SOG/60", format="%.1f", width="small"),
            "TOI": st.column_config.TextColumn("TOI", width="small"),
            "Prob": st.column_config.ProgressColumn(
                "Prob%",
                help="Statistical probability (Negative Binomial/Poisson model)",
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
            "Proj": p.get("projection", 0),
            "Avg": player["avg_sog"],
            "L5": player["last_5_avg"],
            "L10": player.get("last_10_avg", player["avg_sog"]),
            "SOG/60": player.get("sog_per_60", 0),
            "TOI": toi_str,
            "Prob": f"{p['model_prob']:.0f}%",
        })
    csv_df = pd.DataFrame(csv_rows)
    csv = csv_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"nhl_sog_v74_{get_est_date()}.csv", "text/csv")

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
                    st.caption(f"Score: {p['parlay_score']:.0f} | Hit: {hit_rate:.0f}% | Proj: {p.get('projection', 0):.1f} | Prob: {p['model_prob']:.0f}%")
                
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
    
    # Recommended parlay - MINIMUM 3 legs for decent odds
    st.subheader("⭐ Recommended Parlay")
    
    # Find best parlay with prob > 30% and within variance limit
    # Only consider 3+ legs (2 legs has bad odds at sportsbooks)
    best_parlay = None
    for size in [3, 4, 5, 6]:  # Start at 3, no 2-leg parlays
        if size > len(eligible):
            continue
        parlay = generate_best_parlay_v7(eligible, size, threshold)
        if parlay and parlay["adjusted_prob"] >= 0.30 and size <= max_recommended:
            best_parlay = parlay
            break
    
    # Fallback to 3 if nothing better found
    if not best_parlay and len(eligible) >= 3:
        best_parlay = generate_best_parlay_v7(eligible, 3, threshold)
    
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
            st.markdown(f"- **{player['name']}** ({player['team']}) O{threshold-0.5} SOG | Proj: {leg.get('projection', 0):.1f} | Score: {leg['parlay_score']:.0f}")
        
        # Copy button
        copy_text = f"🏒 NHL SOG Parlay ({best_parlay['num_legs']}-leg)\n"
        copy_text += f"Prob: {best_parlay['adjusted_prob']*100:.0f}% | Odds: {best_parlay['adjusted_odds']:+d}\n\n"
        for leg in best_parlay["legs"]:
            copy_text += f"• {leg['player']['name']} O{threshold-0.5} SOG\n"
        
        st.code(copy_text, language=None)
        
        # Save recommended parlay for tracking (only if no existing result)
        date_str = st.session_state.get("analysis_date", get_est_date())
        existing = st.session_state.parlay_history.get(date_str)
        existing_result = existing.get("result") if isinstance(existing, dict) else None
        
        # Only save if there's no existing resolved parlay for this date
        if existing_result not in ["WIN", "LOSS"]:
            parlay_to_save = {
                "date": date_str,
                "threshold": threshold,
                "num_legs": best_parlay["num_legs"],
                "probability": best_parlay["adjusted_prob"],
                "odds": best_parlay["adjusted_odds"],
                "legs": [{
                    "player_id": leg["player"]["player_id"],
                    "player_name": leg["player"]["name"],
                    "team": leg["player"]["team"],
                    "score": leg["parlay_score"],
                    "projection": leg.get("projection", 0),
                } for leg in best_parlay["legs"]],
                "result": None,  # Will be updated when results fetched
                "legs_hit": None,
            }
            st.session_state.parlay_history[date_str] = parlay_to_save
            save_parlay_history(st.session_state.parlay_history, verify_date=date_str)
    else:
        st.warning(f"⚠️ Not enough eligible players for a 3-leg parlay (need 3, have {len(eligible)}). Run analysis with lower filters or wait for more games.")

def display_results_tracker(threshold: int):
    """Display results tracking tab - clean UI."""
    
    # ================================================================
    # HEADER WITH STORAGE STATUS
    # ================================================================
    col_header, col_status = st.columns([3, 1])
    with col_header:
        st.subheader("📈 Results Tracker")
    with col_status:
        if is_cloud_connected():
            st.success("☁️ Synced")
        else:
            st.warning("💾 Local")
    
    # ================================================================
    # QUICK ACTIONS BAR
    # ================================================================
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        check_date = st.date_input("📅 Date", value=datetime.now(EST).date(), label_visibility="collapsed")
        check_date_str = check_date.strftime("%Y-%m-%d")
    
    with col2:
        if st.button("🔄 Fetch", type="primary", use_container_width=True):
            status = st.empty()
            
            # Clear previous parlay debug
            st.session_state.parlay_debug = []
            st.session_state.parlay_debug.append(f"🔄 Fetch clicked for {check_date_str}")
            
            fetch_results(check_date_str, threshold, status)
            
            # ALWAYS run fetch_parlay_results_direct to ensure leg actuals are properly set
            # Don't rely on fetch_results() to update parlay - it may set result but not leg actuals
            if check_date_str in st.session_state.parlay_history:
                parlay = st.session_state.parlay_history[check_date_str]
                if isinstance(parlay, dict) and parlay.get("legs"):
                    st.session_state.parlay_debug.append("🎰 Running fetch_parlay_results_direct...")
                    fetch_parlay_results_direct(check_date_str, status)
                    
                    # Log final state
                    parlay = st.session_state.parlay_history[check_date_str]
                    final_result = parlay.get("result") if isinstance(parlay, dict) else None
                    final_legs_hit = parlay.get("legs_hit") if isinstance(parlay, dict) else None
                    st.session_state.parlay_debug.append(f"🎯 Final: result={final_result}, legs_hit={final_legs_hit}")
                    
                    # Check if legs have actuals
                    legs_with_actual = sum(1 for leg in parlay.get("legs", []) if leg.get("actual_sog") is not None)
                    st.session_state.parlay_debug.append(f"📊 Legs with actual SOG: {legs_with_actual}/{len(parlay.get('legs', []))}")
                else:
                    st.session_state.parlay_debug.append("⚠️ Invalid parlay data for this date")
            else:
                st.session_state.parlay_debug.append("⚠️ No parlay found for this date")
            
            st.rerun()
    
    with col3:
        # Retry Save button - for when Fetch succeeds locally but JSONBin verification fails
        if st.button("💾 Retry Save", use_container_width=True):
            if check_date_str in st.session_state.parlay_history:
                parlay = st.session_state.parlay_history[check_date_str]
                if isinstance(parlay, dict) and parlay.get("result"):
                    st.session_state.save_debug = []
                    st.session_state.save_debug.append(f"💾 Retry Save clicked for {check_date_str}")
                    st.session_state.save_debug.append(f"   Session state: result={parlay.get('result')}, legs_hit={parlay.get('legs_hit')}")
                    
                    # Force save with verification
                    save_parlay_history(st.session_state.parlay_history, verify_date=check_date_str)
                    st.rerun()
                else:
                    st.warning(f"No resolved result for {check_date_str} - run Fetch first")
            else:
                st.warning(f"No parlay for {check_date_str}")
    
    with col4:
        if st.button("⚙️ Tools", use_container_width=True):
            st.session_state.show_tools = not st.session_state.get("show_tools", False)
            st.rerun()
    
    # Show tools panel if enabled
    if st.session_state.get("show_tools", False):
        st.markdown("---")
        st.markdown("##### ⚙️ Tools")
        tool_col1, tool_col2, tool_col3 = st.columns(3)
        
        with tool_col1:
            if st.button("☁️ Reload from Cloud", use_container_width=True):
                # Force reload from JSONBin to see what's actually stored
                cloud_parlay = jsonbin_load_data("parlay_bin_id")
                if cloud_parlay:
                    st.session_state.parlay_history = cloud_parlay
                    st.success("✅ Reloaded from JSONBin")
                    st.rerun()
                else:
                    st.error("❌ Could not load from JSONBin")
        
        with tool_col2:
            if st.button("🔍 Compare Session vs Cloud", use_container_width=True):
                cloud_parlay = jsonbin_load_data("parlay_bin_id")
                session_parlay = st.session_state.parlay_history.get(check_date_str)
                cloud_entry = cloud_parlay.get(check_date_str) if cloud_parlay else None
                
                st.markdown(f"**Date: {check_date_str}**")
                st.markdown("**Session State:**")
                if isinstance(session_parlay, dict):
                    st.json({"result": session_parlay.get("result"), "legs_hit": session_parlay.get("legs_hit")})
                else:
                    st.write("No data")
                
                st.markdown("**JSONBin:**")
                if isinstance(cloud_entry, dict):
                    st.json({"result": cloud_entry.get("result"), "legs_hit": cloud_entry.get("legs_hit")})
                else:
                    st.write("No data")
        
        with tool_col3:
            if st.button("🗑️ Clear Debug Logs", use_container_width=True):
                st.session_state.parlay_debug = []
                st.session_state.save_debug = []
                st.rerun()
        
        # Manual Parlay Override Row
        st.markdown("##### ✏️ Manual Parlay Override")
        if check_date_str in st.session_state.parlay_history:
            parlay = st.session_state.parlay_history[check_date_str]
            if isinstance(parlay, dict):
                current_result = parlay.get("result", "None")
                num_legs = parlay.get("num_legs", len(parlay.get("legs", [])))
                
                st.caption(f"Current: **{current_result}** | Legs: {num_legs}")
                
                override_col1, override_col2, override_col3, override_col4 = st.columns(4)
                
                with override_col1:
                    legs_hit_input = st.number_input("Legs Hit", min_value=0, max_value=num_legs, 
                                                     value=parlay.get("legs_hit") or 0, key="manual_legs_hit")
                
                with override_col2:
                    if st.button("✅ Set WIN", use_container_width=True):
                        parlay["result"] = "WIN"
                        parlay["legs_hit"] = legs_hit_input if legs_hit_input > 0 else num_legs
                        st.session_state.parlay_history[check_date_str] = parlay
                        save_parlay_history(st.session_state.parlay_history, verify_date=check_date_str)
                        st.success("Set to WIN")
                        st.rerun()
                
                with override_col3:
                    if st.button("❌ Set LOSS", use_container_width=True):
                        parlay["result"] = "LOSS"
                        parlay["legs_hit"] = legs_hit_input
                        st.session_state.parlay_history[check_date_str] = parlay
                        save_parlay_history(st.session_state.parlay_history, verify_date=check_date_str)
                        st.success("Set to LOSS")
                        st.rerun()
                
                with override_col4:
                    if st.button("⚪ Set VOID", use_container_width=True):
                        parlay["result"] = "VOID"
                        parlay["legs_hit"] = legs_hit_input
                        st.session_state.parlay_history[check_date_str] = parlay
                        save_parlay_history(st.session_state.parlay_history, verify_date=check_date_str)
                        st.success("Set to VOID")
                        st.rerun()
        else:
            st.caption(f"No parlay exists for {check_date_str}")
    
    st.markdown("---")
    
    # Show debug log if exists (from last Fix Parlay attempt)
    if "parlay_debug" in st.session_state and st.session_state.parlay_debug:
        with st.expander("🔧 Last Parlay Fetch Debug Log", expanded=False):
            for line in st.session_state.parlay_debug:
                st.write(line)
            if st.button("Clear Debug Log"):
                st.session_state.parlay_debug = []
                st.rerun()
    
    # Show save debug if exists
    if "save_debug" in st.session_state and st.session_state.save_debug:
        with st.expander("💾 Save Debug Log", expanded=True):
            for line in st.session_state.save_debug:
                if "❌" in line:
                    st.error(line)
                else:
                    st.success(line)
            if st.button("Clear Save Log"):
                st.session_state.save_debug = []
                st.rerun()
    
    # ================================================================
    # DATA MANAGEMENT: Export & Prune
    # ================================================================
    st.markdown("##### 📦 Data Management")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Export Parlay History
        parlay_data = st.session_state.get("parlay_history", {})
        if parlay_data:
            parlay_json = json.dumps(parlay_data, indent=2, default=str)
            st.download_button(
                label="📥 Export Parlays",
                data=parlay_json,
                file_name=f"parlay_history_{get_est_date()}.json",
                mime="application/json",
                use_container_width=True,
                help=f"Download {len(parlay_data)} parlay records as JSON"
            )
        else:
            st.button("📥 Export Parlays", disabled=True, use_container_width=True)
    
    with export_col2:
        # Export Results History
        results_data = st.session_state.get("results_history", {})
        if results_data:
            results_json = json.dumps(results_data, indent=2, default=str)
            st.download_button(
                label="📥 Export Results",
                data=results_json,
                file_name=f"results_history_{get_est_date()}.json",
                mime="application/json",
                use_container_width=True,
                help=f"Download {len(results_data)} days of pick results as JSON"
            )
        else:
            st.button("📥 Export Results", disabled=True, use_container_width=True)
    
    with export_col3:
        # Export Both Combined
        combined_data = {
            "exported_at": get_est_date(),
            "parlay_history": st.session_state.get("parlay_history", {}),
            "results_history": st.session_state.get("results_history", {})
        }
        combined_json = json.dumps(combined_data, indent=2, default=str)
        st.download_button(
            label="📥 Export ALL",
            data=combined_json,
            file_name=f"sharpslip_backup_{get_est_date()}.json",
            mime="application/json",
            use_container_width=True,
            help="Download complete backup (parlays + results)"
        )
    
    # Prune Section
    st.markdown("##### 🗑️ Prune Old Data")
    prune_col1, prune_col2, prune_col3 = st.columns([2, 2, 1])
    
    with prune_col1:
        keep_days = st.number_input(
            "Keep last N days", 
            min_value=0, 
            max_value=90, 
            value=7,
            help="Set to 0 to delete ALL data"
        )
    
    with prune_col2:
        cutoff_date = (datetime.now(EST) - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        
        # Count records to prune
        parlay_to_prune = [d for d in st.session_state.get("parlay_history", {}).keys() if d < cutoff_date]
        results_to_prune = [d for d in st.session_state.get("results_history", {}).keys() if d < cutoff_date]
        
        st.caption(f"**Cutoff:** {cutoff_date}")
        st.caption(f"**Will delete:** {len(parlay_to_prune)} parlays, {len(results_to_prune)} result days")
    
    with prune_col3:
        if st.button("🗑️ Prune", type="secondary", use_container_width=True):
            st.session_state.confirm_prune = True
            st.rerun()
    
    # Confirmation dialog
    if st.session_state.get("confirm_prune", False):
        st.warning(f"⚠️ This will permanently delete {len(parlay_to_prune)} parlays and {len(results_to_prune)} result days older than {cutoff_date}")
        
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                # Prune parlay history
                for date_key in parlay_to_prune:
                    del st.session_state.parlay_history[date_key]
                
                # Prune results history
                for date_key in results_to_prune:
                    del st.session_state.results_history[date_key]
                
                # Save pruned data to JSONBin
                save_parlay_history(st.session_state.parlay_history)
                save_history(st.session_state.results_history)
                
                st.session_state.confirm_prune = False
                st.success(f"✅ Pruned {len(parlay_to_prune)} parlays and {len(results_to_prune)} result days")
                st.rerun()
        
        with confirm_col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_prune = False
                st.rerun()
    
    # ================================================================
    # PARLAY RESULT CARD (if exists for selected date)
    # ================================================================
    if check_date_str in st.session_state.parlay_history:
        parlay = st.session_state.parlay_history[check_date_str]
        if isinstance(parlay, dict) and parlay.get("legs"):
            result = parlay.get("result")
            legs_hit = parlay.get("legs_hit", 0)
            num_legs = parlay.get("num_legs", len(parlay.get("legs", [])))
            
            # Parlay card
            if result == "WIN":
                st.success(f"### 🎉 PARLAY WIN ({legs_hit}/{num_legs} legs)")
            elif result == "LOSS":
                st.error(f"### ❌ PARLAY LOSS ({legs_hit}/{num_legs} legs)")
            else:
                st.info(f"### ⏳ PARLAY PENDING ({num_legs} legs)")
            
            # Leg details (compact table)
            leg_rows = []
            for leg in parlay.get("legs", []):
                actual = leg.get("actual_sog", "?")
                hit = leg.get("hit", None)
                icon = "✅" if hit == 1 else "❌" if hit == 0 else "⏳"
                leg_rows.append({
                    "": icon,
                    "Player": leg.get("player_name", "?"),
                    "Proj": f"{leg.get('projection', 0):.1f}",
                    "Actual": actual,
                    "Score": f"{leg.get('score', 0):.0f}",
                })
            st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True, height=150)
    
    # ================================================================
    # SELECTED DATE RESULTS (if exists)
    # ================================================================
    if check_date_str in st.session_state.results_history:
        picks = st.session_state.results_history[check_date_str]
        if isinstance(picks, list):
            picks_with_results = [p for p in picks if "actual_sog" in p]
            
            if picks_with_results:
                hits = sum(1 for p in picks_with_results if p.get("hit", 0) == 1)
                total = len(picks_with_results)
                pct = hits/total*100 if total > 0 else 0
                
                st.success(f"**{check_date_str}:** {hits}/{total} ({pct:.0f}% hit rate)")
                
                # Expandable details - just player list (tier breakdown is in Overall section)
                with st.expander(f"📋 Details ({total} picks)", expanded=False):
                    detail_data = []
                    for p in sorted(picks_with_results, key=lambda x: x.get("parlay_score", 0), reverse=True):
                        detail_data.append({
                            "": "✅" if p.get("hit", 0) == 1 else "❌",
                            "Player": p["player"],
                            "Team": p["team"],
                            "Score": f"{p['parlay_score']:.0f}",
                            "Tier": p["tier"].split()[0],  # Just emoji
                            "Actual": p.get("actual_sog", "?"),
                        })
                    st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ================================================================
    # OVERALL PERFORMANCE SUMMARY
    # ================================================================
    st.subheader("📊 Overall Performance")
    
    # Calculate parlay stats
    parlay_wins = 0
    parlay_total = 0
    parlay_prob_sum = 0
    for p in st.session_state.parlay_history.values():
        if isinstance(p, dict) and p.get("result"):
            parlay_total += 1
            if p["result"] == "WIN":
                parlay_wins += 1
            parlay_prob_sum += p.get("probability", 0) * 100
    
    # Calculate pick stats
    all_picks = []
    for picks in st.session_state.results_history.values():
        if isinstance(picks, list):
            for p in picks:
                if "actual_sog" in p:
                    all_picks.append(p)
    
    total_hits = sum(1 for p in all_picks if p.get("hit", 0) == 1)
    total_picks = len(all_picks)
    
    # 4-column summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎰 Parlay Record", f"{parlay_wins}-{parlay_total - parlay_wins}" if parlay_total > 0 else "0-0")
    col2.metric("Parlay Win%", f"{parlay_wins/parlay_total*100:.0f}%" if parlay_total > 0 else "—")
    col3.metric("🎯 Pick Record", f"{total_hits}/{total_picks}" if total_picks > 0 else "0/0")
    col4.metric("Pick Win%", f"{total_hits/total_picks*100:.0f}%" if total_picks > 0 else "—")
    
    # Performance by tier (compact)
    if all_picks:
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
                summary_data.append({"Tier": tier, "W": data["hits"], "L": data["total"] - data["hits"], "Win%": f"{pct:.0f}%"})
        
        if summary_data:
            with st.expander("📊 Performance by Tier", expanded=True):
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    # ================================================================
    # MODEL ANALYTICS (collapsed by default)
    # ================================================================
    if all_picks and len(all_picks) >= 10:
        with st.expander("🎯 Model Analytics", expanded=False):
            # Calibration
            st.markdown("##### Calibration")
            st.caption("Predicted probability vs actual hit rate")
            
            prob_buckets = {
                "80-100%": {"pred_sum": 0, "hits": 0, "total": 0, "min": 80, "max": 100},
                "70-79%": {"pred_sum": 0, "hits": 0, "total": 0, "min": 70, "max": 79},
                "60-69%": {"pred_sum": 0, "hits": 0, "total": 0, "min": 60, "max": 69},
                "<60%": {"pred_sum": 0, "hits": 0, "total": 0, "min": 0, "max": 59},
            }
            
            for p in all_picks:
                prob = p.get("model_prob", 0)
                hit = p.get("hit", 0)
                for bucket_name, bucket in prob_buckets.items():
                    if bucket["min"] <= prob <= bucket["max"]:
                        bucket["pred_sum"] += prob
                        bucket["hits"] += hit
                        bucket["total"] += 1
                        break
            
            cal_data = []
            for bucket_name in ["80-100%", "70-79%", "60-69%", "<60%"]:
                bucket = prob_buckets[bucket_name]
                if bucket["total"] > 0:
                    avg_pred = bucket["pred_sum"] / bucket["total"]
                    actual_rate = bucket["hits"] / bucket["total"] * 100
                    diff = actual_rate - avg_pred
                    cal_data.append({
                        "Bucket": bucket_name,
                        "N": bucket["total"],
                        "Pred": f"{avg_pred:.0f}%",
                        "Actual": f"{actual_rate:.0f}%",
                        "Diff": f"{diff:+.0f}%",
                    })
            
            if cal_data:
                st.dataframe(pd.DataFrame(cal_data), use_container_width=True, hide_index=True)
            
            # Projection accuracy
            st.markdown("##### Projection Accuracy")
            proj_picks = [p for p in all_picks if p.get("projection", 0) > 0]
            if proj_picks:
                avg_proj = sum(p["projection"] for p in proj_picks) / len(proj_picks)
                avg_actual = sum(p.get("actual_sog", 0) for p in proj_picks) / len(proj_picks)
                bias = avg_actual - avg_proj
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Proj", f"{avg_proj:.2f}")
                col2.metric("Avg Actual", f"{avg_actual:.2f}")
                col3.metric("Bias", f"{bias:+.2f}", "Under" if bias > 0.2 else "Over" if bias < -0.2 else "✓")
    
    # ================================================================
    # TOOLS (collapsed, only shown when toggled)
    # ================================================================
    if st.session_state.get("show_tools", False):
        st.markdown("---")
        with st.expander("⚙️ Tools & Settings", expanded=True):
            tool_tab1, tool_tab2, tool_tab3 = st.tabs(["☁️ Storage", "📅 Bulk Backfill", "💾 Backup"])
            
            with tool_tab1:
                if is_cloud_connected():
                    st.success("✅ JSONBin Connected")
                    real_results = sum(1 for v in st.session_state.results_history.values() if isinstance(v, list))
                    real_parlays = sum(1 for v in st.session_state.parlay_history.values() if isinstance(v, dict) and "legs" in v)
                    st.caption(f"📊 {real_results} days of results | 🎰 {real_parlays} parlays")
                else:
                    st.warning("JSONBin not configured")
                    st.markdown("""
                    **Quick Setup:**
                    1. [jsonbin.io](https://jsonbin.io) → Sign up free
                    2. Get API Key from profile
                    3. Create 2 bins (empty `{}`)
                    4. Add to Streamlit secrets:
                    ```toml
                    [jsonbin]
                    api_key = "your_key"
                    results_bin_id = "bin_id_1"
                    parlay_bin_id = "bin_id_2"
                    ```
                    """)
            
            with tool_tab2:
                col_start, col_end = st.columns(2)
                with col_start:
                    bulk_start = st.date_input("Start", value=datetime.now(EST).date() - timedelta(days=7), key="bulk_start")
                with col_end:
                    bulk_end = st.date_input("End", value=datetime.now(EST).date() - timedelta(days=1), key="bulk_end")
                
                if st.button("🚀 Bulk Fetch"):
                    if bulk_start <= bulk_end:
                        dates = []
                        current = bulk_start
                        while current <= bulk_end:
                            date_str = current.strftime("%Y-%m-%d")
                            if date_str in st.session_state.saved_picks:
                                dates.append(date_str)
                            current += timedelta(days=1)
                        
                        if dates:
                            progress = st.progress(0)
                            for i, d in enumerate(dates):
                                progress.progress((i+1)/len(dates))
                                fetch_results(d, threshold, st.empty())
                            st.success(f"✅ Fetched {len(dates)} days")
                            st.rerun()
                        else:
                            st.warning("No saved picks in range")
            
            with tool_tab3:
                # Export
                backup_data = {
                    "results_history": st.session_state.results_history,
                    "parlay_history": st.session_state.parlay_history,
                    "exported_at": get_est_datetime().isoformat(),
                }
                st.download_button(
                    "📥 Export Backup",
                    data=json.dumps(backup_data, indent=2),
                    file_name=f"nhl_sog_backup_{get_est_date()}.json",
                    mime="application/json"
                )
                
                # Import
                uploaded = st.file_uploader("📤 Import Backup", type=["json"])
                if uploaded:
                    try:
                        imported = json.load(uploaded)
                        if st.button("✅ Restore"):
                            for date, picks in imported.get("results_history", {}).items():
                                if date not in st.session_state.results_history:
                                    st.session_state.results_history[date] = picks
                            for date, parlay in imported.get("parlay_history", {}).items():
                                if date not in st.session_state.parlay_history:
                                    st.session_state.parlay_history[date] = parlay
                            save_history(st.session_state.results_history)
                            save_parlay_history(st.session_state.parlay_history)
                            st.success("✅ Restored!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Show message if no data
    if not all_picks and parlay_total == 0:
        st.info("📊 No data yet. Run analysis and fetch results to start tracking.")
# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🏒 NHL SOG Analyzer V8.3")
    st.caption("Calibration: Game diversification, floor protection, LOCK=90")
    
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
        
        # V7.4 Model Info
        with st.expander("ℹ️ V7.4 Statistical Model"):
            st.markdown("""
            **🎲 Probability Model (NEW in V7.4):**
            
            Uses **Negative Binomial distribution** (or Poisson when appropriate):
            
            ```
            λ = Projection (adjusted expected SOG)
            σ² = Player's variance (from std_dev)
            P(hit) = 1 - NegBinom.CDF(threshold-1, μ=λ, σ²)
            ```
            
            **Why Negative Binomial?**
            - SOG data is often *overdispersed* (variance > mean)
            - Poisson assumes variance = mean (too rigid)
            - NB captures the extra variability in player performance
            
            **Projection Formula:**
            ```
            λ = (L5×0.4 + L10×0.3 + Avg×0.3) × opp × venue × PP
            ```
            
            ---
            
            **📊 Parlay Score** (Quality Heuristic):
            - Base = 50 + (Hit% - 70) × 1.5
            - Modifiers for matchup, form, floor, PP, etc.
            - Used for tiering and parlay selection
            
            **🎯 Prob%** = TRUE statistical probability
            
            **🏆 Score** = Quality/confidence indicator
            
            ---
            
            **V8.3 Tiers:**
            - 🔒 LOCK: 90+
            - ✅ STRONG: 80-89
            - 📊 SOLID: 70-79
            - ⚠️ RISKY: 60-69
            - ❌ AVOID: <60
            
            **V8.3 Calibration:**
            - Game diversification (1 per game)
            - Problem player exclusion
            - Floor protection (P10=0, HR<90%)
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
            st.session_state.analysis_date = date_str  # Store the analysis date
    
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
