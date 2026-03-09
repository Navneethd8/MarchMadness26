"""
March Madness 2026 — PuLP Bracket Optimizer
============================================
Formulates the NCAA tournament bracket as an Integer Linear Program (ILP).
Maximises the expected ESPN-style bracket score by choosing the most likely
winners in each tournament slot.

Uses:
  - Historical seed-vs-seed win rates (1985-2025)
  - 2026 regular-season team strength (scoring margin, win %)
  - Massey Ordinal composite rankings
  - Tournament seeds & bracket structure (uses 2025 bracket template)
"""

import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus
import warnings, sys, os

warnings.filterwarnings("ignore")

DATA = "data"

# ───────────────────────────── 1. Load Data ─────────────────────────────

def load_data(prefix):
    teams      = pd.read_csv(f"{DATA}/{prefix}Teams.csv")
    seeds      = pd.read_csv(f"{DATA}/{prefix}NCAATourneySeeds.csv")
    slots      = pd.read_csv(f"{DATA}/{prefix}NCAATourneySlots.csv")
    tourney    = pd.read_csv(f"{DATA}/{prefix}NCAATourneyCompactResults.csv")
    reg_season = pd.read_csv(f"{DATA}/{prefix}RegularSeasonCompactResults.csv")
    if prefix == "M":
        ordinals   = pd.read_csv(f"{DATA}/{prefix}MasseyOrdinals.csv")
    else:
        ordinals = pd.DataFrame()
    return teams, seeds, slots, tourney, reg_season, ordinals


# ───────────────────── 2. Seed-Based Win Probability ─────────────────────

def seed_number(seed_str):
    """Extract numeric seed from strings like 'W01', 'X16a'."""
    return int("".join(c for c in seed_str[1:] if c.isdigit()))


def build_seed_win_rates(seeds, tourney):
    """Historical win rate for seed X vs seed Y (1985-2025)."""
    df = tourney.merge(seeds.rename(columns={"TeamID": "WTeamID", "Seed": "WSeed"}),
                       on=["Season", "WTeamID"])
    df = df.merge(seeds.rename(columns={"TeamID": "LTeamID", "Seed": "LSeed"}),
                  on=["Season", "LTeamID"])
    df["WSeedNum"] = df["WSeed"].apply(seed_number)
    df["LSeedNum"] = df["LSeed"].apply(seed_number)

    # Count wins for each seed-pair
    wins = {}
    for _, row in df.iterrows():
        s1, s2 = row["WSeedNum"], row["LSeedNum"]
        key = (min(s1, s2), max(s1, s2))
        if key not in wins:
            wins[key] = {s1: 0, s2: 0}
        wins[key][s1] = wins[key].get(s1, 0) + 1

    # Convert to win probabilities
    seed_prob = {}
    for (s1, s2), counts in wins.items():
        total = sum(counts.values())
        for s, c in counts.items():
            opp = s2 if s == s1 else s1
            seed_prob[(s, opp)] = c / total

    return seed_prob


# ─────────────── 3. Team Strength from Regular Season ───────────────

def build_team_strength(reg_season, season=2026):
    """Compute per-team stats for the given season including SOS and Upsets."""
    rs = reg_season[reg_season["Season"] == season].copy()

    # Winner perspective
    w = rs.groupby("WTeamID").agg(
        W_count=("WScore", "size"),
        W_pts=("WScore", "sum"),
        W_opp_pts=("LScore", "sum"),
    ).reset_index().rename(columns={"WTeamID": "TeamID"})

    # Loser perspective
    l = rs.groupby("LTeamID").agg(
        L_count=("LScore", "size"),
        L_pts=("LScore", "sum"),
        L_opp_pts=("WScore", "sum"),
    ).reset_index().rename(columns={"LTeamID": "TeamID"})

    stats = w.merge(l, on="TeamID", how="outer").fillna(0)
    stats["Games"]  = stats["W_count"] + stats["L_count"]
    stats["WinPct"] = stats["W_count"] / stats["Games"]
    stats["PPG"]    = (stats["W_pts"] + stats["L_pts"]) / stats["Games"]
    stats["OppPPG"] = (stats["W_opp_pts"] + stats["L_opp_pts"]) / stats["Games"]
    stats["Margin"] = stats["PPG"] - stats["OppPPG"]
    
    # Calculate Strength of Schedule (SOS) - average opponent's win percentage
    # (Simple variant: Opponents Win %) 
    opp_wins = {}
    opp_games = {}
    
    for _, row in rs.iterrows():
        w_id = row["WTeamID"]
        l_id = row["LTeamID"]
        
        # Track opponents for both teams
        opp_wins[w_id] = opp_wins.get(w_id, 0) + stats[stats["TeamID"] == l_id]["W_count"].values[0] if not stats[stats["TeamID"] == l_id].empty else opp_wins.get(w_id, 0)
        opp_games[w_id] = opp_games.get(w_id, 0) + stats[stats["TeamID"] == l_id]["Games"].values[0] if not stats[stats["TeamID"] == l_id].empty else opp_games.get(w_id, 0)
        
        opp_wins[l_id] = opp_wins.get(l_id, 0) + stats[stats["TeamID"] == w_id]["W_count"].values[0] if not stats[stats["TeamID"] == w_id].empty else opp_wins.get(l_id, 0)
        opp_games[l_id] = opp_games.get(l_id, 0) + stats[stats["TeamID"] == w_id]["Games"].values[0] if not stats[stats["TeamID"] == w_id].empty else opp_games.get(l_id, 0)
        
    stats["SOS"] = stats["TeamID"].apply(lambda x: opp_wins.get(x, 0) / opp_games.get(x, 1) if opp_games.get(x, 1) > 0 else 0)
    
    return stats[["TeamID", "Games", "WinPct", "PPG", "OppPPG", "Margin", "SOS"]]


# ──────────────── 4. Massey Ordinal Composite Rank ────────────────

def build_massey_composite(ordinals, season=2026):
    """Average ranking across all systems for end-of-season snapshot."""
    if ordinals.empty:
        return pd.DataFrame(columns=["TeamID", "AvgOrdinalRank"])
    
    # Use the latest available day for the season
    ord_season = ordinals[ordinals["Season"] == season]
    if len(ord_season) == 0:
        # Fallback to previous season if 2026 ordinals not available
        season = 2025
        ord_season = ordinals[ordinals["Season"] == season]
    
    max_day = ord_season["RankingDayNum"].max()
    # Use rankings from roughly the last few weeks
    recent = ord_season[ord_season["RankingDayNum"] >= max_day - 7]
    composite = recent.groupby("TeamID")["OrdinalRank"].mean().reset_index()
    composite.columns = ["TeamID", "AvgOrdinalRank"]
    return composite

def calculate_upset_ratio(reg_season, massey, season=2026):
    """Calculates UpsetRatio (Win% vs teams with better AvgOrdinalRank)."""
    rs = reg_season[reg_season["Season"] == season].copy()
    rank_dict = dict(zip(massey["TeamID"], massey["AvgOrdinalRank"]))
    
    upset_wins = {}
    upset_opps = {}
    
    for _, row in rs.iterrows():
        w_id = row["WTeamID"]
        l_id = row["LTeamID"]
        
        w_rank = rank_dict.get(w_id, 175)
        l_rank = rank_dict.get(l_id, 175)
        
        # If loser has a better rank (lower number), it's an upset win for w_id
        if l_rank < w_rank:
            upset_wins[w_id] = upset_wins.get(w_id, 0) + 1
            upset_opps[w_id] = upset_opps.get(w_id, 0) + 1
            upset_opps[l_id] = upset_opps.get(l_id, 0) + 1
        elif w_rank < l_rank:
            upset_opps[w_id] = upset_opps.get(w_id, 0) + 1
            upset_opps[l_id] = upset_opps.get(l_id, 0) + 1
            
    ratios = []
    for team in set(list(upset_wins.keys()) + list(upset_opps.keys())):
        ratio = upset_wins.get(team, 0) / upset_opps.get(team, 1) if upset_opps.get(team, 0) > 0 else 0
        ratios.append({"TeamID": team, "UpsetRatio": ratio})
        
    return pd.DataFrame(ratios)


# ────────── 5. Combined Win Probability for Any Matchup ──────────

def matchup_win_prob(team_a, team_b, team_info, seed_probs, alpha=0.35, beta=0.25, gamma=0.20, delta=0.10, eps=0.10):
    """
    Blended win probability for team_a beating team_b.
    alpha = weight on seed-based prior
    beta  = weight on scoring margin (logistic)
    gamma = weight on ordinal rank
    delta = weight on strength of schedule
    eps   = weight on upset ratio penalty
    """
    info_a = team_info.get(team_a, {})
    info_b = team_info.get(team_b, {})
    
    seed_a = info_a.get("SeedNum", 8)
    seed_b = info_b.get("SeedNum", 8)
    
    # Component 1: Seed-based historical probability
    p_seed = seed_probs.get((seed_a, seed_b), 0.5)
    
    # Component 2: Scoring margin logistic
    margin_a = info_a.get("Margin", 0.0)
    margin_b = info_b.get("Margin", 0.0)
    diff_margin = margin_a - margin_b
    p_margin = 1.0 / (1.0 + np.exp(-0.15 * diff_margin))
    
    # Component 3: Ordinal rank (lower is better)
    rank_a = info_a.get("AvgOrdinalRank", 175)
    rank_b = info_b.get("AvgOrdinalRank", 175)
    diff_rank = rank_b - rank_a  # positive = a is better
    p_rank = 1.0 / (1.0 + np.exp(-0.02 * diff_rank))
    
    # Component 4: Strength of Schedule
    sos_a = info_a.get("SOS", 0.5)
    sos_b = info_b.get("SOS", 0.5)
    diff_sos = sos_a - sos_b
    p_sos = 1.0 / (1.0 + np.exp(-15.0 * diff_sos))  # Multiplier factor scales with win % difference
    
    # Component 5: Upset Ratio Penalty / Boost
    # If a team has a good upset ratio, they get a small boost against higher seeds
    ur_a = info_a.get("UpsetRatio", 0.0)
    ur_b = info_b.get("UpsetRatio", 0.0)
    diff_ur = ur_a - ur_b
    p_ur = 1.0 / (1.0 + np.exp(-5.0 * diff_ur))
    
    p = alpha * p_seed + beta * p_margin + gamma * p_rank + delta * p_sos + eps * p_ur
    return np.clip(p, 0.01, 0.99)


# ──────────────────── 6. Build the ILP ────────────────────

def build_bracket_ilp(bracket_teams, slots_df, team_info, seed_probs, teams_df):
    """
    Formulate and solve the bracket optimisation problem.
    
    bracket_teams: dict  seed_label -> TeamID  (e.g. "W01" -> 1181)
    slots_df:      DataFrame with columns [Slot, StrongSeed, WeakSeed]
    """
    # ESPN scoring per round
    ROUND_SCORES = {
        "R1": 10,   # Round of 64
        "R2": 20,   # Round of 32
        "R3": 40,   # Sweet 16
        "R4": 80,   # Elite 8
        "R5": 160,  # Final Four
        "R6": 320,  # Championship
    }
    
    # Map slot name to round
    def slot_round(slot):
        if slot.startswith("R"):
            return slot[:2]
        return None

    # Build slot tree
    # Each slot has two children (StrongSeed and WeakSeed might be other slots)
    slot_to_children = {}
    for _, row in slots_df.iterrows():
        slot_to_children[row["Slot"]] = (row["StrongSeed"], row["WeakSeed"])
    
    # Get all team IDs in the bracket
    all_teams = set(bracket_teams.values())
    team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    
    # Get all slots
    all_slots = list(slots_df["Slot"])
    
    # For each slot, determine which teams can possibly reach it
    # (recursively expand slot children)
    def get_teams_for_seed(seed_label):
        """Recursively resolve a seed label to the set of team IDs."""
        if seed_label in bracket_teams:
            return {bracket_teams[seed_label]}
        elif seed_label in slot_to_children:
            strong, weak = slot_to_children[seed_label]
            return get_teams_for_seed(strong) | get_teams_for_seed(weak)
        else:
            return set()
    
    slot_candidates = {}
    for slot in all_slots:
        strong, weak = slot_to_children[slot]
        strong_teams = get_teams_for_seed(strong)
        weak_teams   = get_teams_for_seed(weak)
        slot_candidates[slot] = (strong_teams, weak_teams)
    
    # ─── Decision variables ───
    # x[team, slot] = 1 if team wins in this slot
    x = {}
    for slot in all_slots:
        strong_t, weak_t = slot_candidates[slot]
        for team in strong_t | weak_t:
            x[(team, slot)] = LpVariable(f"x_{team}_{slot}", cat=LpBinary)
    
    # ─── Objective: maximise expected bracket score ───
    prob = LpProblem("MarchMadness_Bracket", LpMaximize)
    
    objective_terms = []
    for slot in all_slots:
        rnd = slot_round(slot)
        if rnd is None:
            continue
        score = ROUND_SCORES.get(rnd, 10)
        strong_t, weak_t = slot_candidates[slot]
        for team in strong_t | weak_t:
            # Expected value = score * P(team wins all games to reach and win this slot)
            # We approximate this by using the direct matchup probability as the weight
            # The ILP will pick the combination that maximizes total expected score
            
            # Average win probability against all possible opponents in this slot
            opponents = (weak_t if team in strong_t else strong_t)
            if opponents:
                avg_prob = np.mean([matchup_win_prob(team, opp, team_info, seed_probs) 
                                    for opp in opponents])
            else:
                avg_prob = 0.5
            
            objective_terms.append(score * avg_prob * x[(team, slot)])
    
    prob += lpSum(objective_terms)
    
    # ─── Constraints ───
    
    # C1: Exactly one winner per slot
    for slot in all_slots:
        strong_t, weak_t = slot_candidates[slot]
        all_slot_teams = strong_t | weak_t
        prob += lpSum(x[(t, slot)] for t in all_slot_teams) == 1, f"one_winner_{slot}"
    
    # C2: To win a slot, the team must have won the child slot (if not a seed)
    for slot in all_slots:
        strong_seed, weak_seed = slot_to_children[slot]
        strong_t, weak_t = slot_candidates[slot]
        
        # Teams from strong side must have won the strong child slot
        if strong_seed in slot_to_children:  # it's a slot, not a seed
            for team in strong_t:
                if (team, strong_seed) in x:
                    prob += x[(team, slot)] <= x[(team, strong_seed)], \
                        f"consistency_{team}_{slot}_from_{strong_seed}"
        
        # Teams from weak side must have won the weak child slot
        if weak_seed in slot_to_children:
            for team in weak_t:
                if (team, weak_seed) in x:
                    prob += x[(team, slot)] <= x[(team, weak_seed)], \
                        f"consistency_{team}_{slot}_from_{weak_seed}"
    
    # ─── Solve ───
    prob.solve()
    
    return prob, x, all_slots, slot_candidates, team_names, ROUND_SCORES


# ──────────────────── 7. Predict Seeds ────────────────────

def predict_seeds(feat_df, teams_df, n_teams=64):
    """Predicts seed assignments using a composite score."""
    team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    df = feat_df.copy()
    max_rank = df["AvgOrdinalRank"].max() + 1
    df["RankScore"] = (max_rank - df["AvgOrdinalRank"]) / max_rank
    margin_range = df["Margin"].max() - df["Margin"].min()
    df["MarginScore"] = (df["Margin"] - df["Margin"].min()) / margin_range if margin_range > 0 else 0.5
    # Since we don't have PPG for PuLP loaded, we will approximate without it
    df["CompositeScore"] = 0.50 * df["WinPct"] + 0.35 * df["RankScore"] + 0.15 * df["MarginScore"]
    df["TeamName"] = df["TeamID"].apply(lambda x: team_names.get(x, str(x)))
    df = df.sort_values("CompositeScore", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    top = df.head(n_teams).copy()
    regions = ["W", "X", "Y", "Z"]
    seed_assignments = []
    for i in range(n_teams):
        seed_line = (i // 4) + 1
        pos_in_line = i % 4
        region = regions[pos_in_line] if seed_line % 2 == 1 else regions[3 - pos_in_line]
        seed_assignments.append({"Seed": f"{region}{seed_line:02d}", "TeamID": int(top.iloc[i]["TeamID"]), "SeedNum": seed_line})
    return pd.DataFrame(seed_assignments)


def build_bracket_slots():
    """Generates the structure of the NCAA tournament bracket."""
    regions = ["W", "X", "Y", "Z"]
    slots = []
    for r in regions:
        for i in range(1, 9):
            slots.append({"Slot": f"R1{r}{i}", "StrongSeed": f"{r}{i:02d}", "WeakSeed": f"{r}{17-i:02d}"})
    for r in regions:
        for i in range(1, 5):
            slots.append({"Slot": f"R2{r}{i}", "StrongSeed": f"R1{r}{2*i-1}", "WeakSeed": f"R1{r}{2*i}"})
    for r in regions:
        for i in range(1, 3):
            slots.append({"Slot": f"R3{r}{i}", "StrongSeed": f"R2{r}{2*i-1}", "WeakSeed": f"R2{r}{2*i}"})
    for r in regions:
        slots.append({"Slot": f"R4{r}1", "StrongSeed": f"R3{r}1", "WeakSeed": f"R3{r}2"})
    slots.append({"Slot": "R5WX", "StrongSeed": "R4W1", "WeakSeed": "R4X1"})
    slots.append({"Slot": "R5YZ", "StrongSeed": "R4Y1", "WeakSeed": "R4Z1"})
    slots.append({"Slot": "R6CH", "StrongSeed": "R5WX", "WeakSeed": "R5YZ"})
    return pd.DataFrame(slots)


# ──────────────────── 8. Print the Bracket ────────────────────

def print_bracket(prob, x, all_slots, slot_candidates, team_names, team_info, round_scores):
    """Pretty-print the optimized bracket."""
    
    print("=" * 70)
    print("     🏀  MARCH MADNESS 2026 — OPTIMAL BRACKET (PuLP ILP)  🏀")
    print("=" * 70)
    print(f"\nSolver Status: {LpStatus[prob.status]}")
    print(f"Expected Bracket Score: {prob.objective.value():.1f}\n")
    
    # Map slot prefix to round name
    round_names = {
        "R1": "ROUND OF 64",
        "R2": "ROUND OF 32",
        "R3": "SWEET 16",
        "R4": "ELITE 8",
        "R5": "FINAL FOUR",
        "R6": "CHAMPIONSHIP",
    }
    
    region_names = {"W": "WEST", "X": "EAST", "Y": "SOUTH", "Z": "MIDWEST"}
    
    # Group results by round
    rounds = {}
    for slot in all_slots:
        rnd = slot[:2] if slot.startswith("R") else None
        if rnd:
            rounds.setdefault(rnd, []).append(slot)
    
    for rnd_key in sorted(rounds.keys()):
        rnd_name = round_names.get(rnd_key, rnd_key)
        pts = round_scores.get(rnd_key, 0)
        print(f"\n{'─' * 70}")
        print(f"  {rnd_name} ({pts} pts each)")
        print(f"{'─' * 70}")
        
        for slot in sorted(rounds[rnd_key]):
            strong_t, weak_t = slot_candidates[slot]
            all_t = strong_t | weak_t
            winner = None
            for t in all_t:
                if (t, slot) in x and x[(t, slot)].value() == 1:
                    winner = t
                    break
            if winner:
                name = team_names.get(winner, str(winner))
                info = team_info.get(winner, {})
                seed = info.get("SeedNum", "?")
                # Figure out opponent
                loser_pool = weak_t if winner in strong_t else strong_t
                region_char = slot[2] if len(slot) > 2 else ""
                region = region_names.get(region_char, "")
                
                region_str = f" [{region}]" if region else ""
                print(f"    {slot:>6s}{region_str}: ({seed:>2}) {name}")
    
    # Print champion
    champ_slot = "R6CH"
    if champ_slot in [s for s in all_slots]:
        strong_t, weak_t = slot_candidates[champ_slot]
        for t in strong_t | weak_t:
            if (t, champ_slot) in x and x[(t, champ_slot)].value() == 1:
                name = team_names.get(t, str(t))
                info = team_info.get(t, {})
                seed = info.get("SeedNum", "?")
                print(f"\n{'🏆' * 3}  CHAMPION: ({seed}) {name}  {'🏆' * 3}")
                break
    
    print(f"\n{'=' * 70}")
    # Print game count
    total_games = sum(1 for slot in all_slots if slot[:2].startswith("R"))
    print(f"Total games in bracket: {total_games}")


# ──────────────────── 8. Markdown Bracket ────────────────────

def generate_markdown_bracket(prob, x, all_slots, slot_candidates, team_names,
                               team_info, round_scores, bracket_teams,
                               slots_df, gender_label, output_path="bracket.md"):
    """Generate a beautifully formatted markdown bracket file with matchups."""
    round_labels = {
        "R1": "ROUND OF 64", "R2": "ROUND OF 32", "R3": "SWEET 16",
        "R4": "ELITE 8", "R5": "FINAL FOUR", "R6": "CHAMPIONSHIP",
    }
    
    # Build slot-to-children mapping
    slot_to_children = {}
    for _, row in slots_df.iterrows():
        slot_to_children[row["Slot"]] = (row["StrongSeed"], row["WeakSeed"])
    
    # Extract all slot winners
    slot_winner = {}
    for slot in all_slots:
        strong_t, weak_t = slot_candidates[slot]
        for t in strong_t | weak_t:
            if (t, slot) in x and x[(t, slot)].value() == 1:
                slot_winner[slot] = t
                break
    
    # Also seed winners = the teams themselves
    for seed_label, team_id in bracket_teams.items():
        slot_winner[seed_label] = team_id
    
    def resolve_team(label):
        return slot_winner.get(label)
    
    def team_str_name(team_id):
        return team_names.get(team_id, str(team_id))
    
    def team_str_seed(team_id):
        info = team_info.get(team_id, {})
        return info.get("SeedNum", "?")
    
    # Pre-calculate round slots
    rounds = {}
    for slot in all_slots:
        if slot[:2] in round_labels and slot in slot_to_children:
            rounds.setdefault(slot[:2], []).append(slot)
    
    out = []
    
    # Add gender header
    out.append(f"\n# {gender_label.upper()} TOURNAMENT PREDICTIONS\n")
    out.append(f"\n## PuLP ILP Optimizer (Expected Score: {prob.objective.value():.1f})\n\n```text\n")
    out.append(f"{'=' * 70}")
    out.append("  🏀  PuLP ILP — PREDICTED BRACKET  🏀")
    out.append(f"{'=' * 70}")
    
    for rnd in sorted(rounds.keys()):
        out.append(f"\n{'─' * 60}")
        out.append(f"  {round_labels[rnd]}")
        out.append(f"{'─' * 60}")
        
        for slot in sorted(rounds[rnd]):
            strong_label, weak_label = slot_to_children[slot]
            team_a = resolve_team(strong_label)
            team_b = resolve_team(weak_label)
            winner = slot_winner.get(slot)
            
            if team_a and team_b and winner:
                loser = team_b if winner == team_a else team_a
                w_name = team_str_name(winner)
                w_seed = team_str_seed(winner)
                l_name = team_str_name(loser)
                l_seed = team_str_seed(loser)
                
                # Format exactly as ML bracket
                out.append(f"    ({w_seed:>2}) {w_name:<20s} def. ({l_seed:>2}) {l_name:<20s}")
    
    champ_slot = "R6CH"
    if champ_slot in slot_winner:
        champ = slot_winner[champ_slot]
        champ_name = team_str_name(champ)
        champ_seed = team_str_seed(champ)
        out.append(f"\n{'🏆' * 3}  CHAMPION: ({champ_seed}) {champ_name}  {'🏆' * 3}")
        
    out.append("```")
    
    # Write file
    md_content = "\n".join(out)
    with open(output_path, "a") as f:
        f.write(md_content + "\n")
    
    print(f"\n📝 Markdown bracket written to: {output_path}")
    return output_path


# ──────────────────────── Main ────────────────────────

def run_bracket_optimizer(prefix, gender_label):
    print(f"\n{'=' * 70}")
    print(f"  🏀  RUNNING PuLP OPTIMIZER FOR: {gender_label.upper()} TOURNAMENT")
    print(f"{'=' * 70}")

    print("Loading data...")
    teams, seeds, slots, tourney, reg_season, ordinals = load_data(prefix)
    
    STRENGTH_SEASON = 2026  # Use 2026 regular-season data for team strength
    
    print(f"Using team strength data from {STRENGTH_SEASON} regular season")
    
    # 3. Build seed-based win probabilities
    print("Building seed-based win probabilities from history...")
    seed_probs = build_seed_win_rates(seeds, tourney)
    
    # 4. Build team strength from regular season
    print(f"Computing team strength from {STRENGTH_SEASON} regular season...")
    strength = build_team_strength(reg_season, STRENGTH_SEASON)
    
    # 5. Build Massey composite rankings
    print("Computing Massey Ordinal composite rankings...")
    massey = build_massey_composite(ordinals, STRENGTH_SEASON)
    
    # 6. Merge all team info for seed prediction
    team_info_df = strength.merge(massey, on="TeamID", how="left")
    team_info_df = team_info_df.fillna({"WinPct": 0.5, "Margin": 0, "AvgOrdinalRank": 175})
    
    print("\n🔮 PREDICTING TOURNAMENT SEEDS (top 64)")
    predicted_seeds = predict_seeds(team_info_df, teams, n_teams=64)
    season_slots = build_bracket_slots()
    bracket_teams = dict(zip(predicted_seeds["Seed"], predicted_seeds["TeamID"]))
    print(f"Teams in bracket: {len(bracket_teams)}")
    print(f"Bracket slots: {len(season_slots)}")
    
    # Add predicted seeds back to team info for the optimizer
    team_info_df = team_info_df.merge(predicted_seeds[["TeamID", "SeedNum"]], on="TeamID", how="left")
    team_info_df = team_info_df.fillna({"SeedNum": 16})
    
    team_info = {}
    for _, row in team_info_df.iterrows():
        team_info[row["TeamID"]] = {
            "SeedNum": int(row["SeedNum"]),
            "WinPct": row["WinPct"],
            "Margin": row["Margin"],
            "AvgOrdinalRank": row["AvgOrdinalRank"],
        }
    
    # Add info for any team not in the merge (shouldn't happen but safety)
    for tid in bracket_teams.values():
        if tid not in team_info:
            team_info[tid] = {"SeedNum": 8, "WinPct": 0.5, "Margin": 0, "AvgOrdinalRank": 175}
    
    # 7. Build and solve ILP
    print("\nFormulating ILP...")
    prob, x, all_slots, slot_candidates, team_names_dict, round_scores = \
        build_bracket_ilp(bracket_teams, season_slots, team_info, seed_probs, teams)
    
    # 8. Print results
    print_bracket(prob, x, all_slots, slot_candidates, team_names_dict, team_info, round_scores)
    
    # 9. Generate markdown bracket file
    generate_markdown_bracket(prob, x, all_slots, slot_candidates, team_names_dict,
                              team_info, round_scores, bracket_teams,
                              slots_df=season_slots, gender_label=gender_label, output_path="bracket.md")
    
    # 9. Validate bracket
    print("\n── Bracket Validation ──")
    total_games_played = 0
    for slot in all_slots:
        rnd = slot[:2]
        if rnd.startswith("R"):
            strong_t, weak_t = slot_candidates[slot]
            winners = [t for t in strong_t | weak_t 
                       if (t, slot) in x and x[(t, slot)].value() == 1]
            assert len(winners) == 1, f"Slot {slot}: expected 1 winner, got {len(winners)}"
            total_games_played += 1
    
    print(f"✅ All {total_games_played} games have exactly one winner")
    
    # Count unique teams that win at least one game
    winning_teams = set()
    for (team, slot), var in x.items():
        if var.value() == 1:
            winning_teams.add(team)
    
    # Find champion
    champ_slots = [s for s in all_slots if s.startswith("R6")]
    champs = []
    for slot in champ_slots:
        strong_t, weak_t = slot_candidates[slot]
        for t in strong_t | weak_t:
            if (t, slot) in x and x[(t, slot)].value() == 1:
                champs.append(t)
    
    assert len(champs) == 1, f"Expected 1 champion, got {len(champs)}"
    print(f"✅ Exactly one champion: {team_names_dict.get(champs[0], champs[0])}")
    print("✅ Bracket is valid!")


def main():
    if os.path.exists("bracket.md"):
        os.remove("bracket.md")
    for prefix, label in [("M", "Men's"), ("W", "Women's")]:
        run_bracket_optimizer(prefix, label)

if __name__ == "__main__":
    main()
