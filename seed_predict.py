"""
March Madness 2026 — Seed Prediction & Bracket Simulation
==========================================================
Ranks teams by composite score, assigns predicted seeds,
and simulates the tournament bracket using trained ML models.

Usage:
    python3 seed_predict.py
"""

import pandas as pd
import numpy as np
import warnings, time
from sklearn.metrics import (log_loss, accuracy_score, roc_auc_score,
                             mean_squared_error, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA = "data"

# ┌──────────────────────────────────────────────────────────────┐
# │  CONFIGURATION — Change these when real seeds are released!  │
# └──────────────────────────────────────────────────────────────┘
USE_PREDICTED_SEEDS = True    # Set to False after Selection Sunday
REAL_SEEDS_SEASON   = 2025   # Change to 2026 once seeds are in the data
STRENGTH_SEASON     = 2026   # Season to pull team strength from


# ═══════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(gender_prefix="M"):
    """Load data for the given gender prefix (M or W)."""
    teams       = pd.read_csv(f"{DATA}/{gender_prefix}Teams.csv")
    seeds       = pd.read_csv(f"{DATA}/{gender_prefix}NCAATourneySeeds.csv")
    tourney_c   = pd.read_csv(f"{DATA}/{gender_prefix}NCAATourneyCompactResults.csv")
    tourney_d   = pd.read_csv(f"{DATA}/{gender_prefix}NCAATourneyDetailedResults.csv")
    reg_compact = pd.read_csv(f"{DATA}/{gender_prefix}RegularSeasonCompactResults.csv")
    reg_detail  = pd.read_csv(f"{DATA}/{gender_prefix}RegularSeasonDetailedResults.csv")
    slots       = pd.read_csv(f"{DATA}/{gender_prefix}NCAATourneySlots.csv")
    
    # Only Men's data has Massey Ordinals
    if gender_prefix == "M":
        print("  Loading Massey Ordinals...")
        ordinals_raw = pd.read_csv(f"{DATA}/MMasseyOrdinals.csv")
        ordinals = {season: group for season, group in ordinals_raw.groupby("Season")}
        print(f"  Ordinals loaded: {len(ordinals_raw)} rows, {len(ordinals)} seasons")
    else:
        ordinals = {}
        
    return teams, seeds, tourney_c, tourney_d, reg_compact, reg_detail, ordinals, slots


# ═══════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING (shared with ml_predict.py)
# ═══════════════════════════════════════════════════════════════

def seed_number(seed_str):
    return int("".join(c for c in seed_str[1:] if c.isdigit()))


def build_season_stats(reg_detail, reg_compact, ordinals, season):
    rd = reg_detail[reg_detail["Season"] == season]
    if len(rd) > 0:
        w_stats = rd.groupby("WTeamID").agg(
            W_games=("WScore", "size"), W_pts=("WScore", "sum"), W_opp_pts=("LScore", "sum"),
            W_fgm=("WFGM", "sum"), W_fga=("WFGA", "sum"),
            W_fgm3=("WFGM3", "sum"), W_fga3=("WFGA3", "sum"),
            W_ftm=("WFTM", "sum"), W_fta=("WFTA", "sum"),
            W_or=("WOR", "sum"), W_dr=("WDR", "sum"),
            W_ast=("WAst", "sum"), W_to=("WTO", "sum"),
            W_stl=("WStl", "sum"), W_blk=("WBlk", "sum"), W_pf=("WPF", "sum"),
            W_opp_or=("LOR", "sum"), W_opp_dr=("LDR", "sum"),
        ).reset_index().rename(columns={"WTeamID": "TeamID"})

        l_stats = rd.groupby("LTeamID").agg(
            L_games=("LScore", "size"), L_pts=("LScore", "sum"), L_opp_pts=("WScore", "sum"),
            L_fgm=("LFGM", "sum"), L_fga=("LFGA", "sum"),
            L_fgm3=("LFGM3", "sum"), L_fga3=("LFGA3", "sum"),
            L_ftm=("LFTM", "sum"), L_fta=("LFTA", "sum"),
            L_or=("LOR", "sum"), L_dr=("LDR", "sum"),
            L_ast=("LAst", "sum"), L_to=("LTO", "sum"),
            L_stl=("LStl", "sum"), L_blk=("LBlk", "sum"), L_pf=("LPF", "sum"),
            L_opp_or=("WOR", "sum"), L_opp_dr=("WDR", "sum"),
        ).reset_index().rename(columns={"LTeamID": "TeamID"})

        stats = w_stats.merge(l_stats, on="TeamID", how="outer").fillna(0)
        g = stats["W_games"] + stats["L_games"]
        stats["Games"]    = g
        stats["WinPct"]   = stats["W_games"] / g
        stats["PPG"]      = (stats["W_pts"] + stats["L_pts"]) / g
        stats["OppPPG"]   = (stats["W_opp_pts"] + stats["L_opp_pts"]) / g
        stats["Margin"]   = stats["PPG"] - stats["OppPPG"]
        total_fga = stats["W_fga"] + stats["L_fga"]
        total_fga3 = stats["W_fga3"] + stats["L_fga3"]
        total_fta = stats["W_fta"] + stats["L_fta"]
        stats["FGPct"]    = (stats["W_fgm"] + stats["L_fgm"]) / total_fga.replace(0, 1)
        stats["FG3Pct"]   = (stats["W_fgm3"] + stats["L_fgm3"]) / total_fga3.replace(0, 1)
        stats["FTPct"]    = (stats["W_ftm"] + stats["L_ftm"]) / total_fta.replace(0, 1)
        stats["ORpg"]     = (stats["W_or"] + stats["L_or"]) / g
        stats["DRpg"]     = (stats["W_dr"] + stats["L_dr"]) / g
        stats["RebMargin"]= stats["ORpg"] + stats["DRpg"] - \
                            (stats["W_opp_or"] + stats["L_opp_or"] + stats["W_opp_dr"] + stats["L_opp_dr"]) / g
        stats["AstTO"]    = (stats["W_ast"] + stats["L_ast"]) / (stats["W_to"] + stats["L_to"]).replace(0, 1)
        stats["StlPG"]    = (stats["W_stl"] + stats["L_stl"]) / g
        stats["BlkPG"]    = (stats["W_blk"] + stats["L_blk"]) / g
        stats["PFPG"]     = (stats["W_pf"] + stats["L_pf"]) / g
        cols = ["TeamID", "Games", "WinPct", "PPG", "OppPPG", "Margin",
                "FGPct", "FG3Pct", "FTPct", "ORpg", "DRpg", "RebMargin",
                "AstTO", "StlPG", "BlkPG", "PFPG"]
        return stats[cols]
    else:
        rc = reg_compact[reg_compact["Season"] == season]
        w = rc.groupby("WTeamID").agg(W_games=("WScore","size"), W_pts=("WScore","sum"), W_opp=("LScore","sum")).reset_index().rename(columns={"WTeamID":"TeamID"})
        l = rc.groupby("LTeamID").agg(L_games=("LScore","size"), L_pts=("LScore","sum"), L_opp=("WScore","sum")).reset_index().rename(columns={"LTeamID":"TeamID"})
        stats = w.merge(l, on="TeamID", how="outer").fillna(0)
        g = stats["W_games"] + stats["L_games"]
        stats["Games"] = g; stats["WinPct"] = stats["W_games"] / g
        stats["PPG"] = (stats["W_pts"] + stats["L_pts"]) / g
        stats["OppPPG"] = (stats["W_opp"] + stats["L_opp"]) / g
        stats["Margin"] = stats["PPG"] - stats["OppPPG"]
        for col in ["FGPct","FG3Pct","FTPct","ORpg","DRpg","RebMargin","AstTO","StlPG","BlkPG","PFPG"]:
            stats[col] = 0.0
        cols = ["TeamID","Games","WinPct","PPG","OppPPG","Margin","FGPct","FG3Pct","FTPct","ORpg","DRpg","RebMargin","AstTO","StlPG","BlkPG","PFPG"]
        return stats[cols]


def build_massey_composite(ordinals, season):
    o = ordinals.get(season)
    if o is None or len(o) == 0:
        return pd.DataFrame(columns=["TeamID", "AvgRank"])
    max_day = o["RankingDayNum"].max()
    recent = o[o["RankingDayNum"] >= max_day - 7]
    comp = recent.groupby("TeamID")["OrdinalRank"].mean().reset_index()
    comp.columns = ["TeamID", "AvgRank"]
    return comp


def build_features_for_season(season, reg_detail, reg_compact, ordinals, seeds):
    stats = build_season_stats(reg_detail, reg_compact, ordinals, season)
    massey = build_massey_composite(ordinals, season)
    stats = stats.merge(massey, on="TeamID", how="left")
    stats["AvgRank"] = stats["AvgRank"].fillna(200)
    s = seeds[seeds["Season"] == season][["TeamID", "Seed"]].copy()
    s["SeedNum"] = s["Seed"].apply(seed_number)
    stats = stats.merge(s[["TeamID", "SeedNum"]], on="TeamID", how="left")
    stats["SeedNum"] = stats["SeedNum"].fillna(16)
    stats["Season"] = season
    return stats


FEATURE_COLS = ["WinPct", "PPG", "OppPPG", "Margin", "FGPct", "FG3Pct",
                "FTPct", "ORpg", "DRpg", "RebMargin", "AstTO", "StlPG",
                "BlkPG", "PFPG", "AvgRank", "SeedNum"]


def get_feature_names():
    feats = []
    for col in FEATURE_COLS:
        feats.extend([f"A_{col}", f"B_{col}", f"D_{col}"])
    return feats


def build_matchup_dataset(tourney_c, reg_detail, reg_compact, ordinals, seeds,
                          min_season=2003, max_season=2025):
    print(f"Building matchup features for seasons {min_season}-{max_season}...")
    all_features = {}
    for season in range(min_season, max_season + 1):
        all_features[season] = build_features_for_season(season, reg_detail, reg_compact, ordinals, seeds)
    rows = []
    games = tourney_c[(tourney_c["Season"] >= min_season) & (tourney_c["Season"] <= max_season)]
    for _, game in games.iterrows():
        season = game["Season"]
        feat = all_features[season]
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        if w_id < l_id:
            team_a, team_b, label = w_id, l_id, 1
        else:
            team_a, team_b, label = l_id, w_id, 0
        a_stats = feat[feat["TeamID"] == team_a]
        b_stats = feat[feat["TeamID"] == team_b]
        if len(a_stats) == 0 or len(b_stats) == 0:
            continue
        a_row, b_row = a_stats.iloc[0], b_stats.iloc[0]
        row = {"Season": season, "TeamA": team_a, "TeamB": team_b, "Label": label}
        for col in FEATURE_COLS:
            row[f"A_{col}"] = a_row[col]; row[f"B_{col}"] = b_row[col]; row[f"D_{col}"] = a_row[col] - b_row[col]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"  Total matchup samples: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
#  3. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, verbosity=0, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)])
    return model

def train_catboost(X_train, y_train, X_val, y_val):
    model = cb.CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=50,
        eval_metric="Logloss", early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model




# ═══════════════════════════════════════════════════════════════
#  4. SEED PREDICTION
# ═══════════════════════════════════════════════════════════════

def predict_seeds(feat_df, teams_df, n_teams=64):
    team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    df = feat_df.copy()
    max_rank = df["AvgRank"].max() + 1
    df["RankScore"] = (max_rank - df["AvgRank"]) / max_rank
    margin_range = df["Margin"].max() - df["Margin"].min()
    df["MarginScore"] = (df["Margin"] - df["Margin"].min()) / margin_range if margin_range > 0 else 0.5
    ppg_range = df["PPG"].max() - df["PPG"].min()
    df["PPGScore"] = (df["PPG"] - df["PPG"].min()) / ppg_range if ppg_range > 0 else 0.5
    df["CompositeScore"] = 0.35 * df["WinPct"] + 0.30 * df["RankScore"] + 0.25 * df["MarginScore"] + 0.10 * df["PPGScore"]
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
    return pd.DataFrame(seed_assignments), df


def build_bracket_slots():
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


# ═══════════════════════════════════════════════════════════════
#  5. BRACKET SIMULATION
# ═══════════════════════════════════════════════════════════════

def predict_matchup_proba(model, model_name, feat_a, feat_b):
    x = np.array([val for a, b in zip(feat_a, feat_b) for val in (a, b, a - b)]).reshape(1, -1)
    if model_name == "CatBoost":
        raw = model.predict(x, prediction_type="Probability")
        return float(raw[0][1]) if raw.ndim > 1 else float(raw[0])
    else:
        return model.predict_proba(x)[0, 1]


def simulate_bracket(model, model_name, team_features, predicted_seeds, slots_df, teams_df):
    team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    seed_to_team = dict(zip(predicted_seeds["Seed"], predicted_seeds["TeamID"]))
    team_to_seed = dict(zip(predicted_seeds["TeamID"], predicted_seeds["Seed"]))
    slot_winners = {}
    for seed_label, team_id in seed_to_team.items():
        slot_winners[seed_label] = team_id
    results_by_round = {}
    prev_rnd = None
    for _, row in slots_df.sort_values("Slot").iterrows():
        slot, strong, weak = row["Slot"], row["StrongSeed"], row["WeakSeed"]
        rnd = slot[:2]
        if rnd != prev_rnd:
            print(f"    Simulating {rnd}...", flush=True); prev_rnd = rnd
        team_a_id, team_b_id = slot_winners.get(strong), slot_winners.get(weak)
        if team_a_id is None or team_b_id is None: continue
        a_feat, b_feat = team_features.get(team_a_id), team_features.get(team_b_id)
        if a_feat is None or b_feat is None:
            slot_winners[slot] = team_a_id; continue
        if team_a_id < team_b_id:
            p = predict_matchup_proba(model, model_name, a_feat, b_feat)
            winner = team_a_id if p > 0.5 else team_b_id
            loser = team_b_id if p > 0.5 else team_a_id
            win_prob = p if winner == team_a_id else 1 - p
        else:
            p = predict_matchup_proba(model, model_name, b_feat, a_feat)
            winner = team_b_id if p > 0.5 else team_a_id
            loser = team_a_id if p > 0.5 else team_b_id
            win_prob = p if winner == team_b_id else 1 - p
        slot_winners[slot] = winner
        results_by_round.setdefault(rnd, []).append({
            "slot": slot, "winner": winner, "winner_name": team_names.get(winner, str(winner)),
            "winner_seed": team_to_seed.get(winner, "??"), "loser": loser,
            "loser_name": team_names.get(loser, str(loser)), "loser_seed": team_to_seed.get(loser, "??"),
            "prob": win_prob,
        })
    return results_by_round, slot_winners


def print_bracket_results(results_by_round, model_name, file=None):
    round_names = {"R1": "ROUND OF 64", "R2": "ROUND OF 32", "R3": "SWEET 16",
                   "R4": "ELITE 8", "R5": "FINAL FOUR", "R6": "CHAMPIONSHIP"}
    out = []
    out.append(f"\n{'=' * 70}")
    out.append(f"  🏀  {model_name} — PREDICTED BRACKET  🏀")
    out.append(f"{'=' * 70}")
    for rnd in sorted(results_by_round.keys()):
        out.append(f"\n{'─' * 60}")
        out.append(f"  {round_names.get(rnd, rnd)}")
        out.append(f"{'─' * 60}")
        for game in results_by_round[rnd]:
            sw = seed_number(game["winner_seed"]) if isinstance(game["winner_seed"], str) and len(game["winner_seed"]) > 1 else "?"
            sl = seed_number(game["loser_seed"]) if isinstance(game["loser_seed"], str) and len(game["loser_seed"]) > 1 else "?"
            out.append(f"    ({sw:>2}) {game['winner_name']:<20s} def. ({sl:>2}) {game['loser_name']:<20s} [{game['prob']:.1%}]")
    if "R6" in results_by_round:
        champ = results_by_round["R6"][0]
        sc = seed_number(champ["winner_seed"]) if isinstance(champ["winner_seed"], str) and len(champ["winner_seed"]) > 1 else "?"
        out.append(f"\n{'🏆' * 3}  CHAMPION: ({sc}) {champ['winner_name']}  {'🏆' * 3}")
        
    res_str = "\n".join(out)
    print(res_str)
    if file is not None:
        file.write(res_str + "\n")


# ═══════════════════════════════════════════════════════════════
#  6. EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════

def eval_model(y_true, y_proba):
    y_pred = (y_proba > 0.5).astype(int)
    mse = mean_squared_error(y_true, y_proba)
    return {
        "mse": mse, "rmse": np.sqrt(mse), "logloss": log_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


# ═══════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════

def run_tournament_prediction(gender_prefix, gender_label):
    print(f"\n{'=' * 90}")
    print(f"  🏀  RUNNING PIPELINE FOR: {gender_label.upper()} TOURNAMENT")
    print(f"{'=' * 90}")
    
    print("Loading data...")
    teams, seeds, tourney_c, tourney_d, reg_compact, reg_detail, ordinals, slots = load_data(gender_prefix)

    # Women's detailed data starts from 2010
    min_season = 2003 if gender_prefix == "M" else 2010
    TRAIN_END, VAL_SEASON = 2024, 2025
    
    matchups = build_matchup_dataset(tourney_c, reg_detail, reg_compact, ordinals, seeds,
                                     min_season=min_season, max_season=VAL_SEASON)
    feature_names = get_feature_names()
    train_df = matchups[matchups["Season"] <= TRAIN_END]
    val_df   = matchups[matchups["Season"] == VAL_SEASON]
    X_train, y_train = train_df[feature_names].values, train_df["Label"].values
    X_val, y_val     = val_df[feature_names].values, val_df["Label"].values
    print(f"\nTraining: {len(X_train)} samples  |  Validation (2025): {len(X_val)} samples")

    results = {}
    for name, train_fn in [("XGBoost", train_xgboost), ("LightGBM", train_lightgbm),
                            ("CatBoost", train_catboost)]:
        print(f"\n── Training {name} ──")
        model = train_fn(X_train, y_train, X_val, y_val)
        pred = model.predict_proba(X_val)[:, 1] if name != "CatBoost" else model.predict_proba(X_val)[:, 1]
        results[name] = {"model": model, "proba": pred, **eval_model(y_val, pred)}
        print(f"   MSE: {results[name]['mse']:.4f}  |  Acc: {results[name]['accuracy']:.2%}  |  F1: {results[name]['f1']:.4f}")

    # ─── Model comparison ───
    print("\n" + "=" * 90)
    print("  MODEL COMPARISON — 2025 Holdout Validation")
    print(f"  Split: Train {min_season}-2024 | Test 2025 tournament")
    print("=" * 90)
    header = f"  {'Model':<10s} {'MSE':>7s} {'RMSE':>7s} {'LogLoss':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s}"
    print(header); print(f"  {'─' * 68}")
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        r = results[name]
        print(f"  {name:<10s} {r['mse']:>7.4f} {r['rmse']:>7.4f} {r['logloss']:>8.4f} "
              f"{r['accuracy']:>6.1%} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}")
    best_name = min(results, key=lambda k: results[k]["mse"])
    print(f"\n  🏅 Best model (by MSE): {best_name}")

    # ─── Build features ───
    print(f"\n{'=' * 70}")
    print(f"  Building {STRENGTH_SEASON} team features...")
    feat_2026 = build_features_for_season(STRENGTH_SEASON, reg_detail, reg_compact, ordinals, seeds)
    print(f"  Features for {len(feat_2026)} teams")
    team_feat_dict = {}
    for _, row in feat_2026.iterrows():
        team_feat_dict[row["TeamID"]] = np.array([row[c] for c in FEATURE_COLS])

    # ─── Seeds ───
    if USE_PREDICTED_SEEDS:
        print(f"\n{'=' * 70}")
        print("  🔮 PREDICTING TOURNAMENT SEEDS (top 64)")
        print(f"{'=' * 70}")
        predicted_seeds, full_rankings = predict_seeds(feat_2026, teams, n_teams=64)
        bracket_slots = build_bracket_slots()
        regions = {"W": "REGION 1", "X": "REGION 2", "Y": "REGION 3", "Z": "REGION 4"}
        for rc, rn in regions.items():
            rs = predicted_seeds[predicted_seeds["Seed"].str.startswith(rc)].sort_values("SeedNum")
            print(f"\n  {rn}:")
            for _, row in rs.iterrows():
                tid = row["TeamID"]
                nm = teams[teams["TeamID"]==tid].iloc[0]["TeamName"] if len(teams[teams["TeamID"]==tid]) > 0 else str(tid)
                f = feat_2026[feat_2026["TeamID"]==tid]
                if len(f) > 0:
                    f = f.iloc[0]
                    print(f"    ({row['SeedNum']:>2d}) {nm:<22s}  Win%: {f['WinPct']:.1%}  Margin: {f['Margin']:+.1f}  Rank: {f['AvgRank']:.0f}")
        bubble = full_rankings.iloc[64:74]
        print(f"\n  {'─' * 60}\n  BUBBLE (first 10 out):")
        for _, row in bubble.iterrows():
            print(f"    #{int(row['Rank']):>3d}  {row['TeamName']:<22s}  Win%: {row['WinPct']:.1%}  Margin: {row['Margin']:+.1f}  Score: {row['CompositeScore']:.3f}")
        use_seeds, use_slots = predicted_seeds, bracket_slots
    else:
        print(f"\n  Using REAL seeds from season {REAL_SEEDS_SEASON}")
        s = seeds[seeds["Season"] == REAL_SEEDS_SEASON].copy()
        s["SeedNum"] = s["Seed"].apply(seed_number)
        use_seeds = s[["Seed", "TeamID", "SeedNum"]]
        use_slots = slots[slots["Season"] == REAL_SEEDS_SEASON][["Slot", "StrongSeed", "WeakSeed"]]

    # ─── Simulate ───
    print(f"\n{'=' * 70}")
    print(f"  SIMULATING {gender_label.upper()} BRACKET — Seeds: {'PREDICTED' if USE_PREDICTED_SEEDS else 'REAL'}")
    print(f"{'=' * 70}")
    
    # Sort models by MSE
    sorted_models = sorted(results.keys(), key=lambda k: results[k]["mse"])
    for idx, model_name in enumerate(sorted_models):
        print(f"\n  ▶ Simulating with {model_name}...", flush=True)
        t0 = time.time()
        model = results[model_name]["model"]
        br, sw = simulate_bracket(model, model_name, team_feat_dict, use_seeds, use_slots, teams)
        print(f"    ✓ Done in {time.time()-t0:.1f}s", flush=True)
        
        with open("ml_predictions.md", "a") as f:
            if idx == 0:
                f.write(f"\n# {gender_label.upper()} TOURNAMENT PREDICTIONS\n")
            f.write(f"\n## #{idx+1} Model: {model_name} (MSE: {results[model_name]['mse']:.4f})\n\n```text\n")
            print_bracket_results(br, model_name, file=f)
            f.write("```\n")
        
        print_bracket_results(br, model_name)

    print(f"\n{'=' * 70}")
    print(f"  Done! All {gender_label} brackets simulated.")
    if USE_PREDICTED_SEEDS:
        print("  💡 Set USE_PREDICTED_SEEDS = False after Selection Sunday")
    print(f"{'=' * 70}")


def main():
    import os
    if os.path.exists("ml_predictions.md"):
        os.remove("ml_predictions.md")
    for prefix, label in [("M", "Men's"), ("W", "Women's")]:
        run_tournament_prediction(prefix, label)


if __name__ == "__main__":
    main()
