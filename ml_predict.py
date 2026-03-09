"""
March Madness 2026 — Kaggle Submission Generator
==================================================
Trains ML models on historical NCAA tournament data for BOTH men's and
women's teams, then generates a Kaggle-format submission CSV with
predictions for every possible matchup.

Models: XGBoost, LightGBM, CatBoost
Best model (by MSE) is selected to generate the submission.

Output: submission.csv (matches SampleSubmissionStage2.csv format)
"""

import pandas as pd
import numpy as np
import warnings, os, time
from sklearn.metrics import (log_loss, accuracy_score, roc_auc_score,
                             mean_squared_error, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA = "data"


# ═══════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_gender_data(prefix):
    """Load all data for one gender. prefix = 'M' or 'W'."""
    teams       = pd.read_csv(f"{DATA}/{prefix}Teams.csv")
    seeds       = pd.read_csv(f"{DATA}/{prefix}NCAATourneySeeds.csv")
    tourney_c   = pd.read_csv(f"{DATA}/{prefix}NCAATourneyCompactResults.csv")
    reg_compact = pd.read_csv(f"{DATA}/{prefix}RegularSeasonCompactResults.csv")
    reg_detail  = pd.read_csv(f"{DATA}/{prefix}RegularSeasonDetailedResults.csv")

    # Massey Ordinals (only available for men)
    ordinals_path = f"{DATA}/{prefix}MasseyOrdinals.csv"
    if os.path.exists(ordinals_path):
        print(f"  Loading {prefix} Massey Ordinals...")
        ordinals_raw = pd.read_csv(ordinals_path)
        ordinals = {season: group for season, group in ordinals_raw.groupby("Season")}
        print(f"  {prefix} Ordinals: {len(ordinals_raw)} rows, {len(ordinals)} seasons")
    else:
        print(f"  ⚠️  No Massey Ordinals for {prefix} — using default rank")
        ordinals = {}

    return {
        "teams": teams, "seeds": seeds, "tourney_c": tourney_c,
        "reg_compact": reg_compact, "reg_detail": reg_detail,
        "ordinals": ordinals, "prefix": prefix,
    }


# ═══════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

FEATURE_COLS = ["WinPct", "PPG", "OppPPG", "Margin", "FGPct", "FG3Pct",
                "FTPct", "ORpg", "DRpg", "RebMargin", "AstTO", "StlPG",
                "BlkPG", "PFPG", "AvgRank", "SeedNum",
                "UpsetWinPct", "UpsetLossPct", "CloseWinPct", "CloseLossPct",
                "SoS", "WinsVsTop50", "WinsVsTop25"]


def seed_number(seed_str):
    return int("".join(c for c in seed_str[1:] if c.isdigit()))


def build_season_stats(reg_detail, reg_compact, data, season):
    rd = reg_detail[reg_detail["Season"] == season]
    if len(rd) > 0:
        w = rd.groupby("WTeamID").agg(
            W_g=("WScore","size"), W_pts=("WScore","sum"), W_opp=("LScore","sum"),
            W_fgm=("WFGM","sum"), W_fga=("WFGA","sum"),
            W_fgm3=("WFGM3","sum"), W_fga3=("WFGA3","sum"),
            W_ftm=("WFTM","sum"), W_fta=("WFTA","sum"),
            W_or=("WOR","sum"), W_dr=("WDR","sum"),
            W_ast=("WAst","sum"), W_to=("WTO","sum"),
            W_stl=("WStl","sum"), W_blk=("WBlk","sum"), W_pf=("WPF","sum"),
            W_opp_or=("LOR","sum"), W_opp_dr=("LDR","sum"),
        ).reset_index().rename(columns={"WTeamID": "TeamID"})
        l = rd.groupby("LTeamID").agg(
            L_g=("LScore","size"), L_pts=("LScore","sum"), L_opp=("WScore","sum"),
            L_fgm=("LFGM","sum"), L_fga=("LFGA","sum"),
            L_fgm3=("LFGM3","sum"), L_fga3=("LFGA3","sum"),
            L_ftm=("LFTM","sum"), L_fta=("LFTA","sum"),
            L_or=("LOR","sum"), L_dr=("LDR","sum"),
            L_ast=("LAst","sum"), L_to=("LTO","sum"),
            L_stl=("LStl","sum"), L_blk=("LBlk","sum"), L_pf=("LPF","sum"),
            L_opp_or=("WOR","sum"), L_opp_dr=("WDR","sum"),
        ).reset_index().rename(columns={"LTeamID": "TeamID"})
        s = w.merge(l, on="TeamID", how="outer").fillna(0)
        g = s["W_g"] + s["L_g"]
        s["Games"] = g
        s["WinPct"] = s["W_g"] / g
        s["PPG"] = (s["W_pts"] + s["L_pts"]) / g
        s["OppPPG"] = (s["W_opp"] + s["L_opp"]) / g
        s["Margin"] = s["PPG"] - s["OppPPG"]
        fga = s["W_fga"] + s["L_fga"]
        fga3 = s["W_fga3"] + s["L_fga3"]
        fta = s["W_fta"] + s["L_fta"]
        s["FGPct"] = (s["W_fgm"] + s["L_fgm"]) / fga.replace(0, 1)
        s["FG3Pct"] = (s["W_fgm3"] + s["L_fgm3"]) / fga3.replace(0, 1)
        s["FTPct"] = (s["W_ftm"] + s["L_ftm"]) / fta.replace(0, 1)
        s["ORpg"] = (s["W_or"] + s["L_or"]) / g
        s["DRpg"] = (s["W_dr"] + s["L_dr"]) / g
        s["RebMargin"] = s["ORpg"] + s["DRpg"] - (s["W_opp_or"]+s["L_opp_or"]+s["W_opp_dr"]+s["L_opp_dr"]) / g
        s["AstTO"] = (s["W_ast"]+s["L_ast"]) / (s["W_to"]+s["L_to"]).replace(0, 1)
        s["StlPG"] = (s["W_stl"]+s["L_stl"]) / g
        s["BlkPG"] = (s["W_blk"]+s["L_blk"]) / g
        s["PFPG"] = (s["W_pf"]+s["L_pf"]) / g
    else:
        rc = reg_compact[reg_compact["Season"] == season]
        w = rc.groupby("WTeamID").agg(W_g=("WScore","size"),W_pts=("WScore","sum"),W_opp=("LScore","sum")).reset_index().rename(columns={"WTeamID":"TeamID"})
        l = rc.groupby("LTeamID").agg(L_g=("LScore","size"),L_pts=("LScore","sum"),L_opp=("WScore","sum")).reset_index().rename(columns={"LTeamID":"TeamID"})
        s = w.merge(l, on="TeamID", how="outer").fillna(0)
        g = s["W_g"] + s["L_g"]
        s["Games"] = g; s["WinPct"] = s["W_g"] / g
        s["PPG"] = (s["W_pts"]+s["L_pts"]) / g
        s["OppPPG"] = (s["W_opp"]+s["L_opp"]) / g
        s["Margin"] = s["PPG"] - s["OppPPG"]
        for col in ["FGPct","FG3Pct","FTPct","ORpg","DRpg","RebMargin","AstTO","StlPG","BlkPG","PFPG"]:
            s[col] = 0.0
    rc = reg_compact[reg_compact["Season"] == season]
    massey = build_massey_composite(data["ordinals"], season)
    rank_dict = dict(zip(massey["TeamID"], massey["AvgRank"]))

    upset_wins, upset_losses = {}, {}
    close_wins, close_losses = {}, {}
    total_wins, total_losses = {}, {}
    total_games = {}
    wins_vs_top50, wins_vs_top25 = {}, {}
    opp_ranks, opp_count = {}, {}

    for _, game in rc.iterrows():
        w, l = game["WTeamID"], game["LTeamID"]
        w_rank = rank_dict.get(w, 200)
        l_rank = rank_dict.get(l, 200)
        margin = game["WScore"] - game["LScore"]

        for tid in [w, l]:
            total_games.setdefault(tid, 0)
            opp_ranks.setdefault(tid, 0.0)
            opp_count.setdefault(tid, 0)

        total_wins.setdefault(w, 0); upset_wins.setdefault(w, 0); close_wins.setdefault(w, 0)
        wins_vs_top50.setdefault(w, 0); wins_vs_top25.setdefault(w, 0)
        total_losses.setdefault(l, 0); upset_losses.setdefault(l, 0); close_losses.setdefault(l, 0)

        total_games[w] += 1; total_games[l] += 1
        total_wins[w] += 1; total_losses[l] += 1

        opp_ranks[w] += l_rank; opp_count[w] += 1
        opp_ranks[l] += w_rank; opp_count[l] += 1

        if w_rank > l_rank: upset_wins[w] += 1
        if l_rank < w_rank: upset_losses[l] += 1
        if margin <= 5:
            close_wins[w] += 1
            close_losses[l] += 1
        if l_rank <= 50: wins_vs_top50[w] += 1
        if l_rank <= 25: wins_vs_top25[w] += 1

    s["UpsetWinPct"] = s["TeamID"].apply(lambda t: upset_wins.get(t, 0) / max(total_wins.get(t, 0), 1))
    s["UpsetLossPct"] = s["TeamID"].apply(lambda t: upset_losses.get(t, 0) / max(total_losses.get(t, 0), 1))
    s["CloseWinPct"] = s["TeamID"].apply(lambda t: close_wins.get(t, 0) / max(total_games.get(t, 0), 1))
    s["CloseLossPct"] = s["TeamID"].apply(lambda t: close_losses.get(t, 0) / max(total_games.get(t, 0), 1))
    s["SoS"] = s["TeamID"].apply(lambda t: opp_ranks.get(t, 200) / max(opp_count.get(t, 0), 1))
    s["WinsVsTop50"] = s["TeamID"].apply(lambda t: wins_vs_top50.get(t, 0))
    s["WinsVsTop25"] = s["TeamID"].apply(lambda t: wins_vs_top25.get(t, 0))

    cols = ["TeamID", "Games", "WinPct", "PPG", "OppPPG", "Margin",
            "FGPct", "FG3Pct", "FTPct", "ORpg", "DRpg", "RebMargin",
            "AstTO", "StlPG", "BlkPG", "PFPG",
            "UpsetWinPct", "UpsetLossPct", "CloseWinPct", "CloseLossPct",
            "SoS", "WinsVsTop50", "WinsVsTop25"]
    return s[cols]


def build_massey_composite(ordinals, season):
    o = ordinals.get(season)
    if o is None or len(o) == 0:
        return pd.DataFrame(columns=["TeamID", "AvgRank"])
    max_day = o["RankingDayNum"].max()
    recent = o[o["RankingDayNum"] >= max_day - 7]
    comp = recent.groupby("TeamID")["OrdinalRank"].mean().reset_index()
    comp.columns = ["TeamID", "AvgRank"]
    return comp


def build_features_for_season(season, data):
    """Build per-team feature vector for a season using gender-specific data."""
    stats = build_season_stats(data["reg_detail"], data["reg_compact"], data, season)
    massey = build_massey_composite(data["ordinals"], season)
    stats = stats.merge(massey, on="TeamID", how="left")
    stats["AvgRank"] = stats["AvgRank"].fillna(200)
    s = data["seeds"][data["seeds"]["Season"] == season][["TeamID", "Seed"]].copy()
    if len(s) > 0:
        s["SeedNum"] = s["Seed"].apply(seed_number)
        stats = stats.merge(s[["TeamID", "SeedNum"]], on="TeamID", how="left")
    else:
        stats["SeedNum"] = np.nan
    stats["SeedNum"] = stats["SeedNum"].fillna(16)
    stats["Season"] = season
    return stats


# ═══════════════════════════════════════════════════════════════
#  3. MATCHUP DATASET
# ═══════════════════════════════════════════════════════════════

def get_feature_names():
    feats = []
    for col in FEATURE_COLS:
        feats.extend([f"A_{col}", f"B_{col}", f"D_{col}"])
    return feats


def build_matchup_dataset(data, min_season, max_season):
    """Build matchup training data from historical tournament games."""
    print(f"  Building {data['prefix']} matchups ({min_season}-{max_season})...")
    all_features = {}
    for season in range(min_season, max_season + 1):
        all_features[season] = build_features_for_season(season, data)
    
    rows = []
    games = data["tourney_c"]
    games = games[(games["Season"] >= min_season) & (games["Season"] <= max_season)]
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
            row[f"A_{col}"] = a_row[col]
            row[f"B_{col}"] = b_row[col]
            row[f"D_{col}"] = a_row[col] - b_row[col]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"    {len(df)} matchup samples")
    return df


# ═══════════════════════════════════════════════════════════════
#  4. MODEL TRAINING
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
#  5. EVALUATION
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


def train_all_models(X_train, y_train, X_val, y_val, gender_label):
    """Train all 3 models, return results dict keyed by model name."""
    results = {}
    train_fns = [
        ("XGBoost", train_xgboost),
        ("LightGBM", train_lightgbm),
        ("CatBoost", train_catboost),
    ]
    for name, fn in train_fns:
        print(f"\n  ── {gender_label} {name} ──")
        model = fn(X_train, y_train, X_val, y_val)
        pred = model.predict_proba(X_val)[:, 1]
        results[name] = {"model": model, "proba": pred, **eval_model(y_val, pred)}
        r = results[name]
        print(f"     MSE: {r['mse']:.4f} | Acc: {r['accuracy']:.2%} | F1: {r['f1']:.4f}")
    return results


def print_comparison_table(results, label):
    print(f"\n{'=' * 90}")
    print(f"  {label} — MODEL COMPARISON")
    print(f"{'=' * 90}")
    header = f"  {'Model':<10s} {'MSE':>7s} {'RMSE':>7s} {'LogLoss':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s}"
    print(header)
    print(f"  {'─' * 68}")
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        r = results[name]
        print(f"  {name:<10s} {r['mse']:>7.4f} {r['rmse']:>7.4f} {r['logloss']:>8.4f} "
              f"{r['accuracy']:>6.1%} {r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} {r['auc']:>7.4f}")
    best = min(results, key=lambda k: results[k]["mse"])
    print(f"\n  🏅 Best (MSE): {best}")
    return best


# ═══════════════════════════════════════════════════════════════
#  6. SUBMISSION GENERATION
# ═══════════════════════════════════════════════════════════════

def predict_single(model, model_name, feat_a, feat_b):
    """Predict P(team_a wins) for a single matchup."""
    x = np.array([val for a, b in zip(feat_a, feat_b) for val in (a, b, a - b)]).reshape(1, -1)
    if model_name == "CatBoost":
        raw = model.predict(x, prediction_type="Probability")
        return float(raw[0][1]) if raw.ndim > 1 else float(raw[0])
    else:
        return model.predict_proba(x)[0, 1]


def predict_batch(model, model_name, X_batch):
    """Predict probabilities for a batch of matchups. Much faster than per-row."""
    if model_name == "CatBoost":
        raw = model.predict(X_batch, prediction_type="Probability")
        return raw[:, 1] if raw.ndim > 1 else raw
    else:
        return model.predict_proba(X_batch)[:, 1]


def generate_submission(m_model, m_model_name, m_feat_dict,
                        w_model, w_model_name, w_feat_dict,
                        submission_template_path, output_path="submission.csv"):
    """Generate Kaggle submission CSV for all matchups."""
    print(f"\n{'=' * 70}")
    print(f"  GENERATING KAGGLE SUBMISSION")
    print(f"  Men's model:   {m_model_name}")
    print(f"  Women's model: {w_model_name}")
    print(f"{'=' * 70}")

    template = pd.read_csv(submission_template_path)
    print(f"  Template: {len(template)} matchups")

    # Parse IDs
    ids = template["ID"].str.split("_", expand=True)
    ids.columns = ["Season", "TeamA", "TeamB"]
    ids["TeamA"] = ids["TeamA"].astype(int)
    ids["TeamB"] = ids["TeamB"].astype(int)

    # Classify as men or women
    is_men = (ids["TeamA"] >= 1101) & (ids["TeamA"] <= 1999)
    n_men = is_men.sum()
    n_women = (~is_men).sum()
    print(f"  Men's matchups: {n_men:,}  |  Women's matchups: {n_women:,}")

    predictions = np.full(len(template), 0.5)  # default

    # ─── Batch predict men ───
    print("  Predicting men's matchups...", flush=True)
    t0 = time.time()
    men_idx = np.where(is_men)[0]
    men_rows = []
    for idx in men_idx:
        ta, tb = ids.iloc[idx]["TeamA"], ids.iloc[idx]["TeamB"]
        fa = m_feat_dict.get(ta)
        fb = m_feat_dict.get(tb)
        if fa is not None and fb is not None:
            men_rows.append((idx, np.array([val for a, b in zip(fa, fb) for val in (a, b, a - b)])))
        # else: stays at 0.5

    if men_rows:
        men_indices, men_X = zip(*men_rows)
        men_X = np.array(men_X)
        men_preds = predict_batch(m_model, m_model_name, men_X)
        men_preds = np.clip(men_preds, 0.01, 0.99)
        for i, idx in enumerate(men_indices):
            predictions[idx] = men_preds[i]
    print(f"    ✓ {len(men_rows):,} men's predictions in {time.time()-t0:.1f}s")

    # ─── Batch predict women ───
    print("  Predicting women's matchups...", flush=True)
    t0 = time.time()
    women_idx = np.where(~is_men)[0]
    women_rows = []
    for idx in women_idx:
        ta, tb = ids.iloc[idx]["TeamA"], ids.iloc[idx]["TeamB"]
        fa = w_feat_dict.get(ta)
        fb = w_feat_dict.get(tb)
        if fa is not None and fb is not None:
            women_rows.append((idx, np.array([val for a, b in zip(fa, fb) for val in (a, b, a - b)])))

    if women_rows:
        women_indices, women_X = zip(*women_rows)
        women_X = np.array(women_X)
        women_preds = predict_batch(w_model, w_model_name, women_X)
        women_preds = np.clip(women_preds, 0.01, 0.99)
        for i, idx in enumerate(women_indices):
            predictions[idx] = women_preds[i]
    print(f"    ✓ {len(women_rows):,} women's predictions in {time.time()-t0:.1f}s")

    # ─── Write submission ───
    template["Pred"] = predictions
    template.to_csv(output_path, index=False)
    print(f"\n  📝 Submission written to: {output_path}")
    print(f"     Rows: {len(template):,}")
    print(f"     Pred range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"     Mean pred: {predictions.mean():.4f}")

    # Sanity checks
    n_default = (predictions == 0.5).sum()
    if n_default > 0:
        print(f"     ⚠️  {n_default} matchups defaulted to 0.5 (missing team features)")

    return template


# ═══════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ─── Load data ───
    print("=" * 70)
    print("  Loading data...")
    print("=" * 70)
    m_data = load_gender_data("M")
    w_data = load_gender_data("W")

    # ─── Train/val split ───
    M_TRAIN_END, M_VAL = 2024, 2025
    W_TRAIN_END, W_VAL = 2024, 2025
    # Women's detailed data starts from 2010
    W_TRAIN_START = 2010
    M_TRAIN_START = 2003

    feature_names = get_feature_names()

    # ─── Men's matchups ───
    print(f"\n{'=' * 70}")
    print("  MEN'S TOURNAMENT MODEL")
    print(f"{'=' * 70}")
    m_matchups = build_matchup_dataset(m_data, M_TRAIN_START, M_VAL)
    m_train = m_matchups[m_matchups["Season"] <= M_TRAIN_END]
    m_val   = m_matchups[m_matchups["Season"] == M_VAL]
    X_m_train, y_m_train = m_train[feature_names].values, m_train["Label"].values
    X_m_val, y_m_val     = m_val[feature_names].values, m_val["Label"].values
    print(f"  Train: {len(X_m_train)} | Val (2025): {len(X_m_val)}")

    m_results = train_all_models(X_m_train, y_m_train, X_m_val, y_m_val, "Men's")
    m_best = print_comparison_table(m_results, "MEN'S")

    # ─── Women's matchups ───
    print(f"\n{'=' * 70}")
    print("  WOMEN'S TOURNAMENT MODEL")
    print(f"{'=' * 70}")
    w_matchups = build_matchup_dataset(w_data, W_TRAIN_START, W_VAL)
    w_train = w_matchups[w_matchups["Season"] <= W_TRAIN_END]
    w_val   = w_matchups[w_matchups["Season"] == W_VAL]
    X_w_train, y_w_train = w_train[feature_names].values, w_train["Label"].values
    X_w_val, y_w_val     = w_val[feature_names].values, w_val["Label"].values
    print(f"  Train: {len(X_w_train)} | Val (2025): {len(X_w_val)}")

    w_results = train_all_models(X_w_train, y_w_train, X_w_val, y_w_val, "Women's")
    w_best = print_comparison_table(w_results, "WOMEN'S")

    # ─── Build 2026 features ───
    print(f"\n{'=' * 70}")
    print("  Building 2026 features for ALL teams...")
    print(f"{'=' * 70}")

    m_feat_2026 = build_features_for_season(2026, m_data)
    w_feat_2026 = build_features_for_season(2026, w_data)
    print(f"  Men's teams: {len(m_feat_2026)}  |  Women's teams: {len(w_feat_2026)}")

    m_feat_dict = {}
    for _, row in m_feat_2026.iterrows():
        m_feat_dict[row["TeamID"]] = np.array([row[c] for c in FEATURE_COLS])

    w_feat_dict = {}
    for _, row in w_feat_2026.iterrows():
        w_feat_dict[row["TeamID"]] = np.array([row[c] for c in FEATURE_COLS])

    # ─── Generate submission ───
    m_model = m_results[m_best]["model"]
    w_model = w_results[w_best]["model"]

    generate_submission(
        m_model, m_best, m_feat_dict,
        w_model, w_best, w_feat_dict,
        submission_template_path=f"{DATA}/SampleSubmissionStage2.csv",
        output_path="submission.csv",
    )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  ✅ Done! Total time: {elapsed/60:.1f} minutes")
    print(f"  Men's best model:   {m_best}")
    print(f"  Women's best model: {w_best}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
