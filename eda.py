"""
March Madness 2026 — Exploratory Data Analysis (EDA) Toolkit
=============================================================
Interactive helper script to answer ad-hoc questions about the dataset.

Usage:
    python3 eda.py                      # Launch interactive menu
    python3 eda.py search "Miami"       # Search teams by name
    python3 eda.py team 1280            # Full team profile by ID
    python3 eda.py seeds 2025           # Show all seeds for a season
    python3 eda.py history 1181         # Tournament history for a team
    python3 eda.py compare 1181 1104    # Head-to-head comparison
    python3 eda.py top 2026             # Top 25 teams by win% this season
    python3 eda.py upsets 2025          # Biggest tournament upsets in a year
    python3 eda.py whynot "Miami OH"    # Why isn't this team in the bracket?
"""

import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

DATA = "data"

# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_all():
    """Load all men's datasets into a dict for easy access."""
    d = {}
    d["teams"]       = pd.read_csv(f"{DATA}/MTeams.csv")
    d["seeds"]       = pd.read_csv(f"{DATA}/MNCAATourneySeeds.csv")
    d["tourney"]     = pd.read_csv(f"{DATA}/MNCAATourneyCompactResults.csv")
    d["tourney_d"]   = pd.read_csv(f"{DATA}/MNCAATourneyDetailedResults.csv")
    d["reg"]         = pd.read_csv(f"{DATA}/MRegularSeasonCompactResults.csv")
    d["reg_d"]       = pd.read_csv(f"{DATA}/MRegularSeasonDetailedResults.csv")
    d["slots"]       = pd.read_csv(f"{DATA}/MNCAATourneySlots.csv")
    d["coaches"]     = pd.read_csv(f"{DATA}/MTeamCoaches.csv")
    d["conferences"] = pd.read_csv(f"{DATA}/MTeamConferences.csv")
    d["conf_names"]  = pd.read_csv(f"{DATA}/Conferences.csv")
    # Ordinals are huge — load lazily
    d["_ordinals_loaded"] = False
    return d

def ensure_ordinals(d):
    if not d["_ordinals_loaded"]:
        print("  Loading Massey Ordinals (128MB, one moment)...")
        d["ordinals"] = pd.read_csv(f"{DATA}/MMasseyOrdinals.csv")
        d["_ordinals_loaded"] = True


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def seed_number(seed_str):
    return int("".join(c for c in seed_str[1:] if c.isdigit()))

def team_name(d, team_id):
    row = d["teams"][d["teams"]["TeamID"] == team_id]
    return row.iloc[0]["TeamName"] if len(row) > 0 else f"Unknown ({team_id})"

def team_id_by_name(d, name):
    """Find team IDs matching a name (case-insensitive partial match)."""
    matches = d["teams"][d["teams"]["TeamName"].str.contains(name, case=False, na=False)]
    return matches


# ═══════════════════════════════════════════════════════════════
#  COMMANDS
# ═══════════════════════════════════════════════════════════════

def cmd_search(d, query):
    """Search teams by name."""
    matches = team_id_by_name(d, query)
    if len(matches) == 0:
        print(f"No teams found matching '{query}'")
        return
    print(f"\nTeams matching '{query}':")
    print(f"{'ID':>6s}  {'Name':<25s}  {'First D1':>8s}  {'Last D1':>7s}")
    print(f"{'─' * 50}")
    for _, row in matches.iterrows():
        print(f"{row['TeamID']:>6d}  {row['TeamName']:<25s}  {row['FirstD1Season']:>8d}  {row['LastD1Season']:>7d}")


def cmd_team_profile(d, team_id):
    """Full profile for a team."""
    team_id = int(team_id)
    name = team_name(d, team_id)
    team_row = d["teams"][d["teams"]["TeamID"] == team_id]
    
    if len(team_row) == 0:
        print(f"Team ID {team_id} not found.")
        return
    
    tr = team_row.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"  {name} (ID: {team_id})")
    print(f"{'=' * 60}")
    print(f"  D1 Seasons: {tr['FirstD1Season']} – {tr['LastD1Season']}")
    
    # Conference
    conf = d["conferences"][d["conferences"]["TeamID"] == team_id].sort_values("Season")
    if len(conf) > 0:
        latest_conf = conf.iloc[-1]["ConfAbbrev"]
        conf_name_row = d["conf_names"][d["conf_names"]["ConfAbbrev"] == latest_conf]
        conf_full = conf_name_row.iloc[0]["Description"] if len(conf_name_row) > 0 else latest_conf
        print(f"  Conference: {conf_full} ({latest_conf})")
    
    # Current coach
    coaches = d["coaches"][d["coaches"]["TeamID"] == team_id].sort_values("Season")
    if len(coaches) > 0:
        latest = coaches.iloc[-1]
        print(f"  Coach ({latest['Season']}): {latest['CoachName']}")
    
    # Current season record
    for season in [2026, 2025]:
        rs = d["reg"][d["reg"]["Season"] == season]
        wins = len(rs[rs["WTeamID"] == team_id])
        losses = len(rs[rs["LTeamID"] == team_id])
        if wins + losses > 0:
            w_scores = rs[rs["WTeamID"] == team_id]["WScore"]
            l_scores = rs[rs["LTeamID"] == team_id]["LScore"]
            ppg = (w_scores.sum() + l_scores.sum()) / (wins + losses)
            opp_w = rs[rs["WTeamID"] == team_id]["LScore"]
            opp_l = rs[rs["LTeamID"] == team_id]["WScore"]
            opp_ppg = (opp_w.sum() + opp_l.sum()) / (wins + losses)
            print(f"\n  {season} Regular Season: {wins}-{losses} ({wins/(wins+losses):.1%})")
            print(f"    PPG: {ppg:.1f}  |  Opp PPG: {opp_ppg:.1f}  |  Margin: {ppg - opp_ppg:+.1f}")
            break
    
    # Tournament history
    seeds = d["seeds"][d["seeds"]["TeamID"] == team_id].sort_values("Season")
    if len(seeds) > 0:
        print(f"\n  Tournament Appearances: {len(seeds)}")
        recent = seeds.tail(10)
        appearances = []
        for _, s in recent.iterrows():
            yr = s["Season"]
            sd = s["Seed"]
            sn = seed_number(sd)
            # How far did they go?
            t_games = d["tourney"][(d["tourney"]["Season"] == yr) & 
                                   ((d["tourney"]["WTeamID"] == team_id) | 
                                    (d["tourney"]["LTeamID"] == team_id))]
            t_wins = len(t_games[t_games["WTeamID"] == team_id])
            rounds = {0: "R64", 1: "R32", 2: "S16", 3: "E8", 4: "F4", 5: "Runner-Up", 6: "🏆 Champ"}
            result = rounds.get(t_wins, f"{t_wins}W")
            appearances.append(f"    {yr}: ({sn:>2d}) seed → {result}")
        print("  Recent:")
        for a in appearances:
            print(a)
    else:
        print(f"\n  ⚠️  No tournament appearances found.")


def cmd_seeds(d, season):
    """Show all seeds for a season."""
    season = int(season)
    s = d["seeds"][d["seeds"]["Season"] == season].copy()
    if len(s) == 0:
        print(f"No seeds found for {season}")
        return
    s["SeedNum"] = s["Seed"].apply(seed_number)
    s["TeamName"] = s["TeamID"].apply(lambda x: team_name(d, x))
    s = s.sort_values(["Seed"])
    
    print(f"\n{'=' * 50}")
    print(f"  {season} NCAA Tournament Seeds ({len(s)} teams)")
    print(f"{'=' * 50}")
    
    regions = {"W": "WEST", "X": "EAST", "Y": "SOUTH", "Z": "MIDWEST"}
    for region_char, region_name in regions.items():
        region_seeds = s[s["Seed"].str.startswith(region_char)].sort_values("SeedNum")
        print(f"\n  {region_name}:")
        for _, row in region_seeds.iterrows():
            print(f"    ({row['SeedNum']:>2d}) {row['TeamName']}")


def cmd_history(d, team_id):
    """Tournament history for a team."""
    team_id = int(team_id)
    name = team_name(d, team_id)
    seeds = d["seeds"][d["seeds"]["TeamID"] == team_id].sort_values("Season")
    
    print(f"\n{'=' * 60}")
    print(f"  {name} — Tournament History")
    print(f"{'=' * 60}")
    
    if len(seeds) == 0:
        print("  No tournament appearances.")
        return
    
    print(f"\n  Total Appearances: {len(seeds)}")
    print(f"\n  {'Year':>6s}  {'Seed':>4s}  {'Result':<12s}  Details")
    print(f"  {'─' * 50}")
    
    for _, s in seeds.iterrows():
        yr = s["Season"]
        sn = seed_number(s["Seed"])
        
        t_games = d["tourney"][(d["tourney"]["Season"] == yr) & 
                                ((d["tourney"]["WTeamID"] == team_id) | 
                                 (d["tourney"]["LTeamID"] == team_id))]
        t_games = t_games.sort_values("DayNum")
        
        t_wins = len(t_games[t_games["WTeamID"] == team_id])
        rounds_map = {0: "R64 Loss", 1: "R32 Loss", 2: "S16 Loss", 3: "E8 Loss", 
                      4: "F4 Loss", 5: "Runner-Up", 6: "🏆 Champion"}
        result = rounds_map.get(t_wins, f"{t_wins} wins")
        
        # Last game detail
        detail = ""
        if len(t_games) > 0:
            last = t_games.iloc[-1]
            if last["WTeamID"] == team_id:
                opp = team_name(d, last["LTeamID"])
                detail = f"W {int(last['WScore'])}-{int(last['LScore'])} vs {opp}"
            else:
                opp = team_name(d, last["WTeamID"])
                detail = f"L {int(last['LScore'])}-{int(last['WScore'])} vs {opp}"
        
        print(f"  {yr:>6d}  ({sn:>2d})  {result:<12s}  {detail}")


def cmd_compare(d, id_a, id_b):
    """Head-to-head comparison of two teams."""
    id_a, id_b = int(id_a), int(id_b)
    name_a = team_name(d, id_a)
    name_b = team_name(d, id_b)
    
    print(f"\n{'=' * 60}")
    print(f"  {name_a} vs {name_b}")
    print(f"{'=' * 60}")
    
    # All-time matchups
    reg = d["reg"]
    tourney = d["tourney"]
    
    all_games = pd.concat([reg, tourney])
    matchups = all_games[((all_games["WTeamID"] == id_a) & (all_games["LTeamID"] == id_b)) |
                          ((all_games["WTeamID"] == id_b) & (all_games["LTeamID"] == id_a))]
    
    a_wins = len(matchups[matchups["WTeamID"] == id_a])
    b_wins = len(matchups[matchups["WTeamID"] == id_b])
    
    print(f"\n  All-time record: {name_a} {a_wins} - {b_wins} {name_b}")
    
    # Current season comparison
    for season in [2026, 2025]:
        rs = reg[reg["Season"] == season]
        stats = {}
        for tid, tname in [(id_a, name_a), (id_b, name_b)]:
            wins = len(rs[rs["WTeamID"] == tid])
            losses = len(rs[rs["LTeamID"] == tid])
            if wins + losses > 0:
                w_pts = rs[rs["WTeamID"] == tid]["WScore"].sum()
                l_pts = rs[rs["LTeamID"] == tid]["LScore"].sum()
                ppg = (w_pts + l_pts) / (wins + losses)
                opp_ppg = (rs[rs["WTeamID"] == tid]["LScore"].sum() + 
                           rs[rs["LTeamID"] == tid]["WScore"].sum()) / (wins + losses)
                stats[tid] = {"W": wins, "L": losses, "PPG": ppg, "OppPPG": opp_ppg}
        
        if len(stats) == 2:
            print(f"\n  {season} Season Comparison:")
            print(f"  {'Stat':<12s}  {name_a:>15s}  {name_b:>15s}")
            print(f"  {'─' * 45}")
            sa, sb = stats[id_a], stats[id_b]
            print(f"  {'Record':<12s}  {sa['W']:>7d}-{sa['L']:<7d}  {sb['W']:>7d}-{sb['L']:<7d}")
            print(f"  {'Win %':<12s}  {sa['W']/(sa['W']+sa['L']):>15.1%}  {sb['W']/(sb['W']+sb['L']):>15.1%}")
            print(f"  {'PPG':<12s}  {sa['PPG']:>15.1f}  {sb['PPG']:>15.1f}")
            print(f"  {'Opp PPG':<12s}  {sa['OppPPG']:>15.1f}  {sb['OppPPG']:>15.1f}")
            print(f"  {'Margin':<12s}  {sa['PPG']-sa['OppPPG']:>+15.1f}  {sb['PPG']-sb['OppPPG']:>+15.1f}")
            break


def cmd_top(d, season):
    """Top 25 teams by win% for a season."""
    season = int(season)
    rs = d["reg"][d["reg"]["Season"] == season]
    
    w = rs.groupby("WTeamID").agg(W=("WScore", "size"), WPts=("WScore", "sum"), 
                                   WOpp=("LScore", "sum")).reset_index().rename(columns={"WTeamID": "TeamID"})
    l = rs.groupby("LTeamID").agg(L=("LScore", "size"), LPts=("LScore", "sum"),
                                   LOpp=("WScore", "sum")).reset_index().rename(columns={"LTeamID": "TeamID"})
    stats = w.merge(l, on="TeamID", how="outer").fillna(0)
    stats["G"] = stats["W"] + stats["L"]
    stats["WinPct"] = stats["W"] / stats["G"]
    stats["PPG"] = (stats["WPts"] + stats["LPts"]) / stats["G"]
    stats["OppPPG"] = (stats["WOpp"] + stats["LOpp"]) / stats["G"]
    stats["Margin"] = stats["PPG"] - stats["OppPPG"]
    stats["TeamName"] = stats["TeamID"].apply(lambda x: team_name(d, x))
    
    # Check if team is seeded
    seeds = d["seeds"][d["seeds"]["Season"] == season]
    seeded_ids = set(seeds["TeamID"].values) if len(seeds) > 0 else set()
    
    top = stats.sort_values("WinPct", ascending=False).head(25)
    
    print(f"\n{'=' * 70}")
    print(f"  Top 25 Teams by Win% — {season} Regular Season")
    print(f"{'=' * 70}")
    print(f"  {'#':>3s}  {'Team':<22s}  {'Record':>8s}  {'Win%':>6s}  {'PPG':>6s}  {'Margin':>7s}  {'Seeded':>6s}")
    print(f"  {'─' * 65}")
    
    for i, (_, row) in enumerate(top.iterrows(), 1):
        record = f"{int(row['W'])}-{int(row['L'])}"
        seeded = "✅" if row["TeamID"] in seeded_ids else "❌"
        print(f"  {i:>3d}  {row['TeamName']:<22s}  {record:>8s}  {row['WinPct']:>5.1%}  "
              f"{row['PPG']:>6.1f}  {row['Margin']:>+7.1f}  {seeded:>6s}")


def cmd_upsets(d, season):
    """Biggest tournament upsets in a season."""
    season = int(season)
    t = d["tourney"][d["tourney"]["Season"] == season].copy()
    s = d["seeds"][d["seeds"]["Season"] == season].copy()
    s["SeedNum"] = s["Seed"].apply(seed_number)
    
    t = t.merge(s[["TeamID", "SeedNum"]].rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}),
                on="WTeamID", how="left")
    t = t.merge(s[["TeamID", "SeedNum"]].rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}),
                on="LTeamID", how="left")
    t["SeedDiff"] = t["WSeed"] - t["LSeed"]  # Positive = upset (higher seed won)
    t["WName"] = t["WTeamID"].apply(lambda x: team_name(d, x))
    t["LName"] = t["LTeamID"].apply(lambda x: team_name(d, x))
    
    upsets = t[t["SeedDiff"] > 0].sort_values("SeedDiff", ascending=False)
    
    print(f"\n{'=' * 70}")
    print(f"  Biggest Upsets — {season} NCAA Tournament")
    print(f"{'=' * 70}")
    
    if len(upsets) == 0:
        print("  No upsets found (or no data for this season).")
        return
    
    for _, row in upsets.head(20).iterrows():
        print(f"  ({int(row['WSeed']):>2d}) {row['WName']:<20s} {int(row['WScore']):>3d} - "
              f"{int(row['LScore']):<3d} ({int(row['LSeed']):>2d}) {row['LName']:<20s}  "
              f"[+{int(row['SeedDiff'])} seed diff]")


def cmd_whynot(d, query):
    """Investigate why a team isn't in the bracket."""
    matches = team_id_by_name(d, query)
    if len(matches) == 0:
        print(f"No teams found matching '{query}'")
        return
    
    # Use most recent bracket season
    bracket_season = 2025  # current template
    seeds = d["seeds"][d["seeds"]["Season"] == bracket_season]
    seeded_ids = set(seeds["TeamID"].values)
    
    for _, team_row in matches.iterrows():
        tid = team_row["TeamID"]
        name = team_row["TeamName"]
        is_seeded = tid in seeded_ids
        
        print(f"\n{'=' * 60}")
        print(f"  Why isn't {name} (ID: {tid}) in the bracket?")
        print(f"{'=' * 60}")
        
        if is_seeded:
            seed_row = seeds[seeds["TeamID"] == tid].iloc[0]
            print(f"\n  ✅ Actually, {name} IS in the {bracket_season} bracket!")
            print(f"     Seed: {seed_row['Seed']} (#{seed_number(seed_row['Seed'])})")
            continue
        
        # Check their record
        for season in [2026, 2025]:
            rs = d["reg"][d["reg"]["Season"] == season]
            wins = len(rs[rs["WTeamID"] == tid])
            losses = len(rs[rs["LTeamID"] == tid])
            if wins + losses == 0:
                continue
            
            w_pts = rs[rs["WTeamID"] == tid]["WScore"].sum()
            l_pts = rs[rs["LTeamID"] == tid]["LScore"].sum()
            ppg = (w_pts + l_pts) / (wins + losses)
            opp_ppg = (rs[rs["WTeamID"] == tid]["LScore"].sum() + 
                       rs[rs["LTeamID"] == tid]["WScore"].sum()) / (wins + losses)
            margin = ppg - opp_ppg
            win_pct = wins / (wins + losses)
            
            print(f"\n  {season} Record: {wins}-{losses} ({win_pct:.1%})")
            print(f"  PPG: {ppg:.1f} | Opp PPG: {opp_ppg:.1f} | Margin: {margin:+.1f}")
            
            # Conference
            conf = d["conferences"][(d["conferences"]["TeamID"] == tid) & 
                                     (d["conferences"]["Season"] == season)]
            if len(conf) > 0:
                conf_abbr = conf.iloc[0]["ConfAbbrev"]
                conf_name_row = d["conf_names"][d["conf_names"]["ConfAbbrev"] == conf_abbr]
                conf_full = conf_name_row.iloc[0]["Description"] if len(conf_name_row) > 0 else conf_abbr
                print(f"  Conference: {conf_full} ({conf_abbr})")
            
            # Compare to the worst seeded team
            if len(seeds) > 0:
                seeded_teams_stats = []
                for sid in seeded_ids:
                    sw = len(rs[rs["WTeamID"] == sid])
                    sl = len(rs[rs["LTeamID"] == sid])
                    if sw + sl > 0:
                        seeded_teams_stats.append({
                            "TeamID": sid, "W": sw, "L": sl,
                            "WinPct": sw / (sw + sl)
                        })
                if seeded_teams_stats:
                    worst = min(seeded_teams_stats, key=lambda x: x["WinPct"])
                    worst_name = team_name(d, worst["TeamID"])
                    print(f"\n  📊 Comparison to weakest seeded team:")
                    print(f"     {name}: {wins}-{losses} ({win_pct:.1%})")
                    print(f"     {worst_name}: {worst['W']}-{worst['L']} ({worst['WinPct']:.1%})")
                    
                    if win_pct < worst["WinPct"]:
                        print(f"\n  💡 {name}'s win% ({win_pct:.1%}) is below the weakest seeded team.")
                    else:
                        print(f"\n  💡 {name}'s win% ({win_pct:.1%}) is actually competitive!")
                        print(f"     They may have been a bubble team or had a weak SOS.")
            
            # Tournament history
            t_apps = d["seeds"][d["seeds"]["TeamID"] == tid]
            last_app = t_apps["Season"].max() if len(t_apps) > 0 else None
            print(f"\n  Tournament History: {len(t_apps)} appearances")
            if last_app:
                print(f"  Last appearance: {int(last_app)}")
            else:
                print(f"  ⚠️  Has NEVER made the tournament")
            break


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════

def interactive(d):
    """Interactive Q&A mode."""
    print("\n" + "=" * 60)
    print("  🏀 March Madness EDA Toolkit — Interactive Mode")
    print("=" * 60)
    print("""
Commands:
  search <name>       Search teams by name
  team <id>           Full team profile
  seeds <year>        Show tournament seeds
  history <id>        Tournament history for a team
  compare <id> <id>   Head-to-head comparison
  top <year>          Top 25 by win%
  upsets <year>       Biggest tournament upsets
  whynot <name>       Why isn't this team in the bracket?
  quit                Exit
""")
    
    while True:
        try:
            raw = input("eda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 🏀")
            break
        
        if not raw:
            continue
        
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd in ("quit", "exit", "q"):
                print("Bye! 🏀")
                break
            elif cmd == "search":
                cmd_search(d, args)
            elif cmd == "team":
                cmd_team_profile(d, args)
            elif cmd == "seeds":
                cmd_seeds(d, args)
            elif cmd == "history":
                cmd_history(d, args)
            elif cmd == "compare":
                ids = args.split()
                if len(ids) == 2:
                    cmd_compare(d, ids[0], ids[1])
                else:
                    print("Usage: compare <team_id_1> <team_id_2>")
            elif cmd == "top":
                cmd_top(d, args if args else "2026")
            elif cmd == "upsets":
                cmd_upsets(d, args if args else "2025")
            elif cmd == "whynot":
                cmd_whynot(d, args)
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands or 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    d = load_all()
    print("Done!\n")
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        args = sys.argv[2:] 
        
        if cmd == "search":
            cmd_search(d, " ".join(args))
        elif cmd == "team":
            cmd_team_profile(d, args[0])
        elif cmd == "seeds":
            cmd_seeds(d, args[0])
        elif cmd == "history":
            cmd_history(d, args[0])
        elif cmd == "compare":
            cmd_compare(d, args[0], args[1])
        elif cmd == "top":
            cmd_top(d, args[0] if args else "2026")
        elif cmd == "upsets":
            cmd_upsets(d, args[0] if args else "2025")
        elif cmd == "whynot":
            cmd_whynot(d, " ".join(args))
        else:
            print(f"Unknown command: {cmd}")
    else:
        interactive(d)


if __name__ == "__main__":
    main()
