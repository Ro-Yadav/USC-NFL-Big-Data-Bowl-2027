"""
Synthetic stand-in for the injury projection notebook.

Produces the same inputs the notebook reads on real data: data/games.csv, data/players.csv,
the cross-position notebook's cached rep-level files (one per family), external/players.csv,
external/injuries_<year>.csv, and nflverse-shaped snaps and schedules.

Planted signal: players injured in the next season had a declining top speed and lower speed
kept through cuts in the current season. Nothing here is real data.
"""
import numpy as np
import pandas as pd

FAMILY_POS = {"db": ["CB", "FS", "SS"], "receivers": ["WR", "TE"], "ol": ["T", "G", "C"], "front7": ["DE", "DT", "OLB"]}
CUTTING = {"db", "receivers"}


def make_all(seed=9, per_family=110):
    rng = np.random.default_rng(seed)
    teams = [f"T{k:02d}" for k in range(32)]
    sched, games = [], []
    gid = 2023090001
    for season in [2023, 2024, 2025]:
        for wk in range(1, 19):
            order = rng.permutation(teams)
            for k in range(0, 32, 2):
                sched.append(dict(season=season, game_type="REG", week=wk, home_team=order[k], away_team=order[k + 1],
                                  game_id=f"{season}_{wk:02d}_{order[k]}"))
                games.append(dict(game_id=gid, season=season, season_type="REG", week=wk,
                                  home_team_abbr=order[k], visitor_team_abbr=order[k + 1]))
                gid += 1
    sched, games = pd.DataFrame(sched), pd.DataFrame(games)

    rows, nid = [], 70000
    for fam, poss in FAMILY_POS.items():
        for i in range(per_family):
            nid += 1
            rows.append(dict(nfl_id=nid, nfl_position=poss[i % len(poss)], draft_year=int(rng.choice([2023, 2024])),
                             draft_round=int(rng.integers(1, 8)), display_name=f"P{nid}", fam=fam,
                             team=rng.choice(teams), gsis_id=f"00-{nid:07d}"))
    pl = pd.DataFrame(rows)
    risk = rng.normal(0, 1, len(pl))                       # latent risk, drives both signal and next-season injury
    pl["_risk"] = risk

    reps = {f: [] for f in FAMILY_POS}
    inj_rows, snap_rows = [], []
    for i, p in pl.iterrows():
        for season in range(p.draft_year, 2026):
            team_games = games[(games.season == season) & ((games.home_team_abbr == p.team) | (games.visitor_team_abbr == p.team))]
            injured_next = season < 2025 and rng.random() < 1 / (1 + np.exp(-(-1.5 + 0.9 * p._risk)))
            pl.loc[i, f"_inj_{season + 1}"] = injured_next
            onset = int(rng.integers(3, 15)) if (season > p.draft_year and pl.loc[i].get(f"_inj_{season}", False)) else None
            for g in team_games.itertuples():
                if onset and onset <= g.week < onset + 4:
                    if g.week - onset < 2:
                        inj_rows.append(dict(season=season, game_type="REG", team=p.team, week=g.week, gsis_id=p.gsis_id,
                                             position=p.nfl_position, full_name=p.display_name, first_name="", last_name="",
                                             report_primary_injury="Knee", report_secondary_injury=None, report_status="Out",
                                             practice_primary_injury=None, practice_secondary_injury=None,
                                             practice_status=None, date_modified=None))
                    continue
                pct = float(np.clip(rng.normal(0.6, 0.15), 0.1, 1))
                snap_rows.append(dict(game_id=f"{season}_{g.week:02d}", season=season, game_type="REG", week=g.week,
                                      player=p.display_name, pfr_player_id=f"PFR{p.nfl_id}", position=p.nfl_position,
                                      team=p.team, opponent="", offense_snaps=int(pct * 60), offense_pct=pct,
                                      defense_snaps=int(pct * 60), defense_pct=pct, st_snaps=0, st_pct=0.0))
                frac = (g.week - 1) / 17
                drift = -0.05 * frac * max(p._risk, 0)        # higher risk -> top speed declines through the season
                for play in range(int(rng.integers(8, 20))):
                    if p.fam in CUTTING:
                        reps[p.fam].append(dict(gameId=g.game_id, playId=play, nflId=p.nfl_id,
                                                max_s=8.5 * (1 + drift) + rng.normal(0, 0.5),
                                                max_acc=rng.normal(4, 0.6), max_decel=rng.normal(5, 0.8),
                                                max_turn=rng.normal(300, 40),
                                                retention=np.clip(0.8 - 0.05 * p._risk + rng.normal(0, 0.12), 0.2, 1.2),
                                                n_cuts=int(rng.integers(0, 3))))
                    else:
                        reps[p.fam].append(dict(gameId=g.game_id, playId=play, nflId=p.nfl_id,
                                                get_off=rng.normal(0.8, 0.1), first_step=rng.normal(3.2, 0.5),
                                                peak_acc=rng.normal(3.0, 0.5), stop_power=rng.normal(3.0, 0.5),
                                                top_speed=4.0 * (1 + drift) + rng.normal(0, 0.3)))
    reps = {f: pd.DataFrame(v) for f, v in reps.items()}
    players = pl[["nfl_id", "display_name", "draft_year", "nfl_position", "draft_round"]].copy()
    ext_players = pd.DataFrame(dict(gsis_id=pl.gsis_id, nfl_id=pl.nfl_id.astype(float), pfr_id=[f"PFR{i}" for i in pl.nfl_id],
                                    display_name=pl.display_name, position=pl.nfl_position))
    return dict(games=games, players=players, reps=reps, ext_players=ext_players, injuries=pd.DataFrame(inj_rows),
                snaps=pd.DataFrame(snap_rows), schedule=sched, truth=pl)
