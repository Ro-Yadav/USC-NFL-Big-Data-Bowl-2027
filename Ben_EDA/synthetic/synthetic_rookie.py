"""
Synthetic stand-in for the Combine -> rookie-injury notebook.

Matches the repo/external schema: data/players.csv, combine_results.csv, combine_tracking.csv,
player_career_successes.csv, external/players.csv (nflverse IDs), external/injuries_<year>.csv,
plus nflverse-shaped snap counts and schedules. 510 players from the 2023-2025 draft classes.

Planted signal (so the pipeline can be checked): players whose Combine tracking shows weaker
braking (lower peak deceleration) and a bigger drop from attempt 1 to attempt 2 are more likely
to be injured as rookies. Timed results carry no injury signal. Nothing here is real data.
"""
import numpy as np
import pandas as pd

FPS = 10
POSITIONS = (["CB"] * 60 + ["FS"] * 25 + ["SS"] * 28 + ["WR"] * 106 + ["TE"] * 42 + ["T"] * 50 + ["G"] * 51
             + ["C"] * 20 + ["DE"] * 36 + ["DT"] * 47 + ["OLB"] * 30 + ["NT"] * 5 + ["RB"] * 10)
DRILLS = {
    "skill": ["THREE_CONE_DRILL", "SHORT_SHUTTLE", "FORTY_YARD_DASH", "W_DRILL", "IN_ROUTE_LEFT"],
    "line": ["THREE_CONE_DRILL", "SHORT_SHUTTLE", "FORTY_YARD_DASH", "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
             "PASS_PRO_MIRROR_DRILL"],
}


def _drill_path(n_cuts, burst, braking, turning, fatigue, rng, n=70):
    """Accelerate, brake into each cut, turn, re-accelerate."""
    s, h, x, y = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    cuts = sorted(rng.choice(np.arange(14, n - 10), n_cuts, replace=False)) if n_cuts else []
    heading, speed = 0.0, 0.0
    turn_left = 0
    for t in range(1, n):
        near = [c for c in cuts if 0 <= c - t <= 5]
        if near:                                            # braking into the cut
            speed = max(2.5, speed - braking * fatigue / FPS)
        else:
            speed = min(7.5, speed + burst * fatigue / FPS)
        if t in cuts:                                       # turn spread over 4 frames
            turn_left = rng.choice([-1, 1]) * rng.uniform(90, 180)
            step = turn_left / 4
        if turn_left:
            heading += step
            turn_left = 0 if abs(turn_left - step) < 1e-6 else turn_left - step
            speed *= (0.6 + 0.08 * turning) ** 0.25
        s[t] = max(0, speed + rng.normal(0, 0.05))
        h[t] = heading
        x[t] = x[t - 1] + s[t] / FPS * np.sin(np.deg2rad(heading))
        y[t] = y[t - 1] + s[t] / FPS * np.cos(np.deg2rad(heading))
    return x, y, s, h % 360


def make_all(seed=5):
    rng = np.random.default_rng(seed)
    n = len(POSITIONS)
    classes = np.repeat([2023, 2024, 2025], [160, 177, n - 337])
    rng.shuffle(classes)
    pl = pd.DataFrame(dict(nfl_id=np.arange(55000, 55000 + n), nfl_position=POSITIONS, draft_year=classes,
                           draft_round=rng.integers(1, 8, n)))
    pl["display_name"] = [f"R{i}" for i in pl.nfl_id]
    pl["gsis_id"] = [f"00-{i:07d}" for i in pl.nfl_id]
    burst, braking, turning, fade = (rng.normal(0, 1, n) for _ in range(4))
    pl["_braking"], pl["_fade"] = braking, fade

    # rookie injury: weaker braking and more fade -> higher risk
    logit = -1.6 - 0.75 * braking + 0.45 * fade
    pl["_inj"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    pl["_games_out"] = np.where(pl._inj, rng.integers(1, 7, n), 0)

    # ---- Combine results (no injury signal) ----
    line = pl.nfl_position.isin(["T", "G", "C", "DE", "DT", "NT"])
    cr = pd.DataFrame(dict(draft_year=pl.draft_year, nfl_id=pl.nfl_id, combine_position=pl.nfl_position,
                           combine_weight=np.where(line, rng.normal(300, 15, n), rng.normal(200, 15, n)).round(),
                           forty=np.where(line, rng.normal(5.1, .15, n), rng.normal(4.55, .1, n)).round(2),
                           ten_yd_split=(np.where(line, 1.78, 1.55) - 0.03 * burst + rng.normal(0, .03, n)).round(2),
                           vertical=rng.normal(33, 3, n).round(1), broad_jump=rng.normal(118, 7, n).round(),
                           three_cone=(np.where(line, 7.7, 6.95) - 0.05 * turning + rng.normal(0, .12, n)).round(2),
                           short_shuttle=np.where(line, rng.normal(4.7, .12, n), rng.normal(4.2, .1, n)).round(2),
                           bench_reps=rng.normal(20, 5, n).round()))
    for c in ["three_cone", "short_shuttle", "vertical", "broad_jump"]:          # not everyone does every drill
        cr.loc[rng.random(n) < 0.35, c] = np.nan

    # ---- Combine tracking ----
    t0, eid, parts = pd.Timestamp("2023-03-01"), 0, []
    for i, p in pl.iterrows():
        drills = DRILLS["line" if line[i] else "skill"]
        for d in drills:
            n_att = 2 if rng.random() < 0.3 else 1
            for att in range(1, n_att + 1):
                eid += 1
                fat = 1.0 if att == 1 else 1 - 0.04 * (1 + fade[i])
                n_cuts = {"FORTY_YARD_DASH": 0, "THREE_CONE_DRILL": 3, "SHORT_SHUTTLE": 2}.get(d, 2)
                x, y, s, h = _drill_path(n_cuts, 7 + burst[i], 9 + 1.5 * braking[i], turning[i], fat, rng)
                ts = (t0 + pd.to_timedelta(eid * 100 + np.arange(len(s)) / FPS, unit="s")).strftime("%Y-%m-%d %H:%M:%S.%f")
                parts.append(pd.DataFrame(dict(draft_year=p.draft_year, event_id=eid, nfl_id=p.nfl_id,
                                               entity_type="player", time=ts, drill_type="x", drill_name=d,
                                               attempt=att, x=x, y=y, s=s, a=np.abs(np.gradient(s)) * FPS,
                                               dis=s / FPS, dir=h)))
    ct = pd.concat(parts, ignore_index=True)

    # ---- schedule, snaps, injury reports (nflverse shapes) ----
    teams = [f"T{k:02d}" for k in range(32)]
    sched = []
    for season in [2023, 2024, 2025]:
        for wk in range(1, 19):
            order = rng.permutation(teams)
            for k in range(0, 32, 2):
                sched.append(dict(season=season, game_type="REG", week=wk, home_team=order[k], away_team=order[k + 1],
                                  game_id=f"{season}_{wk:02d}_{order[k]}"))
    sched = pd.DataFrame(sched)
    pl["team"] = rng.choice(teams, n)
    usage = rng.uniform(0.05, 0.9, n)
    snap_rows, inj_rows = [], []
    for i, p in pl.iterrows():
        onset = int(rng.integers(3, 15)) if p._inj else None
        for season in range(p.draft_year, 2026):
            for wk in range(1, 19):
                out_now = season == p.draft_year and onset and onset <= wk < onset + p._games_out
                if out_now and wk - onset < 2:             # IR players drop off the report after ~2 weeks
                    inj_rows.append(dict(season=season, game_type="REG", team=p.team, week=wk, gsis_id=p.gsis_id,
                                         position=p.nfl_position, full_name=p.display_name, first_name="", last_name="",
                                         report_primary_injury=rng.choice(["Knee", "Ankle", "Hamstring", "Shoulder"]),
                                         report_secondary_injury=None, report_status="Out",
                                         practice_primary_injury=None, practice_secondary_injury=None,
                                         practice_status="Did Not Participate In Practice", date_modified=None))
                if out_now or rng.random() < 0.08:
                    continue
                pct = float(np.clip(usage[i] + rng.normal(0, .08), 0, 1))
                off = p.nfl_position in ["WR", "TE", "T", "G", "C", "RB"]
                snap_rows.append(dict(game_id=f"{season}_{wk:02d}", season=season, game_type="REG", week=wk,
                                      player=p.display_name, pfr_player_id=f"PFR{p.nfl_id}", position=p.nfl_position,
                                      team=p.team, opponent="", offense_snaps=int(pct * 65) if off else 0,
                                      offense_pct=pct if off else 0.0, defense_snaps=0 if off else int(pct * 65),
                                      defense_pct=0.0 if off else pct, st_snaps=0, st_pct=0.0))
    snaps = pd.DataFrame(snap_rows)
    injuries = pd.DataFrame(inj_rows)
    ext_players = pd.DataFrame(dict(gsis_id=pl.gsis_id, nfl_id=pl.nfl_id.astype(float), pfr_id=[f"PFR{i}" for i in pl.nfl_id],
                                    display_name=pl.display_name, position=pl.nfl_position))
    career = pd.DataFrame(dict(nfl_id=pl.nfl_id, career_offensive_snaps=0, career_defensive_snaps=0,
                               career_special_teams_snaps=0,
                               career_games_active=snaps.groupby("pfr_player_id").size().reindex(
                                   [f"PFR{i}" for i in pl.nfl_id]).fillna(0).astype(int).values,
                               career_games_started=0, ap_all_pro_1st_team=0, ap_all_pro_2nd_team=0,
                               pro_bowl_original_ballot=0))
    players = pl[["nfl_id", "display_name", "draft_year", "nfl_position", "draft_round"]].copy()
    return dict(players=players, combine_results=cr, combine_tracking=ct, career=career, ext_players=ext_players,
                injuries=injuries, snaps=snaps, schedule=sched, truth=pl)
