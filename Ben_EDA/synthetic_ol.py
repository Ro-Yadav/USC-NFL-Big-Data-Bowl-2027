"""
Synthetic stand-in for the USC repo data, offensive-line version.

Same schema as ../data (game_tracking, games, player_play, players, combine_results,
combine_tracking) plus nflverse-shaped snaps / injuries / id_map / schedule, so the OL
notebook can be tested end to end. A few OL get a planted pre-injury pattern:
slower get-off and lower first-step acceleration in the weeks before onset, a
high-effort game two games out, then a drop in first-step acceleration. Nothing here is real.
"""
import numpy as np
import pandas as pd

FPS = 10


def _schedule(teams, n_weeks, rng):
    rows = []
    for wk in range(1, n_weeks + 1):
        order = rng.permutation(teams)
        for i in range(0, len(order), 2):
            rows.append(dict(game_id=f"S_{wk:02d}_{order[i+1]}_{order[i]}", season=2099, game_type="REG",
                             week=wk, home_team=order[i], away_team=order[i + 1]))
    return pd.DataFrame(rows)


def _ol_rep(n, start, play_right, is_pass, delay, a_max, decel, rng):
    """One OL rep from the snap: react, accelerate, set/contact, decelerate, then small slides."""
    if is_pass:
        heading = (270 if play_right else 90) + rng.uniform(-60, 60)   # kick back and out
        vmax, t_contact = rng.uniform(2.5, 3.5), int(rng.integers(10, 15))
    else:
        heading = (90 if play_right else 270) + rng.uniform(-25, 25)   # drive forward
        vmax, t_contact = rng.uniform(3.0, 4.2), int(rng.integers(7, 11))
    s = np.zeros(n)
    for t in range(1, n):
        if t < delay:
            s[t] = 0.0
        elif t < delay + t_contact:
            s[t] = min(vmax, s[t - 1] + a_max / FPS)
        else:
            s[t] = max(0.3, s[t - 1] - decel / FPS)
    s = np.clip(s + rng.normal(0, 0.05, n), 0, None)
    h = np.full(n, heading, float)
    if is_pass:                                                         # mirror slides after the set
        for t0 in range(delay + 14, n, 6):
            h[t0:] = (0 if rng.random() < 0.5 else 180) + rng.normal(0, 10)   # lateral slide (+/- y)
            s[t0:t0 + 4] = np.maximum(s[t0:t0 + 4], rng.uniform(1.0, 2.0))
    rad = np.deg2rad(h)
    x = start[0] + np.cumsum(s / FPS * np.sin(rad))
    y = start[1] + np.cumsum(s / FPS * np.cos(rad))
    a = np.abs(np.gradient(s)) * FPS
    return x, y, s, a, h % 360


def make_all(seed=11, n_teams=8, track_weeks=9, season_weeks=18, plays_per_game=60, frames=30):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    sched = _schedule(teams, season_weeks, rng)

    roster, pid = [], 60000
    for t in teams:
        for pos in ["T", "G", "C", "G", "T", "T", "G"]:                # 5 starters + 2 backups
            pid += 1
            roster.append(dict(nflId=pid, team=t, position=pos, displayName=f"OL{pid}",
                               delay=int(rng.integers(2, 5)), a_max=rng.normal(6.0, 0.6),
                               decel=rng.normal(7.0, 0.8), starter=len([r for r in roster if r["team"] == t]) < 5))
    players = pd.DataFrame(roster)
    starters = players[players.starter]

    inj_ids = rng.choice(starters.nflId.values, 8, replace=False)
    onset = {int(p): int(rng.integers(5, track_weeks + 1)) for p in inj_ids}
    spiked = set(int(p) for p in inj_ids[:5])
    injury_kinds = ["Knee", "Ankle", "Back", "Shoulder", "Knee", "Calf", "Hand", "Ankle"]

    def team_game(team, wk):
        g = sched[(sched.week == wk) & ((sched.home_team == team) | (sched.away_team == team))]
        return g.iloc[0] if len(g) else None

    snap_rows = []
    for p in players.itertuples():
        base = rng.uniform(0.93, 1.0) if p.starter else rng.uniform(0.0, 0.15)
        for wk in range(1, season_weeks + 1):
            g = team_game(p.team, wk)
            if g is None:
                continue
            pct = float(np.clip(base + rng.normal(0, 0.03), 0, 1))
            if p.nflId in onset:
                ow = onset[p.nflId]
                if wk == ow:
                    pct = rng.uniform(0.15, 0.4)
                elif ow < wk <= ow + 3:
                    continue
            snap_rows.append(dict(game_id=g.game_id, pfr_game_id=g.game_id, season=2099, game_type="REG", week=wk,
                                  player=p.displayName, pfr_player_id=f"PFR{p.nflId}", position=p.position,
                                  team=p.team, opponent="", offense_snaps=int(pct * 65), offense_pct=pct,
                                  defense_snaps=0, defense_pct=0.0, st_snaps=0, st_pct=0.0))
    snaps = pd.DataFrame(snap_rows)

    inj_rows = []
    for (p, ow), kind in zip(onset.items(), injury_kinds):
        r = players.set_index("nflId").loc[p]
        for wk in range(ow + 1, ow + 4):
            inj_rows.append(dict(season=2099, game_type="REG", team=r.team, week=wk, gsis_id=f"GSIS{p}",
                                 position=r.position, full_name=r.displayName, first_name="", last_name="",
                                 report_primary_injury=kind, report_secondary_injury=None, report_status="Out",
                                 practice_primary_injury=kind, practice_secondary_injury=None,
                                 practice_status="Did Not Participate In Practice", date_modified=None))
    injuries = pd.DataFrame(inj_rows)

    id_map = pd.DataFrame(dict(gsis_id=[f"GSIS{i}" for i in players.nflId], nfl_id=players.nflId.astype(str),
                               pfr_id=[f"PFR{i}" for i in players.nflId], display_name=players.displayName,
                               position=players.position))

    # ---- tracking (starters only, every offensive snap) ----
    tg = sched[sched.week <= track_weeks].reset_index(drop=True)
    games = pd.DataFrame(dict(game_id=np.arange(2099090001, 2099090001 + len(tg)), game_key=0, season=2099,
                              season_type="REG", week=tg.week, home_team_abbr=tg.home_team,
                              visitor_team_abbr=tg.away_team))
    t0 = pd.Timestamp("2099-09-10")
    trk, pp = [], []
    for g in games.itertuples():
        for play_id in range(1, plays_per_game + 1):
            off = g.home_team_abbr if play_id % 2 else g.visitor_team_abbr
            quarter = min(4, 1 + (play_id - 1) * 4 // plays_per_game)
            play_right = bool(rng.random() < 0.5)
            is_pass = bool(rng.random() < 0.58)
            line = starters[starters.team == off]
            for k, p in enumerate(line.itertuples()):
                if p.nflId in onset and g.week >= onset[p.nflId] + 1 and g.week <= onset[p.nflId] + 3:
                    continue
                rel = g.week - onset[p.nflId] if p.nflId in onset else 99
                delay, a_max, decel = p.delay, p.a_max, p.decel
                if -4 <= rel <= -1:                                    # slow decline into injury
                    delay += 1 if rng.random() < 0.15 * (5 + rel) else 0
                    a_max *= 1 - 0.015 * (5 + rel)
                if rel == -2 and p.nflId in spiked:                    # high-effort game (5 of 8 injured)
                    a_max *= 1.25
                if rel == -1 and p.nflId in spiked:                    # drop after it
                    a_max *= 0.92
                if quarter >= 3 and -3 <= rel <= -1:
                    a_max *= 0.95
                x, y, s, a, d = _ol_rep(frames, (60.0, 22 + 2.5 * k), play_right, is_pass, delay, a_max, decel, rng)
                ts = (t0 + pd.to_timedelta((g.game_id - games.game_id.min()) * 86400 + play_id * 60
                                           + np.arange(-2, frames) / FPS, unit="s")).strftime("%Y-%m-%d %H:%M:%S.%f")
                xs = np.r_[np.full(2, x[0]), x]; ys = np.r_[np.full(2, y[0]), y]
                ss = np.r_[[0, 0], s]; aa = np.r_[[0, 0], a]; dd = np.r_[np.full(2, d[0]), d]
                ev = np.array([None] * (frames + 2), dtype=object); ev[2] = "ball_snap"
                trk.append(pd.DataFrame(dict(game_id=g.game_id, play_id=play_id, nfl_id=p.nflId, time=ts,
                                             x=xs, y=ys, s=ss, a=aa, dis=ss / FPS, o=dd, dir=dd, event=ev)))
                pressure = float(is_pass and rng.random() < 0.06 + 0.03 * (delay - 2) + (0.04 if -2 <= rel <= -1 else 0))
                pp.append(dict(game_id=g.game_id, play_id=play_id, nfl_id=p.nflId, team_abbr=off,
                               possession_team=off, quarter=quarter, lined_up_position=p.position,
                               play_direction="right" if play_right else "left",
                               pass_result="C" if is_pass else None,
                               pressure_allowed=pressure if is_pass else None,
                               sack_allowed=0.0 if is_pass else None))
    game_tracking = pd.concat(trk, ignore_index=True)
    player_play = pd.DataFrame(pp)

    players_out = players.rename(columns={"nflId": "nfl_id", "displayName": "display_name",
                                          "position": "nfl_position"})[["nfl_id", "display_name", "nfl_position"]]
    n = len(players)
    combine_results = pd.DataFrame(dict(
        draft_year=2095, nfl_id=players.nflId, combine_position=players.position,
        combine_weight=rng.normal(312, 12, n).round(), arm_length=rng.normal(33.8, 0.8, n).round(2),
        hand_size=rng.normal(10.0, 0.4, n).round(2),
        ten_yd_split=(1.78 + (6.0 - players.a_max) * 0.04 + rng.normal(0, 0.03, n)).round(2),
        forty=rng.normal(5.2, 0.12, n).round(2), vertical=rng.normal(28, 2.5, n).round(1),
        broad_jump=rng.normal(104, 5, n).round(), three_cone=rng.normal(7.8, 0.2, n).round(2),
        short_shuttle=rng.normal(4.75, 0.12, n).round(2), bench_reps=rng.normal(26, 4, n).round()))

    ct, eid = [], 0
    for p in players.itertuples():
        for drill, nfr in [("40 Yard Dash", 60), ("3-Cone", 85), ("Short Shuttle", 50)]:
            for att in (1, 2):
                eid += 1
                s = np.minimum(np.arange(nfr) / FPS * p.a_max * 0.9, 7.0 if drill == "40 Yard Dash" else 4.5)
                s = np.clip(s + rng.normal(0, 0.05, nfr), 0, None)
                h = np.full(nfr, 90.0)
                if drill != "40 Yard Dash":
                    for c in range(15, nfr, 18):
                        h[c:] = (h[c - 1] + 180) % 360
                        s[c:c + 5] *= np.linspace(0.4, 1, 5)
                rad = np.deg2rad(h)
                x, y = np.cumsum(s / FPS * np.sin(rad)), np.cumsum(s / FPS * np.cos(rad))
                ts = (t0 + pd.to_timedelta(eid * 100 + np.arange(nfr) / FPS, unit="s")).strftime("%Y-%m-%d %H:%M:%S.%f")
                ct.append(pd.DataFrame(dict(draft_year=2095, event_id=eid, nfl_id=p.nflId, entity_type="player",
                                            time=ts, drill_type="speed" if drill.startswith("40") else "agility",
                                            drill_name=drill, attempt=att, x=x, y=y, s=s,
                                            a=np.abs(np.gradient(s)) * FPS, dis=s / FPS, dir=h)))
    combine_tracking = pd.concat(ct, ignore_index=True)

    return dict(game_tracking=game_tracking, games=games, player_play=player_play, players=players_out,
                combine_results=combine_results, combine_tracking=combine_tracking, snaps=snaps,
                injuries=injuries, id_map=id_map, schedule=sched, truth=dict(onset_week=onset))
