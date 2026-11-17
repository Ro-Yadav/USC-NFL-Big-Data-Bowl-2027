"""
Synthetic stand-in for Big Data Bowl tracking + nflverse tables.

Shapes match what the notebook's real-data loaders return, so every section can
be developed and tested before the real BDB files are in place. A handful of
DBs get a planted "pre-injury" signal (lower top speed, worse speed retention
through cuts, slower mirror reaction, steeper in-game fade) so you can confirm
the pipeline recovers it. Nothing here is real data.
"""
import numpy as np
import pandas as pd

FPS = 10


def _schedule(teams, n_weeks, rng):
    rows = []
    for wk in range(1, n_weeks + 1):
        order = rng.permutation(teams)
        for i in range(0, len(order), 2):
            home, away = order[i], order[i + 1]
            rows.append(dict(game_id=f"S_{wk:02d}_{away}_{home}", season=2099,
                             game_type="REG", week=wk, home_team=home, away_team=away))
    return pd.DataFrame(rows)


def _path(n, start, heading, speed, cuts, rng, lag_frames=0, fade=1.0):
    """Simple 2D path: constant-ish speed with sharp direction changes at `cuts`.
    `fade` < 1 lowers speed retained after each cut."""
    x, y = np.empty(n), np.empty(n)
    s_arr, d_arr = np.empty(n), np.empty(n)
    pos = np.array(start, float)
    h, s = heading, 0.0
    # spread each break over 4 frames (0.4 s), like a real plant-and-turn
    turn = {}
    for c, ang in cuts:
        for j in range(4):
            turn[c + lag_frames + j] = turn.get(c + lag_frames + j, 0) + ang / 4
    for t in range(n):
        if t in turn:
            h = (h + turn[t]) % 360
            s *= 0.86 * fade ** 0.25              # speed lost through the break
        s = min(speed, s + 0.9 * (speed - s) * 0.35 + 0.15)   # re-accelerate
        s = max(s + rng.normal(0, 0.05), 0)
        rad = np.deg2rad(h)
        # NGS dir convention: 0 = +y, clockwise
        pos = pos + s / FPS * np.array([np.sin(rad), np.cos(rad)])
        x[t], y[t], s_arr[t], d_arr[t] = pos[0], pos[1], s, h + rng.normal(0, 2)
    return x, y, s_arr, d_arr % 360


def make_all(seed=7, n_teams=8, track_weeks=9, season_weeks=18, plays_per_game=50,
             frames=45):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    sched = _schedule(teams, season_weeks, rng)

    # ---- players -----------------------------------------------------------
    roster, pid = [], 40000
    for t in teams:
        for pos in ["CB", "CB", "CB", "SS", "FS", "WR", "WR", "TE"]:
            pid += 1
            roster.append(dict(nflId=pid, team=t, position=pos, displayName=f"P{pid}",
                               top_speed=rng.normal(9.6 if pos in ("CB", "WR") else 9.2, 0.3),
                               react=max(rng.normal(0.30, 0.06), 0.1),
                               retain=rng.normal(1.0, 0.05)))
    players = pd.DataFrame(roster)
    dbs = players[players.position.isin(["CB", "SS", "FS"])]

    # injured DBs: onset week inside the tracking window, plus one benched player
    inj_ids = rng.choice(dbs.nflId.values, 7, replace=False)
    onset = {p: int(rng.integers(5, track_weeks + 1)) for p in inj_ids[:6]}
    benched = int(inj_ids[6])
    injuries_list = ["Hamstring", "Groin", "Calf", "Ankle", "Knee", "Hamstring"]

    def team_week_game(team, wk):
        g = sched[(sched.week == wk) & ((sched.home_team == team) | (sched.away_team == team))]
        return g.iloc[0] if len(g) else None

    # ---- snap counts (nflverse shape) ----------------------------------------
    snap_rows = []
    for p in players.itertuples():
        starter = p.position != "CB" or rng.random() < 0.67
        base = rng.uniform(0.85, 1.0) if starter else rng.uniform(0.15, 0.45)
        for wk in range(1, season_weeks + 1):
            g = team_week_game(p.team, wk)
            if g is None:
                continue
            pct = np.clip(base + rng.normal(0, 0.05), 0, 1)
            if p.nflId in onset:
                ow = onset[p.nflId]
                if wk == ow:
                    pct = rng.uniform(0.15, 0.35)          # left the game
                elif ow < wk <= ow + 3:
                    continue                               # missed games (no row)
            if p.nflId == benched and wk >= 6:
                pct = rng.uniform(0.05, 0.2)
            snap_rows.append(dict(game_id=g.game_id, pfr_game_id=g.game_id, season=2099,
                                  game_type="REG", week=wk, player=p.displayName,
                                  pfr_player_id=f"PFR{p.nflId}", position=p.position,
                                  team=p.team, opponent="",
                                  offense_snaps=0 if p.position in ("CB", "SS", "FS") else int(pct * 65),
                                  offense_pct=0.0 if p.position in ("CB", "SS", "FS") else pct,
                                  defense_snaps=int(pct * 65) if p.position in ("CB", "SS", "FS") else 0,
                                  defense_pct=pct if p.position in ("CB", "SS", "FS") else 0.0,
                                  st_snaps=0, st_pct=0.0))
    snaps = pd.DataFrame(snap_rows)

    # every rostered DB is a regular contributor in this toy league so make starters exact
    # ---- injury report -------------------------------------------------------
    inj_rows = []
    for (p, ow), kind in zip(onset.items(), injuries_list):
        r = players.set_index("nflId").loc[p]
        for wk in range(ow + 1, ow + 4):
            inj_rows.append(dict(season=2099, game_type="REG", team=r.team, week=wk,
                                 gsis_id=f"GSIS{p}", position=r.position, full_name=r.displayName,
                                 first_name="", last_name="", report_primary_injury=kind,
                                 report_secondary_injury=None, report_status="Out",
                                 practice_primary_injury=kind, practice_secondary_injury=None,
                                 practice_status="Did Not Participate In Practice",
                                 date_modified=None))
    injuries = pd.DataFrame(inj_rows)

    id_map = pd.DataFrame(dict(gsis_id=[f"GSIS{i}" for i in players.nflId],
                               nfl_id=players.nflId.astype(str), pfr_id=[f"PFR{i}" for i in players.nflId],
                               display_name=players.displayName, position=players.position))

    combine = pd.DataFrame(dict(
        season=2095, pfr_id=[f"PFR{i}" for i in players.nflId], player_name=players.displayName,
        pos=players.position, wt=rng.normal(195, 10, len(players)).round(),
        forty=(4.42 + (9.6 - players.top_speed) * 0.12 + rng.normal(0, 0.04, len(players))).round(2),
        vertical=rng.normal(37, 2.5, len(players)).round(1),
        broad_jump=rng.normal(124, 5, len(players)).round(),
        cone=(6.9 + rng.normal(0, 0.15, len(players))).round(2),          # unrelated to in-game turning
        shuttle=(4.2 + rng.normal(0, 0.1, len(players))).round(2),
        bench=rng.normal(14, 3, len(players)).round()))

    # ---- BDB-style games / plays / players / tracking -------------------------
    tg = sched[sched.week <= track_weeks].reset_index(drop=True)
    games = pd.DataFrame(dict(gameId=np.arange(2099090001, 2099090001 + len(tg)), season=2099,
                              week=tg.week, homeTeamAbbr=tg.home_team, visitorTeamAbbr=tg.away_team))
    pinfo = players.set_index("nflId")
    play_rows, trk = [], []
    for g in games.itertuples():
        for play_id in range(1, plays_per_game + 1):
            off = g.homeTeamAbbr if play_id % 2 else g.visitorTeamAbbr
            dfn = g.visitorTeamAbbr if off == g.homeTeamAbbr else g.homeTeamAbbr
            quarter = min(4, 1 + (play_id - 1) * 4 // plays_per_game)
            play_rows.append(dict(gameId=g.gameId, playId=play_id, quarter=quarter, possessionTeam=off))
            receivers = players[(players.team == off) & players.position.isin(["WR", "TE"])]
            defenders = players[(players.team == dfn) & players.position.isin(["CB", "SS", "FS"])]
            rec_paths = {}
            for j, r in enumerate(receivers.itertuples()):
                cuts = [(int(rng.integers(12, 20)), rng.choice([-1, 1]) * rng.uniform(70, 140))]
                if rng.random() < 0.5:
                    cuts.append((int(rng.integers(26, 34)), rng.choice([-1, 1]) * rng.uniform(60, 120)))
                start = (40 + j * 8.0, 20)
                rec_paths[r.nflId] = (start, cuts)
                x, y, s, d = _path(frames, start, 0, r.top_speed, cuts, rng)
                trk.append(pd.DataFrame(dict(gameId=g.gameId, playId=play_id, nflId=r.nflId,
                                             club=off, frameId=np.arange(1, frames + 1), x=x, y=y,
                                             s=s, a=np.abs(np.gradient(s)) * FPS, dir=d)))
            rec_ids = list(rec_paths)
            for k, d_ in enumerate(defenders.itertuples()):
                pre = d_.nflId in onset and onset[d_.nflId] - 3 <= g.week < onset[d_.nflId]
                ramp = 0 if not pre else (g.week - (onset[d_.nflId] - 3) + 1) / 3   # worsens toward onset
                late = quarter >= 3
                spd = d_.top_speed * (1 - 0.06 * ramp - (0.04 * ramp if late else 0)) - (0.08 if late else 0)
                react = d_.react + 0.15 * ramp + (0.05 * ramp if late else 0)
                fade = d_.retain * (1 - 0.15 * ramp)
                if k < len(rec_ids):                                  # mirrors a receiver
                    start, cuts = rec_paths[rec_ids[k]]
                    start = (start[0] + rng.normal(0, 1), start[1] + 2.0)
                    x, y, s, d = _path(frames, start, 0, spd, cuts, rng,
                                       lag_frames=int(round(react * FPS)), fade=fade)
                else:                                                 # deep safety drift
                    x, y, s, d = _path(frames, (50, 35), 0, spd * 0.6,
                                       [(20, rng.uniform(-90, 90))], rng, fade=fade)
                trk.append(pd.DataFrame(dict(gameId=g.gameId, playId=play_id, nflId=d_.nflId,
                                             club=dfn, frameId=np.arange(1, frames + 1), x=x, y=y,
                                             s=s, a=np.abs(np.gradient(s)) * FPS, dir=d)))
    tracking = pd.concat(trk, ignore_index=True)
    tracking["event"] = np.where(tracking.frameId == 1, "ball_snap", None)
    tracking["frameType"] = np.where(tracking.frameId == 1, "SNAP", "AFTER_SNAP")
    plays = pd.DataFrame(play_rows)
    bdb_players = players[["nflId", "position", "displayName"]].copy()

    truth = dict(onset_week={int(k): v for k, v in onset.items()}, benched=benched)

    # ---- convert to the USC repo schema (snake_case, time instead of frameId) ----
    t0 = pd.Timestamp("2099-09-10")
    tracking["time"] = (t0 + pd.to_timedelta((tracking.gameId - games.gameId.min()) * 86400
                                             + tracking.playId * 60 + tracking.frameId / FPS, unit="s")
                        ).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    tracking["dis"] = tracking.s / FPS
    tracking["o"] = tracking["dir"]
    trk_out = tracking.rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id"})[
        ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "dis", "o", "dir", "event"]]
    games_out = games.rename(columns={"gameId": "game_id", "homeTeamAbbr": "home_team_abbr",
                                      "visitorTeamAbbr": "visitor_team_abbr"}).assign(
        game_key=0, season_type="REG")
    pp = (tracking[["gameId", "playId", "nflId", "club"]].drop_duplicates()
          .merge(plays, on=["gameId", "playId"])
          .merge(players[["nflId", "position"]], on="nflId"))
    player_play = pp.rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
                                     "club": "team_abbr", "possessionTeam": "possession_team",
                                     "position": "lined_up_position"})
    players_out = players.rename(columns={"nflId": "nfl_id", "displayName": "display_name",
                                          "position": "nfl_position"})[["nfl_id", "display_name", "nfl_position"]]
    combine_results = combine.assign(nfl_id=players.nflId.values).rename(
        columns={"cone": "three_cone", "shuttle": "short_shuttle", "wt": "combine_weight",
                 "pos": "combine_position", "bench": "bench_reps"}).drop(columns=["pfr_id", "player_name"])
    combine_results["draft_year"] = 2095
    combine_results["ten_yd_split"] = (combine_results.forty * 0.36).round(2)

    # combine drill tracking: agility drills reuse each player's turn/retention traits
    ct, eid = [], 0
    drills = {"3-Cone": [(10, 180), (22, -90), (30, -90)], "Short Shuttle": [(8, 180), (20, 180)],
              "40 Yard Dash": []}
    for p in players.itertuples():
        for drill, cuts in drills.items():
            for att in (1, 2):
                eid += 1
                n = 40 if drill != "40 Yard Dash" else 48
                spd = p.top_speed * (0.75 if cuts else 1.0)
                x, y, s_, d = _path(n, (0, 0), 90, spd, cuts, rng, fade=p.retain)
                ts = (t0 + pd.to_timedelta(eid * 100 + np.arange(n) / FPS, unit="s")).strftime("%Y-%m-%d %H:%M:%S.%f")
                ct.append(pd.DataFrame(dict(draft_year=2095, event_id=eid, nfl_id=p.nflId,
                                            entity_type="player", time=ts, drill_type="agility" if cuts else "speed",
                                            drill_name=drill, attempt=att, x=x, y=y, s=s_,
                                            a=np.abs(np.gradient(s_)) * FPS, dis=s_ / FPS, dir=d)))
    combine_tracking = pd.concat(ct, ignore_index=True)

    return dict(game_tracking=trk_out, games=games_out, player_play=player_play, players=players_out,
                combine_results=combine_results, combine_tracking=combine_tracking,
                snaps=snaps, injuries=injuries, id_map=id_map, schedule=sched, truth=truth)