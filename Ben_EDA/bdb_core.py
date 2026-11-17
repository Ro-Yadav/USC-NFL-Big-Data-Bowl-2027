"""
bdb_core.py: shared code for the BDB 2027 project (Ben_EDA/).

Everything the analysis notebooks used to copy into themselves lives here once:
data loading (repo data + nflverse), tracking measures, the snap-based injury definition,
matched controls, Combine drill measures, and rookie outcomes. Rep-level results are cached
in outputs/cache/ so tracking is processed once.

Usage (from a notebook in Ben_EDA/):
    import bdb_core as bc
    P = bc.Project()                 # finds ../data, or falls back to synthetic data
    db = P.family("db")              # injuries, onsets, game-level measures for DBs
    rk = P.rookies()                 # rookie outcomes + Combine traits for all 510 players
"""
from pathlib import Path
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
if str(HERE / "synthetic") not in sys.path:
    sys.path.insert(0, str(HERE / "synthetic"))

# ------------------------------------------------------------------ settings
SEASONS = [2023, 2024, 2025]
SNAP_EVENTS = {"ball_snap", "snap_direct", "autoevent_ballsnap"}
GROUP = {"CB": "DB", "FS": "DB", "SS": "DB", "S": "DB", "DB": "DB", "WR": "Receivers", "TE": "Receivers",
         "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "DE": "Edge", "OLB": "Edge",
         "DT": "Interior DL", "NT": "Interior DL"}
STANDARD_DRILLS = "FORTY|THREE_CONE|3_CONE|SHORT_SHUTTLE|SHUTTLE|BROAD|VERTICAL|BENCH"
SOFT_TISSUE = "hamstring|groin|calf|quad|thigh|adductor|abdom|oblique"
EXCLUDE_INJURY = "concussion|head|illness|personal|rest|not injury|non-football"

MIN_BASELINE_GAMES, DROP_RATIO, MIN_OUT_GAMES = 3, 0.50, 1
MIN_SPEED_FOR_DIR, HARD_CUT_DEG_S, RETENTION_SEC, FIRST_STEP_SEC, ACC_CLIP = 2.0, 150, 0.5, 0.8, 12.0
GETOFF_YDS, GETOFF_MAX_SEC, REP_SEC = 1.0, 1.5, 3.0
MIN_PLAYS_PER_GAME, PRE_WINDOW, MIN_BASE_GAMES_TRK, N_CONTROLS = 5, 2, 2, 4
LOAD_THR, DROP_THR, SPRINT_YDS, IR_MIN_GAMES, MIN_GAMES = 1.10, 0.05, 8.0, 3, 4
MEAS = ["first_step", "time_1yd", "max_decel", "max_turn", "retention"]
MEAS_NAME = {"first_step": "first step", "time_1yd": "time to 1 yard", "max_decel": "braking",
             "max_turn": "turning", "retention": "speed kept"}

# family -> positions, snap column, baseline snap share, measure set, cache key
FAMILIES = {
    "db":        dict(label="DB", positions={"CB", "SS", "FS", "S", "DB"}, snap_col="defense_pct", min_pct=0.60,
                      kind="cutting", cache="db", subgroup=lambda p: "CB" if p == "CB" else "S"),
    "receivers": dict(label="Receivers", positions={"WR", "TE"}, snap_col="offense_pct", min_pct=0.60,
                      kind="cutting", cache="receivers", subgroup=lambda p: "TE" if p == "TE" else "WR"),
    "ol":        dict(label="OL", positions={"T", "G", "C", "OT", "OG"}, snap_col="offense_pct", min_pct=0.60,
                      kind="burst", cache="ol", subgroup=lambda p: "T" if p in {"T", "OT"} else "IOL"),
    "edge":      dict(label="Edge", positions={"DE", "OLB"}, snap_col="defense_pct", min_pct=0.40,
                      kind="burst", cache="front7", subgroup=lambda p: "EDGE"),
    "idl":       dict(label="Interior DL", positions={"DT", "NT"}, snap_col="defense_pct", min_pct=0.40,
                      kind="burst", cache="front7", subgroup=lambda p: "IDL"),
}
# load and response measures for the load-then-decline signature, by measure set
SIGNATURE = {"cutting": ("cuts_per_play", "turn_rate"), "burst": ("effort_per_rep", "first_step")}

IDS = {"game_id": "gameId", "play_id": "playId", "nfl_id": "nflId"}
GAME_KEYS = ["gameId", "playId", "nflId"]
COMBINE_KEYS = ["nfl_id", "event_id", "drill_name", "attempt"]


# ------------------------------------------------------------------ statistics
def cliffs_delta(a, b):
    a = np.asarray(pd.to_numeric(pd.Series(a), errors="coerce").dropna(), float)
    b = np.asarray(pd.to_numeric(pd.Series(b), errors="coerce").dropna(), float)
    return (np.sign(a[:, None] - b[None, :])).mean() if len(a) and len(b) else np.nan


def mwu(a, b, alternative="two-sided"):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    return stats.mannwhitneyu(a, b, alternative=alternative).pvalue if len(a) > 1 and len(b) > 1 else np.nan


def spearman_brown(r, k):
    """Reliability of the average of k measurements whose single-measurement reliability is r."""
    return k * r / (1 + (k - 1) * r)


def reps_for(r, target=0.8):
    return target * (1 - r) / (r * (1 - target)) if r is not None and np.isfinite(r) and r > 0 else np.inf


def n_to_detect_r(r, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(((za + zb) / np.arctanh(abs(r))) ** 2 + 3)) if np.isfinite(r) and abs(r) >= 0.03 else np.nan


# ------------------------------------------------------------------ tracking measures
def prep_tracking(trk, keys, snap_events=None):
    trk = trk.copy()
    trk["time"] = pd.to_datetime(trk.time, format="mixed")
    trk = trk.sort_values(keys + ["time"])
    if snap_events is not None and "event" in trk.columns:
        snap = trk[trk.event.isin(snap_events)].groupby(keys[:-1]).time.min().rename("snap_time")
        trk = trk.join(snap, on=keys[:-1])
        trk["has_snap"] = trk.snap_time.notna()
        trk = trk[~trk.has_snap | (trk.time >= trk.snap_time)].drop(columns="snap_time")
    trk["frameId"] = trk.groupby(keys).cumcount() + 1
    fps = float(round(1 / trk.groupby(keys).time.diff().dt.total_seconds().median()))
    return trk.reset_index(drop=True), fps


def cutting_reps(trk, fps):
    """Per rep (DB-notebook definitions): peak speed, accel/decel, turn rate, hard cuts, speed kept through cuts."""
    trk = trk.sort_values(GAME_KEYS + ["frameId"]).copy()
    g = trk.groupby(GAME_KEYS, sort=False)
    w = max(1, int(round(RETENTION_SEC * fps)))
    trk["long_acc"] = g.s.diff() * fps
    trk["long_acc"] = trk.groupby(GAME_KEYS).long_acc.transform(
        lambda x: x.rolling(3, center=True, min_periods=1).mean()).clip(-ACC_CLIP, ACC_CLIP)
    unwrapped = g.dir.transform(lambda d: np.rad2deg(np.unwrap(np.deg2rad(d.to_numpy()))))
    trk["turn_rate"] = unwrapped.groupby([trk[k] for k in GAME_KEYS]).diff() * fps
    trk.loc[trk.s < MIN_SPEED_FOR_DIR, "turn_rate"] = np.nan
    trk["abs_turn"] = trk.groupby(GAME_KEYS).turn_rate.transform(
        lambda x: x.abs().rolling(3, center=True, min_periods=1).median())
    local_max = trk.groupby(GAME_KEYS).abs_turn.transform(lambda x: x.rolling(5, center=True, min_periods=1).max())
    trk["is_cut"] = (trk.abs_turn >= HARD_CUT_DEG_S) & (trk.abs_turn == local_max)
    s_b, s_a = g.s.shift(w), g.s.shift(-w)
    trk["retention"] = np.where(trk.is_cut & (s_b > MIN_SPEED_FOR_DIR), s_a / s_b, np.nan)
    rep = (trk.groupby(GAME_KEYS)
              .agg(max_s=("s", "max"), max_acc=("long_acc", "max"), max_decel=("long_acc", "min"),
                   max_turn=("abs_turn", "max"), retention=("retention", "mean"), n_cuts=("is_cut", "sum"))
              .reset_index())
    rep["max_decel"] = -rep.max_decel
    return rep


def burst_reps(trk, plays, fps):
    """Per rep (OL-notebook definitions): get-off (interpolated), first step, stop power, top speed, peak accel."""
    trk = (trk[trk.has_snap].sort_values(GAME_KEYS + ["frameId"])
             .merge(plays[["gameId", "playId", "play_direction"]], on=["gameId", "playId"], how="left"))
    g = trk.groupby(GAME_KEYS, sort=False)
    trk["t"] = (trk.frameId - 1) / fps
    x0, y0 = g.x.transform("first"), g.y.transform("first")
    trk["disp"] = np.hypot(trk.x - x0, trk.y - y0)
    trk["long_acc"] = g.s.diff() * fps
    trk["long_acc"] = trk.groupby(GAME_KEYS).long_acc.transform(
        lambda v: v.rolling(3, center=True, min_periods=1).mean()).clip(-ACC_CLIP, ACC_CLIP)
    d_prev, t_prev = trk.groupby(GAME_KEYS).disp.shift(), trk.groupby(GAME_KEYS).t.shift()
    cross = (trk.disp >= GETOFF_YDS) & (d_prev < GETOFF_YDS) & (trk.t <= GETOFF_MAX_SEC)
    trk["go_t"] = np.where(cross, t_prev + (GETOFF_YDS - d_prev) / (trk.disp - d_prev) * (trk.t - t_prev), np.nan)
    trk["fs_acc"] = np.where(trk.t <= FIRST_STEP_SEC, trk.long_acc, np.nan)
    trk["acc_w"] = np.where(trk.t <= REP_SEC, trk.long_acc, np.nan)
    trk["s_w"] = np.where(trk.t <= REP_SEC, trk.s, np.nan)
    rep = (trk.groupby(GAME_KEYS)
              .agg(get_off=("go_t", "min"), first_step=("fs_acc", "max"), peak_acc=("acc_w", "max"),
                   stop_power=("acc_w", "min"), top_speed=("s_w", "max"))
              .reset_index())
    rep["stop_power"] = -rep.stop_power
    return rep


def game_level(rep, kind):
    """Game-level measures from reps. Cutting and burst sets both get top_speed, a load and a response measure."""
    q90, q75 = (lambda x: x.quantile(0.9)), (lambda x: x.quantile(0.75))
    if kind == "cutting":
        out = (rep.assign(sprint=rep.max_s >= SPRINT_YDS).groupby(["nflId", "gameId"])
                  .agg(top_speed=("max_s", q90), decel=("max_decel", q90), turn_rate=("max_turn", q90),
                       retention=("retention", "median"), cuts_per_play=("n_cuts", "mean"),
                       sprint_share=("sprint", "mean"), n_plays=("playId", "nunique")))
    else:
        out = (rep.groupby(["nflId", "gameId"])
                  .agg(get_off=("get_off", "median"), first_step=("first_step", q75), stop_power=("stop_power", q90),
                       top_speed=("top_speed", q90), effort_per_rep=("peak_acc", "median"),
                       n_plays=("playId", "nunique")))
    return out.reset_index()


def combine_attempt_features(ct):
    ct, fps = prep_tracking(ct, COMBINE_KEYS)
    ct["start_t"] = ct.time.where(ct.s >= 0.3).groupby([ct[k] for k in COMBINE_KEYS]).transform("min")
    ct["t"] = (ct.time - ct.start_t).dt.total_seconds()
    ct = ct[ct.t >= 0].copy()
    g = ct.groupby(COMBINE_KEYS, sort=False)
    x0, y0 = g.x.transform("first"), g.y.transform("first")
    ct["disp"] = np.hypot(ct.x - x0, ct.y - y0)
    ct["long_acc"] = g.s.diff() * fps
    ct["long_acc"] = ct.groupby(COMBINE_KEYS).long_acc.transform(
        lambda v: v.rolling(3, center=True, min_periods=1).mean()).clip(-ACC_CLIP, ACC_CLIP)
    unwrapped = g.dir.transform(lambda d: np.rad2deg(np.unwrap(np.deg2rad(d.to_numpy()))))
    ct["turn"] = unwrapped.groupby([ct[k] for k in COMBINE_KEYS]).diff().abs() * fps
    ct.loc[ct.s < MIN_SPEED_FOR_DIR, "turn"] = np.nan
    w = max(1, int(round(RETENTION_SEC * fps)))
    is_cut = (ct.turn >= HARD_CUT_DEG_S) & (ct.turn == ct.groupby(COMBINE_KEYS).turn.transform(
        lambda v: v.rolling(5, center=True, min_periods=1).max()))
    s_b, s_a = g.s.shift(w), g.s.shift(-w)
    ct["retention"] = np.where(is_cut & (s_b > MIN_SPEED_FOR_DIR), s_a / s_b, np.nan)
    ct["fs"] = np.where(ct.t <= FIRST_STEP_SEC, ct.long_acc, np.nan)
    for d, c in [(1, "t1"), (10, "t10"), (40, "t40")]:
        ct[c] = np.where(ct.disp >= d, ct.t, np.nan)
    out = (ct.groupby(COMBINE_KEYS)
             .agg(first_step=("fs", "max"), time_1yd=("t1", "min"), max_decel=("long_acc", "min"),
                  max_turn=("turn", "max"), retention=("retention", "mean"), time_10yd=("t10", "min"),
                  time_40yd=("t40", "min"))
             .reset_index())
    out["max_decel"] = -out.max_decel
    return out


# ------------------------------------------------------------------ injury definition
def build_snap_grid(snaps, schedule, pct_col):
    s = snaps[snaps.game_type == "REG"].copy()
    sch = schedule[schedule.game_type == "REG"]
    team_games = pd.concat([sch[["week", "home_team"]].rename(columns={"home_team": "team"}),
                            sch[["week", "away_team"]].rename(columns={"away_team": "team"})])
    weeks = np.sort(sch.week.unique())
    grid = (pd.MultiIndex.from_product([s.pfr_player_id.unique(), weeks], names=["pfr_player_id", "week"])
              .to_frame(index=False)
              .merge(s[["pfr_player_id", "week", "team", pct_col]], how="left", on=["pfr_player_id", "week"]))
    grid = grid.sort_values(["pfr_player_id", "week"])
    grid["team"] = grid.groupby("pfr_player_id").team.transform(lambda x: x.ffill().bfill())
    grid = grid.merge(team_games, on=["week", "team"])
    grid["pct"] = grid[pct_col].fillna(0.0)
    return grid.sort_values(["pfr_player_id", "week"]).reset_index(drop=True)


def find_onset(pct, weeks, base_min):
    import ruptures as rpt
    if len(pct) < MIN_BASELINE_GAMES + 1:
        return None
    try:
        bkps = rpt.Pelt(model="l2", min_size=1, jump=1).fit(pct.reshape(-1, 1)).predict(pen=0.15)
    except Exception:
        bkps = [len(pct)]
    for i in sorted(set(b for b in bkps[:-1]) | set(b - 1 for b in bkps[:-1])):
        if i < MIN_BASELINE_GAMES:
            continue
        base = np.median(pct[i - MIN_BASELINE_GAMES:i])
        if base < base_min:
            continue
        if pct[i] < DROP_RATIO * base and pct[i:i + 1 + MIN_OUT_GAMES].mean() < DROP_RATIO * base:
            low = pct[i:] < DROP_RATIO * base
            return int(weeks[i]), float(base), int(np.argmin(low)) if not low.all() else int(low.sum())
    return None


def rookie_outcome(players, injuries, snaps, schedule, idmap):
    """Rookie-season games missed with an injury (weeks Out + 3+ game absences after a listing), snaps, games."""
    inj = injuries[injuries.game_type == "REG"].merge(idmap[["gsis_id", "nfl_id"]], on="gsis_id")
    inj = inj.merge(players[["nfl_id", "draft_year"]], on="nfl_id")
    inj = inj[(inj.season == inj.draft_year) & inj.report_primary_injury.notna()
              & ~inj.report_primary_injury.astype(str).str.lower().str.contains(EXCLUDE_INJURY)]
    out_weeks = inj[inj.report_status == "Out"].groupby("nfl_id").week.apply(set).to_dict()
    listed = inj.groupby("nfl_id").week.apply(set).to_dict()
    soft = set(inj[inj.report_primary_injury.astype(str).str.lower().str.contains(SOFT_TISSUE)].nfl_id)
    sn = snaps[snaps.game_type == "REG"].merge(idmap[["pfr_id", "nfl_id"]], left_on="pfr_player_id", right_on="pfr_id")
    sn = sn.merge(players[["nfl_id", "draft_year"]], on="nfl_id")
    sn = sn[sn.season == sn.draft_year]
    sch = schedule[schedule.game_type == "REG"]
    team_weeks = (pd.concat([sch[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
                             sch[["season", "week", "away_team"]].rename(columns={"away_team": "team"})])
                    .groupby(["season", "team"]).week.apply(lambda w: sorted(set(w))).to_dict())
    by_player = {k: g for k, g in sn.groupby("nfl_id")}
    rows = []
    for nid in players.nfl_id:
        weeks, ir = set(out_weeks.get(nid, set())), set()
        g = by_player.get(nid)
        if g is not None and nid in listed:
            played, run = set(g.week), []
            for w in team_weeks.get((g.season.iloc[0], g.team.mode().iloc[0]), []) + [None]:
                if w is not None and w not in played and w > min(played):
                    run.append(w); continue
                if len(run) >= IR_MIN_GAMES and any(run[0] - 2 <= lw <= run[0] for lw in listed[nid]):
                    ir |= set(run)
                run = []
        rows.append(dict(nfl_id=nid, games_out=len(weeks | ir), games_played=0 if g is None else len(g),
                         snaps=0 if g is None else int((g.offense_snaps + g.defense_snaps).sum()),
                         soft_tissue=nid in soft))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ project / data access
class Project:
    """Loads data once, builds or reuses caches, and hands out per-family and rookie tables."""

    def __init__(self, data_dir="../data", ext_dir="external", out_dir="outputs", synthetic=None):
        self.data, self.ext, self.out = Path(data_dir), Path(ext_dir), Path(out_dir)
        self.cache = self.out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.synthetic = (not (self.data / "games.csv").exists()) if synthetic is None else synthetic
        self._fam, self._rk, self._att, self._shared, self._syn = {}, None, None, None, {}
        print("Synthetic demo mode" if self.synthetic else f"Real data from {self.data.resolve()}")

    # ---------- shared tables
    def _nflverse(self, seasons):
        import nflreadpy as nfl
        return dict(snaps=nfl.load_snap_counts(seasons).to_pandas(),
                    injuries=nfl.load_injuries(seasons).to_pandas(),
                    id_map=nfl.load_players().to_pandas()[["gsis_id", "nfl_id", "pfr_id", "display_name", "position"]],
                    schedule=nfl.load_schedules(seasons).to_pandas())

    @staticmethod
    def _tidy(raw, nv):
        T = {}
        games = raw["games"].rename(columns=IDS)
        T["games"] = games[games.season_type.astype(str).str.upper().str.startswith("REG")]
        T["players"] = raw["players"].rename(columns=IDS | {"nfl_position": "position", "display_name": "displayName"})
        pp = raw["player_play"].rename(columns=IDS) if "player_play" in raw else pd.DataFrame(columns=["gameId", "playId"])
        if "play_direction" not in pp.columns:
            pp["play_direction"] = np.nan
        T["plays"] = pp.drop_duplicates(["gameId", "playId"])[["gameId", "playId", "play_direction"]]
        T["snaps"], T["injuries"], T["schedule"] = nv["snaps"], nv["injuries"], nv["schedule"]
        idm = nv["id_map"].dropna(subset=["nfl_id"]).copy()
        idm["nflId"] = pd.to_numeric(idm.nfl_id, errors="coerce")
        T["id_map"] = idm.dropna(subset=["nflId"]).drop_duplicates("nflId")
        T["seasons"] = sorted(T["games"].season.unique().tolist())
        return T

    def shared(self):
        if self._shared is None:
            raw = {k: pd.read_csv(self.data / f"{k}.csv") for k in ["games", "players"]}
            raw["player_play"] = pd.read_csv(self.data / "player_play.csv",
                                             usecols=lambda c: c in {"game_id", "play_id", "play_direction"})
            self._nv = self._nflverse(SEASONS)
            self._shared = self._tidy(raw, self._nv)
        return self._shared

    def _synthetic_family(self, cache_key):
        if cache_key not in self._syn:
            from synthetic_cross import make_family
            self._syn[cache_key] = make_family(cache_key)
        return self._syn[cache_key]

    # ---------- game reps (cached)
    def reps(self, key):
        spec = FAMILIES[key]
        need = "max_s" if spec["kind"] == "cutting" else "get_off"
        name = f"{spec['cache']}_{spec['kind']}_reps.parquet"
        if not self.synthetic:
            for p in [self.cache / name, self.out / "cross_position" / f"{spec['cache']}_reps.parquet"]:
                if p.exists():
                    r = pd.read_parquet(p)
                    if need in r.columns:
                        if p != self.cache / name:            # copy older caches into outputs/cache/
                            r.to_parquet(self.cache / name)
                        return r
        if self.synthetic:
            raw = self._synthetic_family(spec["cache"]); T = self._tidy(raw, raw)
            trk = raw["game_tracking"].rename(columns=IDS)
        else:
            T = self.shared()
            ids = set(T["players"].loc[T["players"].position.isin(
                set().union(*[f["positions"] for f in FAMILIES.values() if f["cache"] == spec["cache"]])), "nflId"])
            cols = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "dir", "event"]
            parts = [c[c.nfl_id.isin(ids)] for f in sorted(glob.glob(str(self.data / "game_tracking_*.csv")))
                     for c in pd.read_csv(f, usecols=cols, chunksize=5_000_000)]
            trk = pd.concat(parts, ignore_index=True).rename(columns=IDS)
        trk = trk[trk.gameId.isin(T["games"].gameId)]
        trk, fps = prep_tracking(trk, GAME_KEYS, SNAP_EVENTS)
        r = cutting_reps(trk, fps) if spec["kind"] == "cutting" else burst_reps(trk, T["plays"], fps)
        if not self.synthetic:
            r.to_parquet(self.cache / name)
        return r

    # ---------- one family: injuries, onsets, game-level measures
    def family(self, key):
        if key in self._fam:
            return self._fam[key]
        spec = FAMILIES[key]
        if self.synthetic:
            raw = self._synthetic_family(spec["cache"]); T = self._tidy(raw, raw)
        else:
            T = self.shared()
        fam_ids = set(T["players"].loc[T["players"].position.isin(spec["positions"]), "nflId"])
        pfr = set(T["id_map"].loc[T["id_map"].nflId.isin(fam_ids), "pfr_id"].dropna())
        snaps = T["snaps"][T["snaps"].pfr_player_id.isin(pfr)]
        grid = pd.concat([build_snap_grid(snaps[snaps.season == yr], T["schedule"][T["schedule"].season == yr],
                                          spec["snap_col"]).assign(season=yr) for yr in T["seasons"]], ignore_index=True)
        on = []
        for (yr, pid), g in grid.groupby(["season", "pfr_player_id"]):
            r = find_onset(g.pct.to_numpy(), g.week.to_numpy(), spec["min_pct"])
            if r:
                on.append(dict(season=yr, pfr_player_id=pid, onset_week=r[0], baseline_pct=r[1], games_low=r[2]))
        onsets = pd.DataFrame(on, columns=["season", "pfr_player_id", "onset_week", "baseline_pct", "games_low"])
        onsets = onsets.merge(T["id_map"][["pfr_id", "gsis_id", "nflId"]], left_on="pfr_player_id", right_on="pfr_id", how="left")
        rep = T["injuries"][T["injuries"].report_status.isin(["Out", "Doubtful", "Questionable"])]
        def match(row):
            m = rep[(rep.gsis_id == row.gsis_id) & (rep.season == row.season) & rep.week.between(row.onset_week, row.onset_week + 1)]
            return m.report_primary_injury.iloc[0] if len(m) else None
        onsets["reported_injury"] = onsets.apply(match, axis=1) if len(onsets) else []
        txt = onsets.reported_injury.astype(str).str.lower()
        onsets["confirmed"] = onsets.reported_injury.notna() & ~txt.str.contains("concussion|head")
        onsets["soft_tissue"] = onsets.confirmed & txt.str.contains(SOFT_TISSUE)
        tracked = T["games"].groupby("season").week.agg(["min", "max"])
        inj = onsets[onsets.confirmed & onsets.nflId.isin(fam_ids)].copy()
        inj = inj[[s in tracked.index and tracked.loc[s, "min"] < w <= tracked.loc[s, "max"] + 1
                   for s, w in zip(inj.season, inj.onset_week)]]
        inj["nflId"] = inj.nflId.astype(int)
        pos_of = T["players"].set_index("nflId").position
        inj["subgroup"] = inj.nflId.map(pos_of).map(spec["subgroup"])
        reps = self.reps(key)
        reps = reps[reps.nflId.isin(fam_ids)]
        gl = game_level(reps, spec["kind"]).merge(T["games"][["gameId", "season", "week"]], on="gameId")
        gl = gl[gl.n_plays >= MIN_PLAYS_PER_GAME]
        base_usage = (grid.merge(T["id_map"][["pfr_id", "nflId"]], left_on="pfr_player_id", right_on="pfr_id")
                          .groupby(["season", "nflId"]).pct.median())
        self._fam[key] = dict(key=key, T=T, spec=spec, fam_ids=fam_ids, onsets=onsets, inj=inj, grid=grid,
                              gl=gl, reps=reps, base_usage=base_usage)
        return self._fam[key]

    # ---------- Combine attempts (cached) and rookie table
    def combine_attempts(self):
        if self._att is not None:
            return self._att
        p = self.cache / "combine_attempts_v2.parquet"
        if not self.synthetic and p.exists():
            att = pd.read_parquet(p)
        else:
            if self.synthetic:
                from synthetic_rookie import make_all
                self._rk_raw = make_all()
                ct = self._rk_raw["combine_tracking"]
            else:
                ct = pd.read_csv(self.data / "combine_tracking.csv")
            ct = ct.dropna(subset=["nfl_id"]).copy()
            ct["drill_name"] = ct.drill_name.astype(str).str.upper().str.replace("-", "_", regex=False)
            att = combine_attempt_features(ct)
            if not self.synthetic:
                att.to_parquet(p)
        pl = self._rookie_raw()["players"]
        att = att.merge(pl[["nfl_id", "nfl_position"]], on="nfl_id", how="left")
        att["group"] = att.nfl_position.map(GROUP).fillna("Other")
        att["standard_drill"] = att.drill_name.str.contains(STANDARD_DRILLS)
        self._att = att
        return att

    def _rookie_raw(self):
        if getattr(self, "_rk_raw", None) is None:
            if self.synthetic:
                from synthetic_rookie import make_all
                self._rk_raw = make_all()
            else:
                self.shared()
                self._rk_raw = dict(players=pd.read_csv(self.data / "players.csv"),
                                    combine_results=pd.read_csv(self.data / "combine_results.csv"),
                                    career=pd.read_csv(self.data / "player_career_successes.csv"),
                                    ext_players=self._nv["id_map"], injuries=self._nv["injuries"],
                                    snaps=self._nv["snaps"], schedule=self._nv["schedule"])
        return self._rk_raw

    def rookies(self):
        """One row per drafted player: rookie injury outcome, exposure, Combine results, Combine movement traits."""
        if self._rk is not None:
            return self._rk
        R = self._rookie_raw()
        idm = R["ext_players"].dropna(subset=["nfl_id"]).copy()
        idm["nfl_id"] = pd.to_numeric(idm.nfl_id, errors="coerce")
        idm = idm.dropna(subset=["nfl_id"]); idm["nfl_id"] = idm.nfl_id.astype(int)
        rk = R["players"].copy()
        rk["group"] = rk.nfl_position.map(GROUP).fillna("Other")
        rk["draft_round"] = pd.to_numeric(rk.get("draft_round"), errors="coerce").fillna(8)
        rk = rk.merge(rookie_outcome(rk, R["injuries"], R["snaps"], R["schedule"], idm), on="nfl_id")
        rk["injured"] = rk.games_out > 0
        att = self.combine_attempts()
        z = att.copy()
        z[MEAS] = z.groupby("drill_name")[MEAS].transform(lambda x: (x - x.mean()) / x.std())
        first = z[z.attempt == z.groupby(["nfl_id", "drill_name"]).attempt.transform("min")]
        comp = first.groupby("nfl_id")[["retention", "max_decel", "max_turn", "first_step"]].mean().rename(
            columns={"retention": "speed_kept", "max_decel": "braking", "max_turn": "turning"})
        rk = rk.merge(comp.reset_index(), on="nfl_id", how="left")
        timed = [c for c in ["combine_weight", "forty", "ten_yd_split", "vertical", "broad_jump", "three_cone",
                             "short_shuttle", "bench_reps"] if c in R["combine_results"].columns]
        rk = rk.merge(R["combine_results"].drop_duplicates("nfl_id", keep="last")[["nfl_id"] + timed], on="nfl_id", how="left")
        if "career" in R:
            rk = rk.merge(R["career"], on="nfl_id", how="left")
        zc = timed + ["speed_kept", "braking", "turning", "first_step"]
        rk[zc] = rk.groupby("group")[zc].transform(lambda x: (x - x.mean()) / x.std())
        self.timed = timed
        self._first_z = first
        self._rk = rk[rk.group != "Other"].copy()
        return self._rk


# ------------------------------------------------------------------ controls and signatures
def match_controls(F, inj=None, seed=0):
    T, gl, spec = F["T"], F["gl"], F["spec"]
    inj = F["inj"] if inj is None else inj
    pos_of = T["players"].set_index("nflId").position
    ever = {(r.season, int(r.nflId)) for r in F["onsets"].dropna(subset=["nflId"]).itertuples()}
    weeks = gl.groupby(["season", "nflId"]).week.apply(np.array).to_dict()
    r_ = np.random.default_rng(seed)
    rows = []
    for r in inj.itertuples():
        pool = [p for p in F["fam_ids"]
                if (r.season, p) not in ever and spec["subgroup"](pos_of.get(p)) == r.subgroup
                and abs(F["base_usage"].get((r.season, p), -1) - r.baseline_pct) <= 0.15
                and (weeks.get((r.season, p), np.array([])) < r.onset_week).sum() >= MIN_BASE_GAMES_TRK + PRE_WINDOW]
        for c in r_.choice(pool, min(N_CONTROLS, len(pool)), replace=False):
            rows.append(dict(nflId=int(c), season=r.season, onset_week=r.onset_week, match_of=r.nflId))
    controls = pd.DataFrame(rows, columns=["nflId", "season", "onset_week", "match_of"])
    return pd.concat([inj[["nflId", "season", "onset_week"]].assign(group="injured", match_of=inj.nflId),
                      controls.assign(group="control")], ignore_index=True)


def signature_at(gl, nid, season, onset, load_col, resp_col):
    g = gl[(gl.nflId == nid) & (gl.season == season) & (gl.week < onset)].sort_values("week")
    if len(g) < MIN_BASE_GAMES_TRK + PRE_WINDOW:
        return None
    base = g.iloc[:-PRE_WINDOW][[load_col, resp_col]].median()
    m2, m1 = g.iloc[-2], g.iloc[-1]
    if not (base[load_col] > 0 and base[resp_col] > 0):
        return None
    return bool(m2[load_col] / base[load_col] > LOAD_THR and (m2[resp_col] - m1[resp_col]) / base[resp_col] > DROP_THR)
