"""
Synthetic data for the cross-position notebook, built from the two existing generators
(synthetic_db.py and synthetic_ol.py) so each position family has planted injuries to find.

- db:        synthetic_db as is (CB/SS/FS with planted pre-injury signals)
- receivers: synthetic_db with its DBs relabeled as receivers (CB/FS -> WR, SS -> TE)
- ol:        synthetic_ol as is
- front7:    synthetic_ol with linemen relabeled as defensive front (T -> DE, G -> DT, C -> NT)

Each call returns the same dict shape as the generators (repo schema + nflverse tables).
Nothing here is real data.
"""
import synthetic_db
import synthetic_ol


def _relabel(D, mapping, snap_from, snap_to):
    D = dict(D)
    players = D["players"].copy()
    keep = players.nfl_position.isin(mapping)
    players.loc[~keep, "nfl_position"] = "XX"              # everyone else drops out of the family
    players.loc[keep, "nfl_position"] = players.loc[keep, "nfl_position"].map(mapping)
    D["players"] = players
    snaps = D["snaps"].copy()
    snaps[snap_to] = snaps[snap_from]
    snaps[snap_to.replace("_pct", "_snaps")] = snaps[snap_from.replace("_pct", "_snaps")]
    D["snaps"] = snaps
    idm = D["id_map"].copy()
    idm["position"] = idm.position.map(mapping).fillna("XX")
    D["id_map"] = idm
    return D


def make_family(key):
    if key == "db":
        return synthetic_db.make_all(seed=7)
    if key == "receivers":
        return _relabel(synthetic_db.make_all(seed=21), {"CB": "WR", "FS": "WR", "SS": "TE"},
                        "defense_pct", "offense_pct")
    if key == "ol":
        return synthetic_ol.make_all(seed=11)
    if key == "front7":
        return _relabel(synthetic_ol.make_all(seed=31), {"T": "DE", "G": "DT", "C": "NT"},
                        "offense_pct", "defense_pct")
    raise ValueError(key)
