"""Discrete-time colonisation model: probability that an unoccupied cell becomes occupied in a given year."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NEIGHBOURHOOD_KM = (50, 100)
RECENT_YEARS = 5
RECENT_RADIUS_KM = 100
FEATURES = ["log_distance_km", "neighbours_50km", "neighbours_100km", "suitability_logit"]
# Tested in notebook 05 and rejected: it made the backtest worse.
CANDIDATE_FEATURES = FEATURES + ["recent_neighbours_100km"]


def spatial_features(xy_km: np.ndarray, first_year: np.ndarray, year: int) -> pd.DataFrame:
    """Features of every cell at the start of `year`, given the year each cell was first occupied.

    - distance to the nearest occupied cell and number of occupied cells within 50 and 100 km;
    - number of cells colonised in the previous 5 years within 100 km: a proxy for an active front.
    """
    occupied = first_year < year
    tree = cKDTree(xy_km[occupied])
    distance, _ = tree.query(xy_km)
    features = {"log_distance_km": np.log1p(distance)}
    for radius in NEIGHBOURHOOD_KM:
        features[f"neighbours_{radius}km"] = np.array([len(n) for n in tree.query_ball_point(xy_km, r=radius)])
    recent = (first_year < year) & (first_year >= year - RECENT_YEARS)
    if recent.any():
        recent_tree = cKDTree(xy_km[recent])
        features["recent_neighbours_100km"] = np.array(
            [len(n) for n in recent_tree.query_ball_point(xy_km, r=RECENT_RADIUS_KM)]
        )
    else:
        features["recent_neighbours_100km"] = np.zeros(len(xy_km), dtype=int)
    return pd.DataFrame(features)


def suitability_logit(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def cell_year_table(cells: pd.DataFrame, suitability: np.ndarray, years) -> pd.DataFrame:
    """One row per cell at risk (not occupied before year t) and year t; `colonised` is 1 if first detected in t."""
    xy_km = cells[["x_km", "y_km"]].to_numpy()
    first_year = cells["first_year"].to_numpy()
    rows = []
    for year in years:
        at_risk = first_year >= year
        features = spatial_features(xy_km, first_year, year).loc[at_risk]
        features["suitability_logit"] = suitability_logit(suitability[at_risk])
        features["colonised"] = (first_year[at_risk] == year).astype(int)
        features["year"] = year
        features["cell"] = np.flatnonzero(at_risk)
        rows.append(features)
    return pd.concat(rows, ignore_index=True)


def simulate(model, cells: pd.DataFrame, suitability: np.ndarray, last_observed_year: int,
             years, n_runs: int, features=FEATURES, seed: int = 0, extra_features=None) -> np.ndarray:
    """Monte Carlo forward simulation from the occupancy observed up to `last_observed_year`.

    `extra_features(year)` optionally returns more predictors for every cell in that year (same row order as `cells`).
    Returns the simulated first-occupation year of every cell, shape (n_runs, n_cells); np.inf = never.
    """
    rng = np.random.default_rng(seed)
    xy_km = cells[["x_km", "y_km"]].to_numpy()
    suit_logit = suitability_logit(suitability)
    observed = np.where(cells["first_year"].to_numpy() <= last_observed_year, cells["first_year"].to_numpy(), np.inf)
    simulated = np.tile(observed, (n_runs, 1))
    for run in range(n_runs):
        first_year = simulated[run]
        for year in years:
            at_risk = np.isinf(first_year)
            table = spatial_features(xy_km, first_year, year)
            table["suitability_logit"] = suit_logit
            if extra_features is not None:
                table = table.join(extra_features(year).reset_index(drop=True))
            probability = model.predict_proba(table.loc[at_risk, list(features)])[:, 1]
            colonised = np.flatnonzero(at_risk)[rng.random(at_risk.sum()) < probability]
            first_year[colonised] = year
    return simulated
