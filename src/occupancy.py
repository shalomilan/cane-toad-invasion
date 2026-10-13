"""Arrival + detection model: the probability that toads are present, not only recorded.

A cell's first record happens after two processes: toads arrive (ecology), and then someone records them
(observation, which depends on survey effort). Following the logic of dynamic occupancy models with imperfect
detection (MacKenzie et al. 2002, 2003), the arrival year is a latent variable:

    P(first record in f) = sum over a <= f of P(arrive in a) * P(missed from a to f-1) * P(detected in f)

Assumption: once toads arrive in a cell, they stay (no local extinction).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from colonization import spatial_features, suitability_logit
from front import AUSTRALIAN_ALBERS


def counts_per_cell_year(records: pd.DataFrame, cells: pd.DataFrame, cell_size_m: float) -> pd.DataFrame:
    """Number of records per grid cell (rows, in the order of `cells`) and year (columns)."""
    import geopandas as gpd

    points = gpd.GeoSeries(
        gpd.points_from_xy(records["decimalLongitude"], records["decimalLatitude"]), crs=4326
    ).to_crs(AUSTRALIAN_ALBERS)
    frame = pd.DataFrame({
        "cx": (points.x // cell_size_m).astype(int).to_numpy(),
        "cy": (points.y // cell_size_m).astype(int).to_numpy(),
        "year": records["year"].astype(int).to_numpy(),
    })
    index = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_frame(cells[["cx", "cy"]]))
    frame["cell"] = index.reindex(pd.MultiIndex.from_frame(frame[["cx", "cy"]])).to_numpy()
    frame = frame.dropna(subset=["cell"]).astype({"cell": int})
    return frame.groupby(["cell", "year"]).size().unstack(fill_value=0).reindex(range(len(cells)), fill_value=0)


def value_at(matrix: pd.DataFrame, cell_ids: np.ndarray, years: np.ndarray) -> np.ndarray:
    """matrix[cell, year] for many pairs; 0 for years not in the matrix."""
    columns = matrix.columns.to_numpy()
    position = np.clip(np.searchsorted(columns, years), 0, len(columns) - 1)
    return np.where(columns[position] == years, matrix.to_numpy()[cell_ids, position], 0)


@dataclass
class LogisticFit:
    """A logistic regression stored as plain arrays, so simulations can draw a different fit per run."""

    features: list
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # intercept first

    def linear(self, table: pd.DataFrame) -> np.ndarray:
        z = (table[self.features].to_numpy(dtype=float) - self.mean) / self.scale
        return self.coef[0] + z @ self.coef[1:]


def _standardise(table: pd.DataFrame, features, weights):
    values = table[features].to_numpy(dtype=float)
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    return mean, np.where(scale > 0, scale, 1.0)


def fit_detection(table: pd.DataFrame, features, weights=None) -> LogisticFit:
    """P(toad recorded in a cell-year | toads present), from cell-years after each cell's first record."""
    weights = np.ones(len(table)) if weights is None else weights
    mean, scale = _standardise(table, features, weights)
    z = (table[features].to_numpy(dtype=float) - mean) / scale
    y = table["detected"].to_numpy(dtype=float)

    def negative_loglik(coef):
        eta = coef[0] + z @ coef[1:]
        return -np.sum(weights * (y * log_expit(eta) + (1 - y) * log_expit(-eta)))

    result = minimize(negative_loglik, np.zeros(len(features) + 1), method="L-BFGS-B")
    return LogisticFit(list(features), mean, scale, result.x)


class ArrivalLikelihood:
    """Log-likelihood of the first-record years given arrival hazards and fixed detection probabilities.

    `table` is a cell x year table of years at risk of a first record (from `colonization.cell_year_table`),
    sorted by cell and year, with the arrival features, `log1m_detection` (log of 1 - P(detected)) and
    `log_detection` for every row.
    """

    def __init__(self, table: pd.DataFrame, features, cell_weights: pd.Series | None = None):
        table = table.sort_values(["cell", "year"], kind="stable").reset_index(drop=True)
        self.features = list(features)
        cells, starts = np.unique(table["cell"].to_numpy(), return_index=True)
        self.cells, self.starts = cells, starts
        self.ends = np.r_[starts[1:], len(table)] - 1
        self.recorded = table["colonised"].to_numpy()[self.ends] == 1
        self.weights = np.ones(len(cells)) if cell_weights is None else cell_weights.reindex(cells).fillna(0).to_numpy()
        row_weights = np.repeat(self.weights, self.ends - self.starts + 1)
        self.mean, self.scale = _standardise(table, self.features, row_weights)
        self.z = (table[self.features].to_numpy(dtype=float) - self.mean) / self.scale
        log1m_d = table["log1m_detection"].to_numpy()
        self.log_detection_last = table["log_detection"].to_numpy()[self.ends]
        self.d_excl = self._cumsum_exclusive(log1m_d)
        self.d_total = np.add.reduceat(log1m_d, starts)
        self.d_excl_last = self.d_excl[self.ends]

    def _cumsum_exclusive(self, values: np.ndarray) -> np.ndarray:
        """Sum of the previous rows of the same cell."""
        total = np.cumsum(values)
        before_cell = np.repeat(np.r_[0, total[self.starts[1:] - 1]], self.ends - self.starts + 1)
        return total - values - before_cell

    def per_cell(self, coef: np.ndarray):
        eta = coef[0] + self.z @ coef[1:]
        log_h, log1m_h = log_expit(eta), log_expit(-eta)
        h_excl = self._cumsum_exclusive(log1m_h)
        term = log_h + h_excl - self.d_excl
        peak = np.maximum.reduceat(term, self.starts)
        lengths = self.ends - self.starts + 1
        m = peak + np.log(np.add.reduceat(np.exp(term - np.repeat(peak, lengths)), self.starts))
        never_arrived = np.add.reduceat(log1m_h, self.starts)
        arrived_missed = self.d_total + m
        loglik = np.where(
            self.recorded,
            self.log_detection_last + self.d_excl_last + m,
            np.logaddexp(never_arrived, arrived_missed),
        )
        return loglik, arrived_missed

    def fit(self) -> LogisticFit:
        def negative(coef):
            return -np.sum(self.weights * self.per_cell(coef)[0])

        result = minimize(negative, np.r_[-5.0, np.zeros(len(self.features))], method="L-BFGS-B")
        return LogisticFit(self.features, self.mean, self.scale, result.x)

    def presence_posterior(self, fit: LogisticFit) -> pd.Series:
        """P(toads present at the end | the cell's record history): 1 for recorded cells."""
        loglik, arrived_missed = self.per_cell(fit.coef)
        posterior = np.where(self.recorded, 1.0, np.exp(arrived_missed - loglik))
        return pd.Series(posterior, index=self.cells)


def simulate_presence(fits, cells: pd.DataFrame, suitability: np.ndarray, start_year: int, years,
                      initial_presence: np.ndarray, log_effort, add_features, n_runs: int, seed: int = 0):
    """Monte Carlo simulation of arrival and detection from `start_year` (the last observed year).

    `fits` is a list of (arrival_fit, detection_fit); run r uses fits[r % len(fits)].
    `initial_presence` is P(present at start | records) per cell; recorded cells have 1.
    `log_effort(year)` returns the effort predictor of every cell; `add_features(table, year)` adds the
    region terms and effort columns to a cell table. Returns (arrival years, record years), each
    (n_runs, n_cells), np.inf = never.
    """
    rng = np.random.default_rng(seed)
    xy_km = cells[["x_km", "y_km"]].to_numpy()
    suit = suitability_logit(suitability)
    observed = cells["first_year"].to_numpy()
    first_record_start = np.where(observed <= start_year, observed, np.inf)
    arrivals = np.full((n_runs, len(cells)), np.inf)
    records = np.tile(first_record_start, (n_runs, 1))
    for run in range(n_runs):
        arrival_fit, detection_fit = fits[run % len(fits)]
        present = rng.random(len(cells)) < initial_presence
        arrivals[run, present] = start_year
        first_record = records[run]
        for year in years:
            table = spatial_features(xy_km, first_record, year)
            table["suitability_logit"] = suit
            table["log_effort"] = log_effort(year)
            table = add_features(table, year)
            absent = ~present
            arrive = np.flatnonzero(absent)[rng.random(absent.sum()) < expit(arrival_fit.linear(table.loc[absent]))]
            present[arrive] = True
            arrivals[run, arrive] = year
            unrecorded = present & np.isinf(first_record)
            detected = np.flatnonzero(unrecorded)[
                rng.random(unrecorded.sum()) < expit(detection_fit.linear(table.loc[unrecorded]))
            ]
            first_record[detected] = year
    return arrivals, records
