"""Invasion front estimation from occurrence records on a regular grid."""

from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd

AUSTRALIAN_ALBERS = 3577


def records_to_cells(records: pd.DataFrame, cell_size_m: float) -> pd.DataFrame:
    """Aggregate point records into grid cells with their first detection year."""
    points = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(records["decimalLongitude"], records["decimalLatitude"]),
        crs=4326,
    ).to_crs(AUSTRALIAN_ALBERS)
    points["cx"] = (points.geometry.x // cell_size_m).astype(int)
    points["cy"] = (points.geometry.y // cell_size_m).astype(int)

    cells = points.groupby(["cx", "cy"]).agg(
        first_year=("year", "min"),
        n_records=("year", "size"),
        lon=("decimalLongitude", "mean"),
        lat=("decimalLatitude", "mean"),
    ).reset_index()
    cells["x_km"] = (cells["cx"] + 0.5) * cell_size_m / 1000
    cells["y_km"] = (cells["cy"] + 0.5) * cell_size_m / 1000
    return cells


def contiguous_cells(cells: pd.DataFrame, seed_year: int, max_gap_cells: int) -> pd.DataFrame:
    """Keep cells connected to the cells occupied by `seed_year`, allowing gaps of up to `max_gap_cells`.

    Isolated cells (e.g. toads transported by people far ahead of the front) are dropped.
    """
    occupied = set(zip(cells["cx"], cells["cy"]))
    seeds = cells.loc[cells["first_year"] <= seed_year, ["cx", "cy"]].itertuples(index=False, name=None)
    queue = deque(seeds)
    connected = set(queue)
    offsets = [
        (dx, dy)
        for dx in range(-max_gap_cells, max_gap_cells + 1)
        for dy in range(-max_gap_cells, max_gap_cells + 1)
    ]
    while queue:
        x, y = queue.popleft()
        for dx, dy in offsets:
            neighbour = (x + dx, y + dy)
            if neighbour in occupied and neighbour not in connected:
                connected.add(neighbour)
                queue.append(neighbour)
    is_connected = [cell in connected for cell in zip(cells["cx"], cells["cy"])]
    return cells.loc[is_connected]


def front_positions(
    cells: pd.DataFrame,
    years,
    fronts: dict,
    k: int = 3,
    seed_year: int | None = None,
    max_gap_cells: int | None = None,
) -> pd.DataFrame:
    """Front position per year and front: the k-th most advanced cell along an axis (km, Australian Albers).

    `fronts` maps a name to (region_mask, axis). The front advances towards smaller values of
    `axis` (west for x_km, south for y_km). Using k > 1 makes the estimate robust to up to k-1
    outlying cells. If `seed_year` and `max_gap_cells` are given, only cells contiguous with the
    core range count.
    """
    rows = {}
    for year in years:
        occupied = cells.loc[cells["first_year"] <= year]
        if max_gap_cells is not None:
            occupied = contiguous_cells(occupied, seed_year, max_gap_cells)
        row = {}
        for name, (region_mask, axis) in fronts.items():
            values = np.sort(occupied.loc[region_mask(occupied), axis].to_numpy())
            row[name] = values[k - 1] if len(values) >= k else np.nan
        rows[year] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def strip_arrivals(cells: pd.DataFrame, axis: str, strip_km: float) -> pd.DataFrame:
    """First detection year per strip of `strip_km` along `axis`.

    `advance_km` grows in the direction the front moves (towards smaller `axis` values).
    """
    strips = cells.assign(strip_km=(cells[axis] // strip_km) * strip_km)
    arrivals = strips.groupby("strip_km").agg(
        arrival_year=("first_year", "min"),
        n_cells=("first_year", "size"),
    ).reset_index()
    arrivals["advance_km"] = -arrivals["strip_km"]
    return arrivals


def speed_km_per_year(front: pd.Series, start: int, end: int, min_years: int = 5) -> float:
    """Linear rate of advance between `start` and `end` (inclusive), positive when the front advances.

    Returns NaN if fewer than `min_years` years have a front position.
    """
    window = front.loc[start:end].dropna()
    if len(window) < min_years:
        return np.nan
    slope = np.polyfit(window.index, window.to_numpy(), 1)[0]
    return -slope
