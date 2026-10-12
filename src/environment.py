"""Environmental predictors (climate, elevation, water) on a regular grid over Australia."""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from download import download_australia_outline
from front import AUSTRALIAN_ALBERS

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

WORLDCLIM_BIO_ZIP = RAW_DIR / "wc2.1_10m_bio.zip"
WORLDCLIM_ELEV_ZIP = RAW_DIR / "wc2.1_10m_elev.zip"
HYDRORIVERS_ZIP = RAW_DIR / "HydroRIVERS_v10_au_shp.zip"

BIOCLIM_VARIABLES = {
    "bio1": "annual_mean_temp",
    "bio5": "max_temp_warmest_month",
    "bio6": "min_temp_coldest_month",
    "bio12": "annual_precip",
    "bio15": "precip_seasonality",
    "bio17": "precip_driest_quarter",
}
MIN_RIVER_DISCHARGE_M3S = 1.0


def australia_grid(cell_size_m: float) -> gpd.GeoDataFrame:
    """Cells whose centre falls on Australian land, indexed like `front.records_to_cells`."""
    land = download_australia_outline().to_crs(AUSTRALIAN_ALBERS)
    xmin, ymin, xmax, ymax = land.total_bounds
    cx, cy = np.meshgrid(
        np.arange(xmin // cell_size_m, xmax // cell_size_m + 1),
        np.arange(ymin // cell_size_m, ymax // cell_size_m + 1),
    )
    cells = pd.DataFrame({"cx": cx.ravel().astype(int), "cy": cy.ravel().astype(int)})
    centres = gpd.points_from_xy((cells["cx"] + 0.5) * cell_size_m, (cells["cy"] + 0.5) * cell_size_m)
    grid = gpd.GeoDataFrame(cells, geometry=centres, crs=AUSTRALIAN_ALBERS)
    grid = grid.loc[grid.within(land.union_all())].reset_index(drop=True)
    lonlat = grid.to_crs(4326).geometry
    grid["lon"], grid["lat"] = lonlat.x, lonlat.y
    return grid


def sample_raster(zip_path: Path, member: str, lon, lat) -> np.ndarray:
    with rasterio.open(f"zip://{zip_path.as_posix()}!/{member}") as raster:
        values = np.array([value[0] for value in raster.sample(zip(lon, lat))], dtype=float)
        values[values == raster.nodata] = np.nan
    return values


def distance_to_rivers_km(grid: gpd.GeoDataFrame) -> np.ndarray:
    """Distance from each cell centre to the nearest river with mean discharge >= 1 m3/s."""
    shapefile = next(n for n in zipfile.ZipFile(HYDRORIVERS_ZIP).namelist() if n.endswith(".shp"))
    rivers = gpd.read_file(f"zip://{HYDRORIVERS_ZIP.as_posix()}!{shapefile}", bbox=(112, -44, 154, -10))
    rivers = rivers.loc[rivers["DIS_AV_CMS"] >= MIN_RIVER_DISCHARGE_M3S, ["geometry"]].to_crs(AUSTRALIAN_ALBERS)
    nearest = gpd.sjoin_nearest(grid[["geometry"]], rivers, distance_col="distance_m")
    return nearest.groupby(level=0)["distance_m"].min().reindex(grid.index).to_numpy() / 1000


def environmental_grid(cell_size_m: float, overwrite: bool = False) -> pd.DataFrame:
    cache = PROCESSED_DIR / f"australia_grid_{int(cell_size_m / 1000)}km_env.csv"
    if cache.exists() and not overwrite:
        return pd.read_csv(cache)

    grid = australia_grid(cell_size_m)
    for code, name in BIOCLIM_VARIABLES.items():
        member = f"wc2.1_10m_{code.replace('bio', 'bio_')}.tif"
        grid[name] = sample_raster(WORLDCLIM_BIO_ZIP, member, grid["lon"], grid["lat"])
    grid["elevation"] = sample_raster(WORLDCLIM_ELEV_ZIP, "wc2.1_10m_elev.tif", grid["lon"], grid["lat"])
    grid["distance_to_river_km"] = distance_to_rivers_km(grid)

    table = pd.DataFrame(grid.drop(columns="geometry")).dropna()
    table.to_csv(cache, index=False)
    return table
