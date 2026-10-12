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
SEASONAL_PRECIP_VARIABLES = {
    "bio18": "precip_warmest_quarter",
    "bio19": "precip_coldest_quarter",
}
MIN_RIVER_DISCHARGE_M3S = 1.0


def climate_zone(table: pd.DataFrame) -> pd.Series:
    """Köppen arid classes (Peel et al. 2007) plus rainfall seasonality for non-arid cells.

    Arid threshold (mm) = 20 * mean annual temperature + 280 if >= 70% of rain falls in summer,
    + 0 if >= 70% falls in winter, + 140 otherwise. Desert if rain < threshold / 2.
    The summer half-year share is approximated from the warmest and coldest quarters.
    """
    annual = table["annual_precip"]
    middle_quarters = annual - table["precip_warmest_quarter"] - table["precip_coldest_quarter"]
    summer_share = (table["precip_warmest_quarter"] + middle_quarters / 2) / annual
    threshold = 20 * table["annual_mean_temp"] + np.select(
        [summer_share >= 0.7, summer_share <= 0.3], [280, 0], default=140
    )
    return pd.Series(
        np.select(
            [annual < threshold / 2, annual < threshold, summer_share >= 0.6, summer_share <= 0.4],
            ["Desert (BW)", "Semi-arid (BS)", "Humid, summer rain", "Humid, winter rain"],
            default="Humid, year-round rain",
        ),
        index=table.index,
    )


def major_rivers(min_discharge_m3s: float) -> gpd.GeoDataFrame:
    shapefile = next(n for n in zipfile.ZipFile(HYDRORIVERS_ZIP).namelist() if n.endswith(".shp"))
    rivers = gpd.read_file(f"zip://{HYDRORIVERS_ZIP.as_posix()}!{shapefile}", bbox=(112, -44, 154, -10))
    return rivers.loc[rivers["DIS_AV_CMS"] >= min_discharge_m3s, ["DIS_AV_CMS", "geometry"]].to_crs(AUSTRALIAN_ALBERS)


def land_grid(land: gpd.GeoDataFrame, crs, cell_size_m: float) -> gpd.GeoDataFrame:
    """Cells whose centre falls on `land`, in an equal-area `crs`, indexed like `front.records_to_cells`."""
    land = land.to_crs(crs)
    xmin, ymin, xmax, ymax = land.total_bounds
    cx, cy = np.meshgrid(
        np.arange(xmin // cell_size_m, xmax // cell_size_m + 1),
        np.arange(ymin // cell_size_m, ymax // cell_size_m + 1),
    )
    cells = pd.DataFrame({"cx": cx.ravel().astype(int), "cy": cy.ravel().astype(int)})
    centres = gpd.points_from_xy((cells["cx"] + 0.5) * cell_size_m, (cells["cy"] + 0.5) * cell_size_m)
    grid = gpd.GeoDataFrame(cells, geometry=centres, crs=crs)
    grid = grid.loc[grid.within(land.union_all())].reset_index(drop=True)
    lonlat = grid.to_crs(4326).geometry
    grid["lon"], grid["lat"] = lonlat.x, lonlat.y
    return grid


def australia_grid(cell_size_m: float) -> gpd.GeoDataFrame:
    return land_grid(download_australia_outline(), AUSTRALIAN_ALBERS, cell_size_m)


def add_climate(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for code, name in (BIOCLIM_VARIABLES | SEASONAL_PRECIP_VARIABLES).items():
        member = f"wc2.1_10m_{code.replace('bio', 'bio_')}.tif"
        grid[name] = sample_raster(WORLDCLIM_BIO_ZIP, member, grid["lon"], grid["lat"])
    grid["elevation"] = sample_raster(WORLDCLIM_ELEV_ZIP, "wc2.1_10m_elev.tif", grid["lon"], grid["lat"])
    return grid


def sample_raster(zip_path: Path, member: str, lon, lat) -> np.ndarray:
    with rasterio.open(f"zip://{zip_path.as_posix()}!/{member}") as raster:
        values = np.array([value[0] for value in raster.sample(zip(lon, lat))], dtype=float)
        values[values == raster.nodata] = np.nan
    return values


def distance_to_rivers_km(grid: gpd.GeoDataFrame) -> np.ndarray:
    """Distance from each cell centre to the nearest river with mean discharge >= 1 m3/s."""
    rivers = major_rivers(MIN_RIVER_DISCHARGE_M3S)[["geometry"]]
    nearest = gpd.sjoin_nearest(grid[["geometry"]], rivers, distance_col="distance_m")
    return nearest.groupby(level=0)["distance_m"].min().reindex(grid.index).to_numpy() / 1000


def environmental_grid(cell_size_m: float, overwrite: bool = False) -> pd.DataFrame:
    cache = PROCESSED_DIR / f"australia_grid_{int(cell_size_m / 1000)}km_env.csv"
    if cache.exists() and not overwrite:
        return pd.read_csv(cache)

    grid = add_climate(australia_grid(cell_size_m))
    grid["distance_to_river_km"] = distance_to_rivers_km(grid)

    table = pd.DataFrame(grid.drop(columns="geometry")).dropna()
    table["climate_zone"] = climate_zone(table)
    table.to_csv(cache, index=False)
    return table
