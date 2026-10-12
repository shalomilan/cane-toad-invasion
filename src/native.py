"""Native range of Rhinella marina sensu stricto (east of the Andes) and niche comparison tools."""

import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from download import GBIF_NATIVE_ZIP, download_countries
from environment import WORLDCLIM_ELEV_ZIP, add_climate, land_grid, sample_raster

SOUTH_AMERICA_ALBERS = "ESRI:102033"
NATIVE_BBOX = (-82, -25, -50, 13)
ANDES_ELEVATION_M = 2000
NORTHERN_LIMIT_LONGITUDE = -71.5
SOUTHERN_LIMIT_LATITUDE = -20
EXCLUDED_COUNTRIES = ["AR", "UY", "PY", "CL", "AW", "MQ", "CW", "BQ"]
MIN_YEAR = 1970
MAX_UNCERTAINTY_M = 10_000


def andean_crest(lat_step: float = 0.5) -> pd.Series:
    """Easternmost longitude above 2000 m per latitude band: the eastern edge of the Andes."""
    lats = np.arange(SOUTHERN_LIMIT_LATITUDE, 10 + lat_step, lat_step)
    lons = np.arange(-81, -58, 0.1)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    elevation = sample_raster(
        WORLDCLIM_ELEV_ZIP, "wc2.1_10m_elev.tif", lon_grid.ravel(), lat_grid.ravel()
    ).reshape(lat_grid.shape)
    crest = {
        lat: lons[elevation[i] > ANDES_ELEVATION_M].max() if (elevation[i] > ANDES_ELEVATION_M).any() else np.nan
        for i, lat in enumerate(lats)
    }
    return pd.Series(crest, name="crest_lon")


def load_native_records() -> pd.DataFrame:
    """GBIF records of R. marina in South America, with a flag for each filtering step."""
    with zipfile.ZipFile(GBIF_NATIVE_ZIP) as archive:
        records = pd.read_csv(archive.open(archive.namelist()[0]), sep="\t", low_memory=False, on_bad_lines="skip")
    records = records.dropna(subset=["decimalLatitude", "decimalLongitude"])

    crest = andean_crest()
    band = (records["decimalLatitude"] * 2).round().clip(SOUTHERN_LIMIT_LATITUDE * 2, 20) / 2
    crest_lon = band.map(crest).fillna(NORTHERN_LIMIT_LONGITUDE)

    records["east_of_andes"] = records["decimalLongitude"] > crest_lon
    records["in_latitude_range"] = records["decimalLatitude"] >= SOUTHERN_LIMIT_LATITUDE
    records["country_ok"] = ~records["countryCode"].isin(EXCLUDED_COUNTRIES)
    records["recent"] = records["year"] >= MIN_YEAR
    records["precise"] = ~(records["coordinateUncertaintyInMeters"] > MAX_UNCERTAINTY_M)
    return records


def native_land() -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = NATIVE_BBOX
    return download_countries().clip((xmin, ymin, xmax, ymax))


def native_grid(cell_size_m: float) -> pd.DataFrame:
    grid = add_climate(land_grid(native_land(), SOUTH_AMERICA_ALBERS, cell_size_m))
    return pd.DataFrame(grid.drop(columns="geometry")).dropna()


def niche_density(scores_occupied: np.ndarray, scores_background: np.ndarray, axis_grid: np.ndarray) -> np.ndarray:
    """Occurrence density corrected by environment availability (Broennimann et al. 2012).

    Both densities are Gaussian kernel estimates on a 2D grid of the first two principal components.
    """
    xx, yy = np.meshgrid(axis_grid[0], axis_grid[1], indexing="ij")
    points = np.vstack([xx.ravel(), yy.ravel()])
    occupied = gaussian_kde(scores_occupied.T)(points).reshape(xx.shape)
    available = gaussian_kde(scores_background.T)(points).reshape(xx.shape)
    occupied /= occupied.max()
    available /= available.max()
    corrected = np.where(available > 0.001, occupied / np.maximum(available, 1e-12), 0)
    return corrected / corrected.max(), available


def schoener_d(z1: np.ndarray, z2: np.ndarray) -> float:
    return 1 - 0.5 * np.abs(z1 / z1.sum() - z2 / z2.sum()).sum()
