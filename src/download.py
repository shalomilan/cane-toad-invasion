"""Download cane toad (Rhinella marina) occurrence records from the Atlas of Living Australia."""

import os
import urllib.request
import warnings
from pathlib import Path

with warnings.catch_warnings():
    # galah-python has invalid escape sequences; the SyntaxWarning prints the local install path.
    warnings.simplefilter("ignore", SyntaxWarning)
    import galah
import geopandas as gpd
from dotenv import load_dotenv

SPECIES = "Rhinella marina"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_PATH = RAW_DIR / "rhinella_marina_ala.csv"
OUTLINE_PATH = RAW_DIR / "australia_outline.gpkg"
NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

FIELDS = [
    "recordID",
    "scientificName",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "eventDate",
    "year",
    "month",
    "basisOfRecord",
    "occurrenceStatus",
    "stateProvince",
    "dataResourceName",
    "multimedia",
]


def download_occurrences(overwrite: bool = False) -> Path:
    if RAW_PATH.exists() and not overwrite:
        print(f"Using cached file: {RAW_PATH.name}")
        return RAW_PATH

    load_dotenv()
    email = os.getenv("ALA_EMAIL")
    if not email:
        raise RuntimeError("ALA_EMAIL is missing. Copy .env.example to .env and set it.")

    galah.galah_config(atlas="Australia", email=email)
    occurrences = galah.atlas_occurrences(taxa=SPECIES, fields=FIELDS)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    occurrences.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(occurrences):,} records to {RAW_PATH.name}")
    return RAW_PATH


COUNTRIES_PATH = RAW_DIR / "ne_50m_admin_0_countries.gpkg"


def download_countries(overwrite: bool = False) -> gpd.GeoDataFrame:
    if COUNTRIES_PATH.exists() and not overwrite:
        return gpd.read_file(COUNTRIES_PATH)

    zip_path = RAW_DIR / "ne_50m_admin_0_countries.zip"
    urllib.request.urlretrieve(NATURAL_EARTH_URL, zip_path)
    countries = gpd.read_file(zip_path)[["ADMIN", "ISO_A2_EH", "CONTINENT", "geometry"]]
    countries.to_file(COUNTRIES_PATH)
    zip_path.unlink()
    return countries


def download_australia_outline(overwrite: bool = False) -> gpd.GeoDataFrame:
    if OUTLINE_PATH.exists() and not overwrite:
        return gpd.read_file(OUTLINE_PATH)

    countries = download_countries()
    australia = countries.loc[countries["ADMIN"] == "Australia", ["ADMIN", "geometry"]]
    australia.to_file(OUTLINE_PATH)
    return australia


GBIF_RHINELLA_MARINA_KEY = 5216933
GBIF_NATIVE_ZIP = RAW_DIR / "gbif_rhinella_marina_south_america.zip"
GBIF_CITATION = RAW_DIR / "gbif_rhinella_marina_south_america_citation.txt"


def download_gbif_native_range(overwrite: bool = False) -> Path:
    """Request a citable GBIF download (with DOI) of R. marina records from South America."""
    if GBIF_NATIVE_ZIP.exists() and not overwrite:
        return GBIF_NATIVE_ZIP

    import time

    from pygbif import occurrences

    load_dotenv()
    user, password, email = (os.getenv(k) for k in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"))
    if not all((user, password, email)):
        raise RuntimeError("GBIF_USER, GBIF_PWD and GBIF_EMAIL must be set in .env.")

    download_key = occurrences.download(
        [
            f"taxonKey = {GBIF_RHINELLA_MARINA_KEY}",
            "continent = SOUTH_AMERICA",
            "hasCoordinate = TRUE",
            "hasGeospatialIssue = FALSE",
            "occurrenceStatus = PRESENT",
        ],
        format="SIMPLE_CSV",
        user=user,
        pwd=password,
        email=email,
    )[0]
    while (meta := occurrences.download_meta(download_key))["status"] not in ("SUCCEEDED", "FAILED", "KILLED"):
        time.sleep(30)
    if meta["status"] != "SUCCEEDED":
        raise RuntimeError(f"GBIF download {download_key} ended with status {meta['status']}")

    occurrences.download_get(download_key, path=str(RAW_DIR))
    (RAW_DIR / f"{download_key}.zip").rename(GBIF_NATIVE_ZIP)
    GBIF_CITATION.write_text(
        f"GBIF.org ({meta['created'][:10]}) GBIF Occurrence Download https://doi.org/{meta['doi']}\n",
        encoding="utf-8",
    )
    return GBIF_NATIVE_ZIP


if __name__ == "__main__":
    download_occurrences()
    download_australia_outline()


HYDROLOGY_POINTS_URL = "https://d28rz98at9flks.cloudfront.net/83149/SurfaceHydrologyPointsNational.gdb.zip"
HYDROLOGY_POINTS_ZIP = RAW_DIR / "SurfaceHydrologyPointsNational.gdb.zip"


def download_hydrology_points(overwrite: bool = False) -> Path:
    """Geoscience Australia, Surface Hydrology Points (National), CC BY 4.0: bores, dams, tanks, pools, springs."""
    if not HYDROLOGY_POINTS_ZIP.exists() or overwrite:
        urllib.request.urlretrieve(HYDROLOGY_POINTS_URL, HYDROLOGY_POINTS_ZIP)
    print(f"Hydrology points: {HYDROLOGY_POINTS_ZIP.name}")
    return HYDROLOGY_POINTS_ZIP
