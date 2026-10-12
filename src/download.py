"""Download cane toad (Rhinella marina) occurrence records from the Atlas of Living Australia."""

import os
import urllib.request
from pathlib import Path

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


def download_australia_outline(overwrite: bool = False) -> gpd.GeoDataFrame:
    if OUTLINE_PATH.exists() and not overwrite:
        return gpd.read_file(OUTLINE_PATH)

    zip_path = RAW_DIR / "ne_50m_admin_0_countries.zip"
    urllib.request.urlretrieve(NATURAL_EARTH_URL, zip_path)
    countries = gpd.read_file(zip_path)
    australia = countries.loc[countries["ADMIN"] == "Australia", ["ADMIN", "geometry"]]
    australia.to_file(OUTLINE_PATH)
    zip_path.unlink()
    return australia


if __name__ == "__main__":
    download_occurrences()
    download_australia_outline()
