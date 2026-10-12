# Forecasting the cane toad invasion front in Australia

> Work in progress.

**Question:** where will the cane toad (*Rhinella marina*) invasion front be by 2030, and how reliable is that forecast?

## Approach
1. Download and clean occurrence records from the Atlas of Living Australia (ALA); quantify sampling bias.
2. Estimate invasion front position and spread rate over time, by region.
3. Model habitat suitability with spatially blocked cross-validation.
4. Forecast the front and validate with a temporal backtest (train up to 2015, predict 2016–2025).

## Reproduce
```bash
uv sync
cp .env.example .env   # add the email registered at ala.org.au
```

## Data
Occurrence records: [Atlas of Living Australia](https://www.ala.org.au).
