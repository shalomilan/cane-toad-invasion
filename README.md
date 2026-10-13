# Forecasting the cane toad invasion front in Australia

> Work in progress.

**Question:** where will the cane toad (*Rhinella marina*) invasion front be by 2030, and how reliable is that forecast?

## Approach
1. Download and clean occurrence records from the Atlas of Living Australia (ALA); quantify sampling bias.
2. Estimate invasion front position and spread rate over time, by region.
3. Model habitat suitability with spatially blocked cross-validation.
4. Forecast the front and validate with a temporal backtest (train up to 2015, predict 2016–2025).

## What the forecast means
The target is the **probability that toads are present** in each 25 km cell, not the probability that someone records them. Records are presence × detection: a cell can host toads for years before anyone reports one, especially where few people look for frogs. The model therefore has two parts: *arrival* of toads and *detection* given survey effort (an occupancy approach with imperfect detection, MacKenzie et al. 2002, 2003).

## Assumptions
- **Once toads arrive in a cell, they stay.** No local extinction is modelled. Reasonable in the humid tropics; it may fail at the arid margins, where populations can die out in droughts and recolonise later.
- Survey effort is measured by the records of other amphibian species in the same cell and year.

## Reproduce
```bash
uv sync
cp .env.example .env   # add the email registered at ala.org.au
```

## Data
Occurrence records: [Atlas of Living Australia](https://www.ala.org.au).
