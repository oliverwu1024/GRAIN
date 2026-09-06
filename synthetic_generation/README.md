# Synthetic Time Series Generation

Generates synthetic time series with a Bayesian Local Global Trend (BLGT) model
fitted to benchmark forecasting datasets (M3, M4) from the
[Monash Time Series Forecasting Archive](https://forecastingdata.com/).

## Requirements

R with the following packages:

```r
install.packages(c("Rlgt", "forecast", "tidyverse", "moments", "ggplot2"))
```

## Files

| File | Purpose |
| --- | --- |
| `load_cif_tsf.R` | Reads `.tsf` files, rebuilds timestamps, and splits series into train/test. |
| `blgt_2.Rmd` | Fits BLGT models and generates the synthetic series. |

## Usage

1. **Download the data.** Get the yearly `.tsf` file from
   [Zenodo record 4656222](https://zenodo.org/records/4656222) and place it in
   this directory. Other frequencies (monthly, quarterly, hourly, daily, weekly)
   come from their own Zenodo records in the same archive.

2. **Convert `.tsf` to CSV.** Set the file path in `load_cif_tsf.R` (the
   `read_tsf("")` call near the bottom is a placeholder), then run it:

   ```r
   source("load_cif_tsf.R")
   ```

   This writes `m3_yearly.csv`.

3. **Generate the synthetic series.** Knit `blgt_2.Rmd`, or run its chunks
   interactively in RStudio.

## Output

One CSV per frequency, each containing 200 generated series:

```
m3_yearly_200.csv
```
then put the csv into the convert_prior folder and follow the direction there.

you can also download the file from 
