# ARK Holdings Matching System
A quantitative screening tool that finds the closest Korean (KOSPI 200) equivalents to Cathie Wood's ARK Investment holdings, using KNN similarity matching on shared financial features.

## Overview
This system pulls ARK's 13F holdings, enriches them with fundamental data from yfinance and DART (Korea's financial disclosure system), then matches each holding to the 3 most similar KOSPI 200 stocks using K-Nearest Neighbors (K=3, Euclidean distance). Matches are ranked by distance and flagged as ✓ (close), ⚠ (moderate), or ✗ (poor).

## Project Structure
```
├── raw-data/
│   ├── 13F_ARK_Raw.csv              # Raw ARK 13F filing data
│   ├── ark_enriched.csv             # ARK holdings with full fundamentals
│   └── kr_financials_cache.csv      # Cached DART financial data for KOSPI stocks
├── output/
│   ├── results.xlsx                 # KNN match results
│   └── distance_distribution.png   # Histogram of rank-1 match distances
├── data_engineering.ipynb           # Feature engineering: cleans 13F, fetches yfinance fundamentals
├── knn_pipeline.py                  # All pipeline functions (data fetching, KNN, output)
├── evaluate.py                      # Evaluation metrics and distance distribution plot
├── main.py                          # Entry point: runs the full pipeline end-to-end
├── .env.example                     # Template for environment variables
└── requirements.txt
```

## How It Works
1. **Data Engineering** (`data_engineering.ipynb`) — filters ARK 13F to common stocks, fetches fundamentals via yfinance (PER, PBR, rev growth, gross margin, P/S ratio, market cap, sector), exports `ark_enriched.csv`
2. **Korean Universe** — pulls KOSPI top 200 by market cap via FinanceDataReader
3. **Korean Fundamentals** — scrapes PER, PBR, market cap, and sector for each KOSPI stock from Naver Finance; optionally enriches with rev growth, gross margin, P/S ratio from DART OpenAPI (cached to `kr_financials_cache.csv`)
4. **Theme Mapping** — maps GICS sectors (US) and KRX sub-sectors (Korean) to 7 shared theme labels: AI/Tech, Genomics/Bio, EV/Consumer, Fintech, Defense/Industrial, Internet/SaaS, Energy/Climate
5. **Feature Encoding** — theme labels are one-hot encoded and weighted 3x relative to numeric features
6. **Feature Scaling** — StandardScaler fitted on ARK data, applied to both; Korean market cap converted KRW→USD
7. **KNN Matching** — `NearestNeighbors(k=3, metric='euclidean')` fitted on KOSPI, queried with ARK
8. **Evaluation** — theme match rate, distance stats, flag distribution, and distance histogram
9. **Export** — results written to `output/results.xlsx`

## Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set DART API key for richer Korean fundamentals
cp .env.example .env
# Edit .env and add your key from https://opendart.fss.or.kr/

# 3. Run data engineering notebook to generate ark_enriched.csv
jupyter notebook data_engineering.ipynb

# 4. Run the full pipeline
python main.py
```

## Features Used for Matching
| Feature | ARK Source | Korean Source | Weight |
|---|---|---|---|
| `per` | yfinance `trailingPE` | Naver Finance | 1x |
| `pbr` | yfinance `priceToBook` | Naver Finance | 1x |
| `market_cap` | yfinance | Naver Finance (KRW→USD) | 1x |
| `rev_growth` | yfinance `revenueGrowth` | DART income statement | 1x |
| `gross_margin` | yfinance `grossMargins` | DART income statement | 1x |
| `ps_ratio` | yfinance `priceToSalesTrailing12Months` | DART-computed | 1x |
| `theme_*` (7 binary) | GICS sector → theme map | KRX sub-sector → theme map | **3x** |

`rev_growth`, `gross_margin`, `ps_ratio` require a DART API key. Without it, only `per`, `pbr`, `market_cap` are used.

## Evaluation Results (with DART, 3x theme weight)
| Metric | Value |
|---|---|
| ARK stocks matched | 88 |
| Theme match rate (rank 1) | 39.8% |
| Coverage (≥1 ✓ match) | 52.3% |
| ✓ flags | 36.4% |
| ⚠ flags | 29.5% |
| ✗ flags | 34.1% |
| Rank-1 distance mean | 2.72 |

## Limitations
- Korean fundamentals scraped from Naver Finance may break if Naver changes their HTML structure
- DART financial data takes ~40 min for 200 stocks on first run; subsequent runs use cache
- Holding companies and preferred shares often lack DART income statement data and are excluded
- ~4 ARK tickers (PSTG, THAR, VLDXD, AXL) are not on yfinance and excluded
- No ground truth labels — theme match rate and distance are internal consistency metrics, not true accuracy

## Sources
- ARK 13F filings: SEC EDGAR
- US fundamentals: [yfinance](https://github.com/ranaroussi/yfinance)
- Korean stock universe: [FinanceDataReader](https://github.com/financedata-org/FinanceDataReader)
- Korean fundamentals: [Naver Finance](https://finance.naver.com)
- Korean financial statements: [DART OpenAPI](https://opendart.fss.or.kr/) via [dart-fss](https://github.com/dart-fss/dart-fss)
