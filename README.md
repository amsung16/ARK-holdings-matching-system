# ARK Holdings Matching System
A quantitative screening tool that finds the closest Korean (KOSPI 200) equivalents to Cathie Wood's ARK Investment holdings, using KNN similarity matching on shared financial features.

## Overview
This system pulls ARK's 13F holdings, enriches them with fundamental data from yfinance, then matches each holding to the 3 most similar KOSPI 200 stocks using K-Nearest Neighbors (K=3, Euclidean distance). Matches are ranked by distance and flagged as ✓ (close), ⚠ (moderate), or ✗ (poor).

## Project Structure
```
├── raw-data/
│   ├── 13F_ARK_Raw.csv          # Raw ARK 13F filing data
│   └── ark_enriched.csv         # Engineered ARK holdings with fundamentals (per, pbr, market_cap, sector)
├── output/
│   └── results.xlsx             # KNN match results
├── data_engineering.ipynb       # Feature engineering: cleans 13F data, fetches yfinance fundamentals
├── knn_pipeline.py              # All pipeline functions (data fetching, KNN, output)
├── main.py                      # Entry point: runs the full pipeline end-to-end
└── requirements.txt
```

## How It Works
1. **Data Engineering** (`data_engineering.ipynb`) — filters ARK 13F to common stocks, fetches fundamentals (PER, PBR, revenue growth, gross margin, P/S ratio, market cap, sector) via yfinance, exports `ark_enriched.csv`
2. **Korean Universe** — pulls KOSPI top 200 by market cap via FinanceDataReader
3. **Korean Fundamentals** — scrapes PER, PBR, market cap, and sector for each KOSPI stock from Naver Finance
4. **Theme Mapping** — maps GICS sectors (US) and KRX sub-sectors (Korean) to shared theme labels (AI/Tech, Genomics/Bio, EV/Consumer, Fintech, Defense/Industrial, Internet/SaaS, Energy/Climate)
5. **Feature Scaling** — StandardScaler fitted on ARK data, applied to both (market cap converted KRW→USD)
6. **KNN Matching** — `NearestNeighbors(k=3, metric='euclidean')` fitted on KOSPI, queried with ARK
7. **Export** — results written to `output/results.xlsx`

## Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data engineering notebook to generate ark_enriched.csv
jupyter notebook data_engineering.ipynb

# 3. Run the full pipeline
python main.py
```

## Features Used for Matching
| Feature | Source |
|---|---|
| `per` | yfinance `trailingPE` / Naver Finance |
| `pbr` | yfinance `priceToBook` / Naver Finance |
| `market_cap` | yfinance / Naver Finance (KRW→USD) |

## Limitations
- Korean fundamentals (PER, PBR) are scraped from Naver Finance — may break if Naver changes their HTML structure
- Revenue growth and gross margin are not available for Korean stocks, so matching is limited to valuation multiples and market cap
- KOSPI sector data (`Dept`) from FinanceDataReader is unavailable; sector is scraped from Naver Finance instead
- ~4 ARK tickers (e.g. PSTG, THAR, VLDXD, AXL) are not found on yfinance and are excluded

## Sources
- ARK 13F filings: SEC EDGAR
- US fundamentals: [yfinance](https://github.com/ranaroussi/yfinance)
- Korean stock universe: [FinanceDataReader](https://github.com/financedata-org/FinanceDataReader)
- Korean fundamentals: [Naver Finance](https://finance.naver.com)
