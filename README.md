# ARK Holdings Matching System

Finds the closest Korean (KOSPI 200) equivalents to ARK Investment's holdings using KNN similarity matching on shared financial fundamentals.

**The core idea:** Given ARK's quarterly 13F filing, which Korean large-cap stocks have the most similar financial profile (sector, gross margin, revenue growth, P/S ratio, market cap) to what ARK holds?

---

## Quickstart — Running a New Quarter

When ARK releases a new 13F filing, running the pipeline takes two steps:

### Step 1 — Download the ARK 13F CSV

Go to [ARK's website](https://ark-funds.com/funds/arkk/) → Holdings → Download CSV for the quarter.  
Save it to the `raw-data/` folder, e.g. `raw-data/ARK Q2 2026 13F.csv`.

### Step 2 — Run the pipeline

```bash
python3 run_quarter.py --quarter Q2_2026 --ark_csv "raw-data/ARK Q2 2026 13F.csv"
```

That's it. The output will be saved to `output/results_Q2_2026.xlsx`.

> **Note:** The Korean data fetch (Step 2/3) calls DART's API for ~200 stocks and takes around 20–30 minutes. Subsequent runs for the same quarter are instant because the data is cached.

To reuse cached Korean data and skip the DART fetch:
```bash
python3 run_quarter.py --quarter Q2_2026 --ark_csv "raw-data/ARK Q2 2026 13F.csv" --skip_korean
```

---

## Output

`output/results_{quarter}.xlsx` contains three sheets:

| Sheet | Contents |
|---|---|
| **Portfolio** | Recommended Korean stocks ranked by suggested allocation (%), with the ARK holdings they matched |
| **All Matches** | Full KNN output — every ARK stock with its top 3 Korean matches and distances |
| **Korean → ARK** | Reverse lookup — given a Korean stock, which ARK holdings does it match |

**Signal column:**
- ✓ Good match (distance < 1.5) — included in portfolio
- ⚠ Weak match (distance 1.5–3.0)
- ✗ No match (distance > 3.0)

---

## Setup (first time only)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get a DART API key**  
Register at [https://opendart.fss.or.kr](https://opendart.fss.or.kr) → Apply for API key (free).

**3. Set up your `.env` file**
```
DART_API_KEY=your_key_here
```

---

## How It Works

1. **ARK 13F** — Load the quarterly filing, filter to common stock only, fetch TTM fundamentals from yfinance (gross margin, P/S ratio, market cap, revenue growth) for each ticker

2. **Korean universe** — Top 200 KOSPI stocks by market cap. Sector data from Naver Finance, financial fundamentals from DART OpenAPI

3. **Feature engineering** — Both sides share the same 5 features:
   - `market_cap` (USD, KRW converted at 1350)
   - `gross_margin` (TTM gross profit / revenue)
   - `ps_ratio` (market cap / TTM revenue)
   - `rev_growth` (YoY revenue growth)
   - `rd_intensity` (R&D spend / revenue, where available)
   - Plus one-hot encoded sector theme (weighted 3×)

4. **KNN matching** — For each ARK stock, find the 3 closest Korean stocks using nan-aware Euclidean distance. Sector alignment is weighted 3× to ensure thematic similarity

5. **Portfolio construction** — Korean stocks that are the closest match (rank 1) to at least one ARK holding, within distance threshold. Allocation weight = sum of ARK's % allocations for all matched holdings, normalized to 100%

---

## Project Structure

```
├── run_quarter.py                  ← Main entry point (use this)
│
├── raw-data/
│   ├── ARK ... 13F.csv             ← Downloaded ARK 13F filings
│   ├── ark_enriched_{Q}.csv        ← Cached ARK fundamentals per quarter
│   └── kr_fundamentals_{Q}.csv     ← Cached Korean fundamentals per quarter
│
├── output/
│   ├── results_{Q}.xlsx            ← Match results and portfolio
│   ├── backtest_portfolio.xlsx     ← Historical backtest portfolio detail
│   ├── backtest_result.png         ← Backtest chart
│   └── nav_history.csv             ← Daily NAV history
│
├── knn_pipeline.py                 ← Core KNN logic and data fetching
├── feature_engineering_quarterly.py       ← ARK fundamentals (yfinance)
├── feature_engineering_korean_quarterly.py ← Korean fundamentals (DART)
├── backtest.py                     ← Walk-forward backtest (optional)
├── main.py                         ← Current-snapshot pipeline (optional)
└── evaluate.py                     ← Match quality metrics (optional)
```

---

## Quarterly Schedule

ARK files 13F reports 45 days after each quarter ends:

| Quarter | Period Ends | 13F Available ~| Run Pipeline |
|---|---|---|---|
| Q1 | March 31 | May 15 | May 15 onward |
| Q2 | June 30 | August 14 | August 14 onward |
| Q3 | September 30 | November 14 | November 14 onward |
| Q4 | December 31 | February 14 | February 14 onward |

---

## Important Notes

- This tool finds **structurally similar** Korean stocks to ARK's holdings — stocks in the same sector with similar financials. It does not predict returns or guarantee that the strategy replicates ARK's performance.
- ARK's US holdings tend to be earlier-stage and higher-growth than their KOSPI 200 equivalents. The matches are the closest available within the Korean large-cap investable universe.
- Korean data coverage is ~85–100 stocks per quarter (out of 200) due to DART API availability.
