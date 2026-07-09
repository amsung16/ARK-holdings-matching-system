import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from pathlib import Path

# Quarter snapshot dates (end of each 13F reporting period)
QUARTERS = {
    'Q3 2025': ('2025-09-30', 'raw-data/ARK Investment Management LLC Q3 2025 13F Top Portfolio Holdings.csv'),
    'Q4 2025': ('2025-12-31', 'raw-data/ARK Investment Management LLC Q4 2025 13F Top Portfolio Holdings.csv'),
    'Q1 2026': ('2026-03-31', 'raw-data/ARK Investment Management LLC Q1 2026 13F Top Portfolio Holdings.csv'),
}

def _val(series, label):
    '''safely get a row from a series by label'''
    try:
        return float(series.loc[label])
    except Exception:
        return None

def get_fundamentals_as_of(sym, snapshot_date):
    '''compute TTM fundamentals for a US ticker as of snapshot_date'''
    snap = pd.Timestamp(snapshot_date)
    ticker = yf.Ticker(sym)

    try:
        qf  = ticker.quarterly_financials
        qbs = ticker.quarterly_balance_sheet

        # Filter to quarters on or before snapshot_date
        qf  = qf.loc[:,  qf.columns  <= snap]
        qbs = qbs.loc[:, qbs.columns <= snap]

        if qf.shape[1] < 4:
            return None

        # TTM: sum of last 4 quarters
        ttm4 = qf.iloc[:, :4]
        ttm_rev  = ttm4.loc['Total Revenue'].sum()      if 'Total Revenue' in ttm4.index  else None
        ttm_gp   = ttm4.loc['Gross Profit'].sum()       if 'Gross Profit'  in ttm4.index  else None
        ttm_ni   = ttm4.loc['Net Income'].sum()         if 'Net Income'    in ttm4.index  else None

        gross_margin = ttm_gp / ttm_rev if ttm_gp and ttm_rev else None

        # Shares outstanding (most recent quarter)
        shares = None
        for lbl in ('Ordinary Shares Number', 'Share Issued', 'Diluted Average Shares'):
            if lbl in qbs.index:
                shares = float(qbs.loc[lbl].iloc[0])
                break
        if shares is None:
            for lbl in ('Diluted Average Shares', 'Basic Average Shares'):
                if lbl in qf.index:
                    shares = float(qf.loc[lbl].iloc[0])
                    break

        # Book value (most recent quarter)
        book = None
        for lbl in ('Common Stock Equity', 'Stockholders Equity'):
            if lbl in qbs.index:
                book = float(qbs.loc[lbl].iloc[0])
                break

        # Price on snapshot_date
        end = (snap + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        hist = ticker.history(start=snapshot_date, end=end)
        if hist.empty:
            return None
        price = float(hist['Close'].iloc[0])

        market_cap = price * shares if shares else None
        ps_ratio   = market_cap / ttm_rev if market_cap and ttm_rev and ttm_rev != 0 else None

        bps = book / shares if book and shares else None
        pbr = price / bps   if bps  and bps  > 0 else None

        sector = ticker.info.get('sector')

        return {
            'gross_margin': gross_margin,
            'ps_ratio':     ps_ratio,
            'market_cap':   market_cap,
            'pbr':          pbr,
            'sector':       sector,
        }

    except Exception:
        return None


def engineer_quarter(label, snapshot_date, csv_path):
    print(f'\n── {label} (snapshot: {snapshot_date}) ──')
    df_13f = pd.read_csv(csv_path)
    df_13f.columns = df_13f.columns.str.strip().str.replace('"', '')
    df_13f['Sym'] = df_13f['Sym'].str.strip().str.replace('"', '')

    # Filter to common stock only
    df_13f = df_13f[df_13f['Cl'].str.strip().str.replace('"', '') == 'COMMON STOCK']
    df_13f['%'] = df_13f['%'].str.strip().str.replace('"', '')

    records = []
    for _, row in tqdm(df_13f.iterrows(), total=len(df_13f), desc=f'Fetching {label}'):
        sym = row['Sym']
        if not isinstance(sym, str) or not sym.strip():
            continue
        fund = get_fundamentals_as_of(sym.strip(), snapshot_date)
        if fund is None:
            continue
        records.append({
            'Sym':          sym,
            'Issuer Name':  row['Issuer Name'],
            '%':            row['%'],
            **fund,
        })

    result = pd.DataFrame(records)
    print(f'  {len(result)}/{len(df_13f)} stocks enriched')
    return result


def main():
    Path('raw-data').mkdir(exist_ok=True)

    for label, (snapshot_date, csv_path) in QUARTERS.items():
        df = engineer_quarter(label, snapshot_date, csv_path)
        filename = f'raw-data/ark_enriched_{label.replace(" ", "_")}.csv'
        df.to_csv(filename, index=False)
        print(f'  → saved: {filename}')

    print('\nDone.')


if __name__ == '__main__':
    main()
