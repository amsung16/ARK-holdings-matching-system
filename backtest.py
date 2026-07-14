import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pykrx import stock as pykrx_stock
import FinanceDataReader as fdr
from tqdm import tqdm
from knn_pipeline import (
    get_korean_universe, get_korean_fundamentals,
    map_themes, encode_theme, build_feature_matrix,
    scale_features, run_knn, build_output,
    US_THEME_MAP, KR_THEME_MAP,
)

STARTING_CAPITAL = 1_000_000_000  # ₩1B

NUMERIC_FEATURES = ['market_cap', 'gross_margin', 'ps_ratio', 'rd_intensity', 'rev_growth']

REBALANCE_SCHEDULE = [
    {
        'label':          'Q3_2025',
        'rebalance_date': '20251114',
        'ark_csv':  'raw-data/ark_enriched_Q3_2025.csv',
        'ark_13f':  'raw-data/ARK Investment Management LLC Q3 2025 13F Top Portfolio Holdings.csv',
        'kr_csv':   'raw-data/kr_fundamentals_Q3_2025.csv',
    },
    {
        'label':          'Q4_2025',
        'rebalance_date': '20260214',
        'ark_csv':  'raw-data/ark_enriched_Q4_2025.csv',
        'ark_13f':  'raw-data/ARK Investment Management LLC Q4 2025 13F Top Portfolio Holdings.csv',
        'kr_csv':   'raw-data/kr_fundamentals_Q4_2025.csv',
    },
    {
        'label':          'Q1_2026',
        'rebalance_date': '20260515',
        'ark_csv':  'raw-data/ark_enriched_Q1_2026.csv',
        'ark_13f':  'raw-data/ARK Investment Management LLC Q1 2026 13F Top Portfolio Holdings.csv',
        'kr_csv':   'raw-data/kr_fundamentals_Q1_2026.csv',
    },
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_open_price(code, date_str):
    '''get open price on date_str (YYYYMMDD); tries up to 5 days forward'''
    end = (pd.Timestamp(date_str) + pd.Timedelta(days=7)).strftime('%Y%m%d')
    try:
        df = pykrx_stock.get_market_ohlcv_by_date(date_str, end, str(code).zfill(6))
        return float(df['시가'].iloc[0]) if not df.empty else None
    except Exception:
        return None


def get_daily_close(code, start, end):
    '''fetch daily close prices for a stock between start and end (YYYYMMDD)'''
    try:
        df = pykrx_stock.get_market_ohlcv_by_date(start, end, str(code).zfill(6))
        return df['종가'].rename(str(code).zfill(6)) if not df.empty else None
    except Exception:
        return None


def get_benchmark(start, end):
    '''KOSPI index daily close'''
    df = fdr.DataReader('KS11',
                        f'{start[:4]}-{start[4:6]}-{start[6:]}',
                        f'{end[:4]}-{end[4:6]}-{end[6:]}')
    return df['Close']


# ── KNN pipeline per quarter ──────────────────────────────────────────────────

def build_portfolio_for_quarter(period, kr_sector_df):
    '''
    Runs KNN for one quarter and returns:
      - portfolio_df: deduplicated Korean stocks with conviction_weight column
      - ark_13f_df: 13F allocation table for this quarter
    '''
    ark_df = pd.read_csv(period['ark_csv'])
    kr_raw = pd.read_csv(period['kr_csv'])
    kr_raw['Code'] = kr_raw['Code'].astype(str).str.zfill(6)

    # Merge sector into Korean data
    kr_df = kr_raw.merge(kr_sector_df[['Code', 'company', 'sector']], on='Code', how='left')
    kr_df = kr_df.dropna(subset=['sector'])

    # Theme mapping
    ark_df  = map_themes(ark_df,  'sector', US_THEME_MAP)
    kr_df   = map_themes(kr_df,   'sector', KR_THEME_MAP)

    # Encode themes
    ark_df, kr_df = encode_theme(ark_df, kr_df)
    theme_cols = [c for c in ark_df.columns if c.startswith('theme_')]
    features   = NUMERIC_FEATURES + theme_cols

    ark_mat = build_feature_matrix(ark_df, features)
    kr_mat  = build_feature_matrix(kr_df,  features)

    if ark_mat.empty or kr_mat.empty:
        return pd.DataFrame(), pd.DataFrame()

    weights = {f: 3.0 for f in theme_cols}
    ark_scaled, kr_scaled = scale_features(ark_mat, kr_mat, features, weights=weights)
    distances, indices    = run_knn(ark_scaled, kr_scaled, k=3)

    output = build_output(ark_mat, kr_mat, distances, indices)

    # Filter rank 1, distance < 1.5
    close_matches = output[(output['rank'] == 1) & (output['distance'] < 1.5)].copy()

    # Load 13F for conviction weights
    ark_13f = pd.read_csv(period['ark_13f'])
    ark_13f.columns = ark_13f.columns.str.strip().str.replace('"', '')
    ark_13f['Sym'] = ark_13f['Sym'].astype(str).str.strip().str.replace('"', '')
    ark_13f['%']   = ark_13f['%'].astype(str).str.replace('"','').str.replace('%','').str.strip().astype(float)

    # Merge ARK % weights into matches
    close_matches = close_matches.merge(
        ark_13f[['Sym', '%']].rename(columns={'Sym': 'ark_sym', '%': 'ark_pct'}),
        on='ark_sym', how='left'
    )
    close_matches['ark_pct'] = close_matches['ark_pct'].fillna(0)

    # Aggregate by Korean stock
    portfolio = (
        close_matches.groupby(['kr_code', 'kr_company'])
        .agg(
            ark_stocks    = ('ark_sym',  lambda x: ', '.join(sorted(x))),
            n_ark_stocks  = ('ark_sym',  'count'),
            raw_weight    = ('ark_pct',  'sum'),
            avg_distance  = ('distance', 'mean'),
        )
        .reset_index()
    )
    total = portfolio['raw_weight'].sum()
    portfolio['conviction_weight'] = portfolio['raw_weight'] / total if total > 0 else 1 / len(portfolio)
    portfolio = portfolio.sort_values('conviction_weight', ascending=False).reset_index(drop=True)

    return portfolio, ark_13f


# ── Portfolio mechanics ───────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, capital):
        self.cash     = float(capital)
        self.holdings = {}   # code → shares
        self.nav_history = []

    def rebalance(self, date_str, target_weights):
        '''
        target_weights: dict {code: weight (0-1)}
        All buys/sells at open price on date_str.
        '''
        nav = self._compute_nav(date_str, use_open=True)
        transactions = []

        new_codes = set(target_weights.keys())
        old_codes = set(self.holdings.keys())

        # 1. Sell stocks that exit the portfolio
        for code in old_codes - new_codes:
            price = get_open_price(code, date_str)
            if price and self.holdings[code] > 0:
                proceeds = self.holdings[code] * price
                self.cash += proceeds
                transactions.append({'code': code, 'action': 'SELL ALL', 'shares': self.holdings[code], 'price': price})
                del self.holdings[code]

        # 2. Adjust/buy remaining positions
        for code, weight in target_weights.items():
            price = get_open_price(code, date_str)
            if not price:
                continue
            target_value  = nav * weight
            target_shares = int(target_value / price)
            current_shares = self.holdings.get(code, 0)
            diff = target_shares - current_shares

            if diff > 0:
                cost = diff * price
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[code] = self.holdings.get(code, 0) + diff
                    transactions.append({'code': code, 'action': 'BUY', 'shares': diff, 'price': price})
            elif diff < 0:
                proceeds = abs(diff) * price
                self.cash += proceeds
                self.holdings[code] = current_shares + diff
                transactions.append({'code': code, 'action': 'SELL', 'shares': abs(diff), 'price': price})

        return transactions

    def _compute_nav(self, date_str, use_open=False):
        nav = self.cash
        for code, shares in self.holdings.items():
            price = get_open_price(code, date_str) if use_open else get_open_price(code, date_str)
            if price:
                nav += shares * price
        return nav

    def record_daily_nav(self, date, prices):
        nav = self.cash
        for code, shares in self.holdings.items():
            price = prices.get(str(code).zfill(6))
            if price:
                nav += shares * price
        self.nav_history.append({'date': date, 'nav': nav})


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest():
    print('Fetching Korean sector data (Naver Finance)...')
    kr_universe  = get_korean_universe()
    kr_sector_df = get_korean_fundamentals(kr_universe)
    kr_sector_df['Code'] = kr_sector_df['Code'].astype(str).str.zfill(6)
    kr_sector_df = kr_sector_df[['Code', 'company', 'sector']].dropna(subset=['sector'])
    print(f'  {len(kr_sector_df)} stocks with sector data')

    portfolio = Portfolio(STARTING_CAPITAL)
    all_codes = set()
    rebalance_portfolios = {}

    # Run KNN and rebalance for each quarter
    for period in REBALANCE_SCHEDULE:
        print(f'\n── {period["label"]} (rebalance: {period["rebalance_date"]}) ──')
        port_df, _ = build_portfolio_for_quarter(period, kr_sector_df)

        if port_df.empty:
            print('  No portfolio built — skipping')
            continue

        print(f'  {len(port_df)} Korean stocks selected')
        print(port_df[['kr_code','kr_company','conviction_weight','n_ark_stocks','avg_distance']].to_string(index=False))

        target_weights = dict(zip(
            port_df['kr_code'].astype(str).str.zfill(6),
            port_df['conviction_weight']
        ))
        rebalance_portfolios[period['rebalance_date']] = target_weights
        all_codes.update(target_weights.keys())

        txns = portfolio.rebalance(period['rebalance_date'], target_weights)
        print(f'  Transactions: {len(txns)}')

    # Track daily NAV
    print('\nTracking daily NAV...')
    start_date = REBALANCE_SCHEDULE[0]['rebalance_date']
    end_date   = pd.Timestamp.today().strftime('%Y%m%d')

    # Fetch all price series
    price_data = {}
    for code in tqdm(all_codes, desc='Fetching prices'):
        series = get_daily_close(code, start_date, end_date)
        if series is not None:
            price_data[code] = series

    price_df = pd.DataFrame(price_data).fillna(method='ffill')

    # Replay portfolio day by day
    portfolio2 = Portfolio(STARTING_CAPITAL)
    reb_idx = 0
    reb_dates = [p['rebalance_date'] for p in REBALANCE_SCHEDULE]

    for date, row in price_df.iterrows():
        date_str = date.strftime('%Y%m%d')
        if reb_idx < len(reb_dates) and date_str >= reb_dates[reb_idx]:
            tw = rebalance_portfolios.get(reb_dates[reb_idx], {})
            portfolio2.rebalance(reb_dates[reb_idx], tw)
            reb_idx += 1
        prices = row.to_dict()
        portfolio2.record_daily_nav(date, prices)

    nav_df = pd.DataFrame(portfolio2.nav_history).set_index('date')
    nav_df['cum_return'] = nav_df['nav'] / STARTING_CAPITAL

    # Benchmark
    benchmark = get_benchmark(start_date, end_date)
    benchmark_cum = benchmark / benchmark.iloc[0]

    # Results
    port_total = (nav_df['cum_return'].iloc[-1] - 1) * 100
    bench_total = (benchmark_cum.iloc[-1] - 1) * 100
    active = port_total - bench_total
    daily_ret = nav_df['cum_return'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    print('\n' + '=' * 55)
    print('  WALK-FORWARD BACKTEST RESULTS  (₩1B starting capital)')
    print('=' * 55)
    print(f"Period          : {start_date} → {end_date}")
    print(f"Rebalances      : {len(REBALANCE_SCHEDULE)}")
    print(f"Portfolio return: {port_total:+.2f}%")
    print(f"KOSPI return    : {bench_total:+.2f}%")
    print(f"Active return   : {active:+.2f}%")
    print(f"Sharpe ratio    : {sharpe:.3f}")
    print(f"Final NAV       : ₩{nav_df['nav'].iloc[-1]:,.0f}")
    print('=' * 55)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(nav_df.index, nav_df['cum_return'], linewidth=2.5, color='steelblue',
            label=f'Portfolio {port_total:+.1f}%')
    ax.plot(benchmark_cum.index, benchmark_cum.values, linewidth=2, color='tomato',
            linestyle='--', label=f'KOSPI {bench_total:+.1f}%')
    for p in REBALANCE_SCHEDULE:
        rd = pd.Timestamp(p['rebalance_date'])
        ax.axvline(rd, color='gray', linewidth=0.8, linestyle=':')
        ax.text(rd, ax.get_ylim()[1] * 0.98, p['label'], fontsize=7,
                ha='center', color='gray')
    ax.axhline(1.0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title('Walk-Forward Backtest: Korean ARK-Mirror Portfolio vs KOSPI', fontsize=13)
    ax.set_ylabel('Cumulative Return (1 = ₩1B start)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/backtest_result.png', dpi=150)
    plt.close()
    print('Chart saved to output/backtest_result.png')

    # Export NAV table
    nav_df.to_csv('output/nav_history.csv')
    print('NAV history saved to output/nav_history.csv')

    # Export portfolio to Excel
    _export_backtest_excel(
        rebalance_portfolios, kr_sector_df, nav_df, benchmark_cum,
        port_total, bench_total, active, sharpe
    )

    return nav_df


def _export_backtest_excel(rebalance_portfolios, kr_sector_df, nav_df, benchmark_cum,
                           port_total, bench_total, active, sharpe):
    path = 'output/backtest_portfolio.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # One sheet per quarter
        for period in REBALANCE_SCHEDULE:
            label = period['label']
            rd    = period['rebalance_date']
            tw    = rebalance_portfolios.get(rd, {})
            if not tw:
                continue

            port_df, _ = build_portfolio_for_quarter(period, kr_sector_df)
            port_df['rebalance_date'] = rd
            port_df['conviction_pct'] = (port_df['conviction_weight'] * 100).round(2)
            display = port_df[[
                'rebalance_date', 'kr_code', 'kr_company',
                'conviction_pct', 'n_ark_stocks', 'avg_distance', 'ark_stocks'
            ]].rename(columns={
                'rebalance_date': 'Rebalance Date',
                'kr_code':        'Code',
                'kr_company':     'Company',
                'conviction_pct': 'Weight (%)',
                'n_ark_stocks':   '# ARK Stocks',
                'avg_distance':   'Avg Distance',
                'ark_stocks':     'Matched ARK Stocks',
            })
            display.to_excel(writer, sheet_name=label, index=False)

        # NAV history sheet
        nav_export = nav_df.copy()
        nav_export.index = nav_export.index.strftime('%Y-%m-%d')
        nav_export['NAV (₩)']       = nav_export['nav'].round(0)
        nav_export['Cum Return (%)'] = ((nav_export['cum_return'] - 1) * 100).round(2)
        nav_export[['NAV (₩)', 'Cum Return (%)']].to_excel(writer, sheet_name='NAV History')

        # Summary sheet
        summary = pd.DataFrame([
            ['Period',           f"{REBALANCE_SCHEDULE[0]['rebalance_date']} → {pd.Timestamp.today().strftime('%Y%m%d')}"],
            ['Starting Capital', f'₩{STARTING_CAPITAL:,}'],
            ['Final NAV',        f'₩{nav_df["nav"].iloc[-1]:,.0f}'],
            ['Portfolio Return', f'{port_total:+.2f}%'],
            ['KOSPI Return',     f'{bench_total:+.2f}%'],
            ['Active Return',    f'{active:+.2f}%'],
            ['Sharpe Ratio',     f'{sharpe:.3f}'],
            ['Rebalances',       len(REBALANCE_SCHEDULE)],
        ], columns=['Metric', 'Value'])
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f'Portfolio exported to {path}')


if __name__ == '__main__':
    run_backtest()
