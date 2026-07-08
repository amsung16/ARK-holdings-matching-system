import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from pykrx import stock as pykrx_stock
import FinanceDataReader as fdr
from tqdm import tqdm

START = '20250707'
END   = '20260706'


def fetch_prices(codes):
    '''fetch daily close prices for a list of KRX stock codes'''
    prices = {}
    for code in tqdm(codes, desc='Fetching prices'):
        code_str = str(code).zfill(6)
        try:
            df = pykrx_stock.get_market_ohlcv_by_date(START, END, code_str)
            if df.empty:
                continue
            prices[code_str] = df['종가']
        except Exception:
            continue
    return pd.DataFrame(prices)


def fetch_benchmark():
    '''fetch KOSPI index as benchmark'''
    df = fdr.DataReader('KS11', START[:4]+'-'+START[4:6]+'-'+START[6:],
                              END[:4]+'-'+END[4:6]+'-'+END[6:])
    return df['Close']


def compute_returns(price_df):
    '''compute cumulative and total returns from a price DataFrame'''
    cum_returns = price_df / price_df.iloc[0]
    total_returns = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
    return cum_returns, total_returns


def run_forward_test():
    # Load portfolio
    portfolio = pd.read_excel('output/results.xlsx', sheet_name='Portfolio (Rank1 < 1.5)')
    portfolio['kr_code'] = portfolio['kr_code'].astype(str).str.zfill(6)
    codes = portfolio['kr_code'].tolist()
    names = dict(zip(portfolio['kr_code'], portfolio['kr_company']))

    print(f"Portfolio: {len(codes)} stocks")
    print(f"Period: {START} → {END}")
    print()

    # Fetch prices
    price_df = fetch_prices(codes)
    missing = [c for c in codes if c not in price_df.columns]
    if missing:
        print(f"Missing price data for: {missing}")

    # Drop stocks with insufficient data (< 80% of trading days)
    min_obs = int(price_df.shape[0] * 0.8)
    price_df = price_df.dropna(thresh=min_obs, axis=1)
    price_df = price_df.fillna(method='ffill')
    available_codes = price_df.columns.tolist()
    print(f"Stocks with price data: {len(available_codes)}/{len(codes)}")
    print()

    # Equal-weighted portfolio cumulative return
    cum_ret, total_ret = compute_returns(price_df)
    portfolio_cum = cum_ret.mean(axis=1)

    # Benchmark
    benchmark = fetch_benchmark()
    benchmark = benchmark.reindex(price_df.index, method='ffill')
    benchmark_cum = benchmark / benchmark.iloc[0]

    # Summary
    portfolio_total = (portfolio_cum.iloc[-1] - 1) * 100
    benchmark_total = (benchmark_cum.iloc[-1] - 1) * 100
    active_return   = portfolio_total - benchmark_total

    daily_ret = portfolio_cum.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    print('=' * 50)
    print('  FORWARD TEST RESULTS')
    print('=' * 50)
    print(f"Portfolio return : {portfolio_total:+.2f}%")
    print(f"KOSPI return     : {benchmark_total:+.2f}%")
    print(f"Active return    : {active_return:+.2f}%")
    print(f"Sharpe ratio     : {sharpe:.3f}")
    print()
    print("Individual stock returns:")
    for code in available_codes:
        ret = total_ret[code]
        name = names.get(code, code)
        print(f"  {code} {name:12s} {ret:+.1f}%")
    print('=' * 50)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    for code in available_codes:
        ax.plot(cum_ret.index, cum_ret[code], linewidth=0.8, alpha=0.4,
                label=names.get(code, code))
    ax.plot(portfolio_cum.index, portfolio_cum, linewidth=2.5,
            color='steelblue', label=f'Portfolio (equal-weight) {portfolio_total:+.1f}%')
    ax.plot(benchmark_cum.index, benchmark_cum, linewidth=2.5,
            color='tomato', linestyle='--', label=f'KOSPI {benchmark_total:+.1f}%')
    ax.axhline(1.0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title('Forward Test: Korean ARK-Mirror Portfolio vs KOSPI (1Y)', fontsize=13)
    ax.set_ylabel('Cumulative Return (1 = start)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/forward_test.png', dpi=150)
    plt.close()
    print("Chart saved to output/forward_test.png")


if __name__ == '__main__':
    run_forward_test()
