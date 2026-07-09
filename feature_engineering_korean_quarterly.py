import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pykrx import stock as pykrx_stock
from dotenv import load_dotenv

load_dotenv()

# Quarter config: snapshot date + DART filing window
QUARTERS = {
    'Q3_2025': {
        'snapshot_date': '20250930',
        'dart_bgn_de':   '20250101',
        'dart_end_de':   '20250930',
        'dart_report_tp': 'quarter',
    },
    'Q4_2025': {
        'snapshot_date': '20251231',
        'dart_bgn_de':   '20250101',
        'dart_end_de':   '20251231',
        'dart_report_tp': 'quarter',
    },
    'Q1_2026': {
        'snapshot_date': '20260331',
        'dart_bgn_de':   '20250101',
        'dart_end_de':   '20260331',
        'dart_report_tp': 'quarter',
    },
}


def get_kospi200_as_of(snapshot_date):
    '''get top 200 KOSPI stocks by market cap (uses current listing as proxy)'''
    import FinanceDataReader as fdr
    kospi = fdr.StockListing('KOSPI')
    kospi = kospi.nlargest(200, 'Marcap').reset_index(drop=True)
    kospi['Code'] = kospi['Code'].astype(str).str.zfill(6)
    return kospi[['Code', 'Marcap']]


def get_price_on_date(code, snapshot_date):
    '''get opening price for a stock on snapshot_date'''
    end = (pd.Timestamp(snapshot_date) + pd.Timedelta(days=5)).strftime('%Y%m%d')
    df = pykrx_stock.get_market_ohlcv_by_date(snapshot_date, end, code)
    if df.empty:
        return None
    return float(df['시가'].iloc[0])  # open price


def _parse(val):
    try:
        return float(str(val).replace(',', ''))
    except:
        return None


def _get_row(df, label_col, label):
    rows = df[df[label_col].astype(str) == label]
    return rows.iloc[0] if not rows.empty else None


def _best_col(val_cols, snapshot_date):
    '''pick the column with the longest cumulative period ending on or before snapshot_date'''
    snap = pd.Timestamp(snapshot_date)
    best_col, best_months = None, 0
    for col in val_cols:
        col_str = str(col)
        try:
            parts = col_str.split("'")
            date_range = [p for p in parts if '-' in p and len(p) == 17][0]
            start_str, end_str = date_range.split('-', 1)
            # handle YYYYMMDD-YYYYMMDD format
            start = pd.Timestamp(start_str[:4]+'-'+start_str[4:6]+'-'+start_str[6:8])
            end   = pd.Timestamp(end_str[:4]+'-'+end_str[4:6]+'-'+end_str[6:8])
            if end > snap:
                continue
            months = (end.year - start.year) * 12 + (end.month - start.month) + 1
            if months > best_months:
                best_months = months
                best_col = col
        except Exception:
            continue
    return best_col, best_months


def get_dart_fundamentals(corp, bgn_de, end_de, report_tp, snapshot_date, price, marcap):
    '''extract gross_margin, ps_ratio, per, pbr from DART + price'''
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fs = corp.extract_fs(
            bgn_de=bgn_de,
            end_de=end_de,
            report_tp=report_tp,
            separate=False,
            progressbar=False,
        )

    stmts = fs._statements
    # Income statement
    is_df = None
    for key in ('is', 'cis'):
        candidate = stmts.get(key)
        if candidate is not None and not candidate.empty:
            lk = [c for c in candidate.columns if 'label_ko' in str(c)]
            if lk and (candidate[lk[0]].astype(str) == '매출액').any():
                is_df = candidate
                break
    if is_df is None:
        return None

    # Balance sheet
    bs_df = stmts.get('bs')

    label_col_is = [c for c in is_df.columns if 'label_ko' in str(c)][0]
    excl = {c for c in is_df.columns if any(x in str(c) for x in ['label', 'concept', 'class'])}
    val_cols = [c for c in is_df.columns if c not in excl]
    if not val_cols:
        return None

    # Pick the best (most cumulative) column available before snapshot
    vc, months = _best_col(val_cols, snapshot_date)
    if vc is None or months == 0:
        return None

    rev_row = _get_row(is_df, label_col_is, '매출액')
    gp_row  = _get_row(is_df, label_col_is, '매출총이익')
    cg_row  = _get_row(is_df, label_col_is, '매출원가')
    ni_row  = _get_row(is_df, label_col_is, '당기순이익')

    rev = _parse(rev_row[vc]) if rev_row is not None else None
    gp  = _parse(gp_row[vc])  if gp_row  is not None else None
    if gp is None and cg_row is not None:
        cogs = _parse(cg_row[vc])
        gp = rev - cogs if rev and cogs else None
    ni = _parse(ni_row[vc]) if ni_row is not None else None

    # Annualize to full-year equivalent
    factor  = 12 / months
    rev_ann = rev * factor if rev else None
    ni_ann  = ni  * factor if ni  else None

    gross_margin = gp / rev if gp and rev else None
    ps_ratio     = marcap / rev_ann if rev_ann and marcap else None  # both KRW → dimensionless ratio

    # Book value from balance sheet for PBR
    pbr = None
    if bs_df is not None and not bs_df.empty:
        label_col_bs = [c for c in bs_df.columns if 'label_ko' in str(c)]
        if label_col_bs:
            lk_bs = label_col_bs[0]
            excl_bs = {c for c in bs_df.columns if any(x in str(c) for x in ['label', 'concept', 'class'])}
            val_bs  = [c for c in bs_df.columns if c not in excl_bs]
            if val_bs:
                vc_bs = val_bs[0]
                eq_row = _get_row(bs_df, lk_bs, '자본총계')
                if price and marcap and eq_row is not None:
                    shares = marcap / price
                    equity = _parse(eq_row[vc_bs])
                    bps    = equity / shares if equity and shares > 0 else None
                    pbr    = price / bps if bps and bps > 0 else None

    return {
        'gross_margin': gross_margin,
        'ps_ratio':     ps_ratio,
        'pbr':          pbr,
    }


def engineer_korean_quarter(label, cfg, corp_list, cache_path):
    snapshot = cfg['snapshot_date']
    print(f'\n── Korean {label} (snapshot: {snapshot}) ──')

    # Load cache
    if Path(cache_path).exists():
        cached = pd.read_csv(cache_path)
        cached['Code'] = cached['Code'].astype(str).str.zfill(6)
        done = set(cached['Code'])
        print(f'  Cache: {len(done)} stocks already fetched')
    else:
        cached = pd.DataFrame()
        done = set()

    # KOSPI 200 universe on snapshot date
    universe = get_kospi200_as_of(snapshot)
    to_fetch = universe[~universe['Code'].isin(done)]
    print(f'  Fetching {len(to_fetch)} new stocks...')

    new_records = []
    for _, row in tqdm(to_fetch.iterrows(), total=len(to_fetch)):
        code = row['Code']
        try:
            corp = corp_list.find_by_stock_code(code)
            if corp is None:
                continue

            price  = get_price_on_date(code, snapshot)
            if price is None:
                continue

            marcap = float(row['Marcap'])
            fund   = get_dart_fundamentals(
                corp, cfg['dart_bgn_de'], cfg['dart_end_de'],
                cfg['dart_report_tp'], cfg['snapshot_date'], price, marcap
            )
            if fund is None:
                continue

            new_records.append({
                'Code':       code,
                'market_cap': marcap / 1350,  # KRW→USD
                'price':      price,
                **fund,
            })
        except Exception:
            continue

    new_df = pd.DataFrame(new_records)
    combined = pd.concat([cached, new_df], ignore_index=True) if not new_df.empty else cached
    combined.to_csv(cache_path, index=False)
    print(f'  Saved {len(combined)} stocks to {cache_path}')
    return combined


def main():
    import dart_fss as dart
    dart.set_api_key(os.getenv('DART_API_KEY'))
    corp_list = dart.get_corp_list()

    for label, cfg in QUARTERS.items():
        cache_path = f'raw-data/kr_fundamentals_{label}.csv'
        engineer_korean_quarter(label, cfg, corp_list, cache_path)

    print('\nAll quarters done.')


if __name__ == '__main__':
    main()
