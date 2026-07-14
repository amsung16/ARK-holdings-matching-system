#!/usr/bin/env python3
"""
run_quarter.py — ARK → Korean stock matching pipeline for a single quarter.

Usage:
    python3 run_quarter.py --quarter Q2_2026 --ark_csv "raw-data/ARK Q2 2026 13F.csv"

Add --skip_korean to reuse cached Korean data (skips the slow DART fetch):
    python3 run_quarter.py --quarter Q2_2026 --ark_csv "raw-data/ARK Q2 2026 13F.csv" --skip_korean
"""

import argparse
import os
import sys
import warnings
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

NUMERIC_FEATURES = ['market_cap', 'gross_margin', 'ps_ratio', 'rd_intensity', 'rev_growth']

QUARTER_END_DATES = {
    'Q1': '0331',
    'Q2': '0630',
    'Q3': '0930',
    'Q4': '1231',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_quarter(label):
    """'Q2_2026' → config dict with snapshot/DART dates."""
    parts = label.split('_')
    if len(parts) != 2 or parts[0] not in QUARTER_END_DATES:
        print(f"ERROR: Quarter must be like Q1_2026, Q2_2026 etc. Got: {label}")
        sys.exit(1)
    q, year = parts[0], parts[1]
    end = f"{year}{QUARTER_END_DATES[q]}"
    return {
        'snapshot_date':  end,
        'dart_bgn_de':    f"{year}0101",
        'dart_end_de':    end,
        'dart_report_tp': 'quarter',
    }


def load_13f(ark_csv):
    """Load and clean ARK 13F CSV, returns common stock rows only."""
    df = pd.read_csv(ark_csv)
    df.columns = df.columns.str.strip().str.replace('"', '')
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.replace('"', '')
    df = df[df['Cl'] == 'COMMON STOCK'].copy()
    df['%'] = pd.to_numeric(df['%'].str.replace('%', '', regex=False), errors='coerce')
    return df


# ── Step 1: ARK fundamentals ──────────────────────────────────────────────────

def fetch_ark_fundamentals(ark_csv, quarter_label, snapshot_date):
    cache = f'raw-data/ark_enriched_{quarter_label}.csv'
    if Path(cache).exists():
        print(f'  Using cached ARK data: {cache}')
        return pd.read_csv(cache)

    from feature_engineering_quarterly import get_fundamentals_as_of

    df_13f = load_13f(ark_csv)
    snap = f"{snapshot_date[:4]}-{snapshot_date[4:6]}-{snapshot_date[6:]}"

    records = []
    for _, row in tqdm(df_13f.iterrows(), total=len(df_13f), desc='  ARK fundamentals'):
        sym = row['Sym']
        if not isinstance(sym, str) or not sym.strip():
            continue
        fund = get_fundamentals_as_of(sym.strip(), snap)
        if fund is None:
            continue
        records.append({'Sym': sym, 'Issuer Name': row['Issuer Name'], '%': row['%'], **fund})

    result = pd.DataFrame(records)
    result.to_csv(cache, index=False)
    print(f'  {len(result)}/{len(df_13f)} ARK stocks enriched → saved to {cache}')
    return result


# ── Step 2: Korean fundamentals ───────────────────────────────────────────────

def fetch_korean_fundamentals(cfg, quarter_label):
    cache = f'raw-data/kr_fundamentals_{quarter_label}.csv'

    import dart_fss as dart
    from feature_engineering_korean_quarterly import engineer_korean_quarter

    dart.set_api_key(os.getenv('DART_API_KEY'))
    corp_list = dart.get_corp_list()

    return engineer_korean_quarter(quarter_label, cfg, corp_list, cache)


# ── Step 3: KNN matching ──────────────────────────────────────────────────────

def run_matching(ark_df, kr_df):
    from knn_pipeline import (
        get_korean_universe, get_korean_fundamentals,
        map_themes, encode_theme, build_feature_matrix,
        scale_features, run_knn, build_output,
        US_THEME_MAP, KR_THEME_MAP,
    )

    print('  Fetching Korean sector data...')
    kr_universe = get_korean_universe()
    kr_sector   = get_korean_fundamentals(kr_universe)
    kr_sector['Code'] = kr_sector['Code'].astype(str).str.zfill(6)
    kr_df['Code']     = kr_df['Code'].astype(str).str.zfill(6)

    kr_merged = kr_df.merge(kr_sector[['Code', 'company', 'sector']], on='Code', how='left')
    kr_merged = kr_merged.dropna(subset=['sector'])

    ark_themed = map_themes(ark_df,    'sector', US_THEME_MAP)
    kr_themed  = map_themes(kr_merged, 'sector', KR_THEME_MAP)

    ark_enc, kr_enc = encode_theme(ark_themed, kr_themed)
    theme_cols = [c for c in ark_enc.columns if c.startswith('theme_')]
    features   = NUMERIC_FEATURES + theme_cols

    ark_mat = build_feature_matrix(ark_enc, features)
    kr_mat  = build_feature_matrix(kr_enc,  features)

    weights = {f: 3.0 for f in theme_cols}
    ark_scaled, kr_scaled = scale_features(ark_mat, kr_mat, features, weights=weights)
    distances, indices    = run_knn(ark_scaled, kr_scaled, k=3)
    output = build_output(ark_mat, kr_mat, distances, indices)

    return output, kr_merged


# ── Step 4: Export ────────────────────────────────────────────────────────────

def export_results(output, ark_df, ark_csv, quarter_label):
    out_path = f'output/results_{quarter_label}.xlsx'
    Path('output').mkdir(exist_ok=True)

    # Load 13F % allocations
    df_13f = load_13f(ark_csv)
    pct_map = dict(zip(df_13f['Sym'], df_13f['%']))

    # Portfolio: rank 1, distance < 1.5
    close = output[(output['rank'] == 1) & (output['distance'] < 1.5)].copy()
    close['ark_pct'] = close['ark_sym'].map(pct_map).fillna(0)

    portfolio = (
        close.groupby(['kr_code', 'kr_company'])
        .agg(
            matched_ark_stocks=('ark_sym',  lambda x: ', '.join(sorted(x))),
            n_ark_matches=     ('ark_sym',  'count'),
            raw_weight=        ('ark_pct',  'sum'),
            avg_distance=      ('distance', 'mean'),
        )
        .reset_index()
    )
    total = portfolio['raw_weight'].sum()
    portfolio['allocation_pct'] = (portfolio['raw_weight'] / total * 100).round(2) if total > 0 else 0
    portfolio = portfolio.sort_values('allocation_pct', ascending=False).reset_index(drop=True)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:

        # Sheet 1: Recommended portfolio
        portfolio[[
            'kr_code', 'kr_company', 'allocation_pct',
            'n_ark_matches', 'avg_distance', 'matched_ark_stocks',
        ]].rename(columns={
            'kr_code':            'Korean Code',
            'kr_company':         'Korean Company',
            'allocation_pct':     'Suggested Allocation (%)',
            'n_ark_matches':      '# ARK Matches',
            'avg_distance':       'Avg Distance',
            'matched_ark_stocks': 'Matched ARK Holdings',
        }).to_excel(writer, sheet_name='Portfolio', index=False)

        # Sheet 2: All matches (full KNN output)
        output.rename(columns={
            'ark_sym':    'ARK Ticker',
            'kr_code':    'Korean Code',
            'kr_company': 'Korean Company',
            'distance':   'Distance',
            'rank':       'Rank',
            'flag':       'Signal',
        }).to_excel(writer, sheet_name='All Matches', index=False)

        # Sheet 3: Korean stock → ARK holdings reverse lookup
        reverse = (
            output[output['distance'] < 1.5]
            .groupby(['kr_code', 'kr_company'])
            .agg(
                ark_matches=  ('ark_sym',  lambda x: ', '.join(sorted(x))),
                best_distance=('distance', 'min'),
                n_matches=    ('ark_sym',  'count'),
            )
            .reset_index()
            .sort_values('best_distance')
            .rename(columns={
                'kr_code':     'Korean Code',
                'kr_company':  'Korean Company',
                'ark_matches': 'Matched ARK Holdings',
                'best_distance': 'Best Distance',
                'n_matches':   '# ARK Matches',
            })
        )
        reverse.to_excel(writer, sheet_name='Korean → ARK', index=False)

    print(f'\n  Results saved to {out_path}')
    print(f'  Portfolio: {len(portfolio)} Korean stocks selected')
    print(f'\n  Top 5 recommendations:')
    for _, row in portfolio.head(5).iterrows():
        print(f'    {row["kr_company"]:20s}  {row["allocation_pct"]:5.1f}%  ← {row["matched_ark_stocks"]}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ARK → Korean stock matching pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--quarter',      required=True, help='Quarter label e.g. Q2_2026')
    parser.add_argument('--ark_csv',      required=True, help='Path to ARK 13F CSV file')
    parser.add_argument('--skip_korean',  action='store_true',
                        help='Skip DART fetch and use cached Korean data (much faster)')
    args = parser.parse_args()

    if not os.getenv('DART_API_KEY'):
        print('ERROR: DART_API_KEY not found. Add it to your .env file.')
        sys.exit(1)
    if not Path(args.ark_csv).exists():
        print(f'ERROR: File not found: {args.ark_csv}')
        sys.exit(1)

    cfg = parse_quarter(args.quarter)

    print(f'\n{"="*55}')
    print(f'  ARK → Korean Matching  |  {args.quarter}')
    print(f'  Snapshot date: {cfg["snapshot_date"]}')
    print(f'{"="*55}')

    print('\n[1/3] ARK fundamentals')
    ark_df = fetch_ark_fundamentals(args.ark_csv, args.quarter, cfg['snapshot_date'])

    print('\n[2/3] Korean fundamentals (DART)')
    if args.skip_korean and Path(f'raw-data/kr_fundamentals_{args.quarter}.csv').exists():
        print(f'  Using cached data (--skip_korean)')
        kr_df = pd.read_csv(f'raw-data/kr_fundamentals_{args.quarter}.csv')
    else:
        kr_df = fetch_korean_fundamentals(cfg, args.quarter)

    print('\n[3/3] KNN matching')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        output, kr_merged = run_matching(ark_df, kr_df)

    export_results(output, ark_df, args.ark_csv, args.quarter)
    print('\nDone.\n')


if __name__ == '__main__':
    main()
