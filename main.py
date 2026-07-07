import os
import pandas as pd
from dotenv import load_dotenv
from knn_pipeline import (
    get_korean_universe, get_korean_fundamentals, get_korean_financials,
    map_themes, encode_theme, build_feature_matrix, scale_features,
    run_knn, build_output, export_results, export_reverse_lookup,
    US_THEME_MAP, KR_THEME_MAP,
)
from evaluate import evaluate_matches, plot_distance_distribution

load_dotenv()
USE_DART = bool(os.getenv('DART_API_KEY'))  # enrich Korean data if key is available

NUMERIC_FEATURES = (
    ['per', 'pbr', 'market_cap', 'rev_growth', 'gross_margin', 'ps_ratio']
    if USE_DART else
    ['per', 'pbr', 'market_cap']
)

def main():
    ark_enriched = pd.read_csv('raw-data/ark_enriched.csv')

    kr_universe  = get_korean_universe()
    kr_raw       = get_korean_fundamentals(kr_universe)

    if USE_DART:
        print("DART_API_KEY found — enriching Korean stocks with rev_growth, gross_margin, ps_ratio")
        kr_raw = get_korean_financials(kr_raw)
    else:
        print("No DART_API_KEY — using per, pbr, market_cap only. Set DART_API_KEY in .env for richer features.")

    ark_df       = map_themes(ark_enriched, 'sector', US_THEME_MAP)
    kr_df        = map_themes(kr_raw, 'sector', KR_THEME_MAP)

    ark_df, kr_df = encode_theme(ark_df, kr_df)
    theme_cols    = [c for c in ark_df.columns if c.startswith('theme_')]
    FEATURES      = NUMERIC_FEATURES + theme_cols

    ark_mat      = build_feature_matrix(ark_df, FEATURES)
    kr_mat       = build_feature_matrix(kr_df,  FEATURES)

    weights = {f: 3.0 for f in theme_cols}
    ark_scaled, kr_scaled = scale_features(ark_mat, kr_mat, FEATURES, weights=weights)
    distances, indices    = run_knn(ark_scaled, kr_scaled, k=3)

    output = build_output(ark_mat, kr_raw, distances, indices)
    export_results(output, 'output/results.xlsx')
    export_reverse_lookup(output, 'output/results.xlsx')
    print(output)

    evaluate_matches(output, ark_df, kr_df)
    plot_distance_distribution(output)

if __name__ == '__main__':
    main()
