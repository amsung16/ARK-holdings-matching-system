import pandas as pd
from knn_pipeline import (
    get_korean_universe, get_korean_fundamentals,
    map_themes, build_feature_matrix, scale_features,
    run_knn, build_output, export_results,
    US_THEME_MAP, KR_THEME_MAP,
)

FEATURES = ['per', 'pbr', 'market_cap']

def main():
    ark_enriched = pd.read_csv('raw-data/ark_enriched.csv')

    kr_universe  = get_korean_universe()
    kr_raw       = get_korean_fundamentals(kr_universe)

    ark_df       = map_themes(ark_enriched, 'sector', US_THEME_MAP)
    kr_df        = map_themes(kr_raw, 'sector', KR_THEME_MAP)

    ark_mat      = build_feature_matrix(ark_df, FEATURES)
    kr_mat       = build_feature_matrix(kr_df,  FEATURES)

    ark_scaled, kr_scaled = scale_features(ark_mat, kr_mat, FEATURES)
    distances, indices    = run_knn(ark_scaled, kr_scaled, k=3)

    output = build_output(ark_mat, kr_raw, distances, indices)
    export_results(output, 'output/results.xlsx')
    print(output)

if __name__ == '__main__':
    main()
