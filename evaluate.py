import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_matches(output_df, ark_df, kr_df):
    '''
    Computes evaluation metrics for KNN match results.

    Args:
        output_df : result of build_output() — columns: ark_sym, kr_code, kr_company, distance, rank, flag
        ark_df    : ARK DataFrame with theme column
        kr_df     : Korean DataFrame with Code and theme columns

    Returns:
        dict of metrics, also prints a summary report
    '''
    # Merge themes into output
    ark_themes = ark_df[['Sym', 'theme']].rename(columns={'Sym': 'ark_sym', 'theme': 'ark_theme'})
    kr_themes  = kr_df[['Code', 'theme']].rename(columns={'Code': 'kr_code', 'theme': 'kr_theme'})

    df = output_df.merge(ark_themes, on='ark_sym', how='left')
    df = df.merge(kr_themes, on='kr_code', how='left')
    df['theme_match'] = df['ark_theme'] == df['kr_theme']

    # ── Distance stats ──────────────────────────────────────────
    rank1 = df[df['rank'] == 1]
    dist_mean   = rank1['distance'].mean()
    dist_median = rank1['distance'].median()
    dist_std    = rank1['distance'].std()

    # ── Flag distribution (all ranks) ───────────────────────────
    flag_counts = df['flag'].value_counts().to_dict()
    total       = len(df)
    flag_pct    = {k: round(v / total * 100, 1) for k, v in flag_counts.items()}

    # ── Theme match rate (rank 1 only) ──────────────────────────
    theme_match_rate = rank1['theme_match'].mean()

    # ── Theme match rate per ARK theme ──────────────────────────
    per_theme = (rank1.groupby('ark_theme')['theme_match']
                      .agg(['mean', 'count'])
                      .rename(columns={'mean': 'match_rate', 'count': 'n_stocks'})
                      .sort_values('match_rate', ascending=False))

    # ── Coverage: % of ARK stocks with at least one ✓ ──────────
    good_matches   = df[df['flag'] == '✓']['ark_sym'].nunique()
    total_ark_syms = df['ark_sym'].nunique()
    coverage       = good_matches / total_ark_syms

    metrics = {
        'n_ark_stocks':       total_ark_syms,
        'n_matches':          total,
        'dist_mean_rank1':    round(dist_mean, 4),
        'dist_median_rank1':  round(dist_median, 4),
        'dist_std_rank1':     round(dist_std, 4),
        'theme_match_rate':   round(theme_match_rate, 4),
        'coverage_pct':       round(coverage * 100, 1),
        'flag_distribution':  flag_pct,
    }

    # ── Print report ────────────────────────────────────────────
    print('=' * 50)
    print('  KNN MATCH EVALUATION REPORT')
    print('=' * 50)
    print(f"ARK stocks matched    : {total_ark_syms}")
    print(f"Total match rows      : {total}")
    print()
    print(f"Rank-1 distance  mean : {dist_mean:.4f}")
    print(f"Rank-1 distance median: {dist_median:.4f}")
    print(f"Rank-1 distance   std : {dist_std:.4f}")
    print()
    print(f"Theme match rate (rank 1): {theme_match_rate*100:.1f}%")
    print(f"Coverage (≥1 ✓ match)    : {coverage*100:.1f}%")
    print()
    print("Flag distribution:")
    for flag in ['✓', '⚠', '✗']:
        print(f"  {flag}  {flag_pct.get(flag, 0):5.1f}%  ({flag_counts.get(flag, 0)} matches)")
    print()
    print("Theme match rate by ARK theme:")
    print(per_theme.to_string())
    print('=' * 50)

    return metrics


def plot_distance_distribution(output_df):
    '''plots histogram of rank-1 match distances with flag threshold lines'''
    rank1_dist = output_df[output_df['rank'] == 1]['distance']

    plt.figure(figsize=(8, 5))
    plt.hist(rank1_dist, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    plt.axvline(1.5, color='green',  linestyle='--', label='✓ threshold (1.5)')
    plt.axvline(3.0, color='orange', linestyle='--', label='⚠ threshold (3.0)')
    plt.xlabel('Euclidean Distance (rank-1 match)')
    plt.ylabel('Number of ARK stocks')
    plt.title('Distribution of Best-Match Distances')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/distance_distribution.png', dpi=150)
    plt.close()
    print("Saved to output/distance_distribution.png")
