from pathlib import Path
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import FinanceDataReader as fdr
import os

def get_ark_fundamentals(tickers):
   '''loops through ARK tickers, calls yfinance, 
   returns DataFrame with rev_growth, gross_margin, 
   ps_ratio, market_cap, sector'''
   records = []
   for ticker in tqdm(tickers, desc='Pulling ARK fundamentals'):
      try:
         stock = yf.Ticker(ticker)
         info = stock.info
         record = {
            'Sym':          ticker,
            'rev_growth':   info.get('revenueGrowth'),
            'gross_margin': info.get('grossMargins'),
            'ps_ratio':     info.get('priceToSalesTrailing12Months'),
            'market_cap':   info.get('marketCap'),
            'sector':       info.get('sector'),
            'per':          info.get('trailingPE'),
            'pbr':          info.get('priceToBook'),
         }
         records.append(record)
      except Exception as e:
         continue
   return pd.DataFrame(records)

   

def get_korean_universe():
   '''pulls KOSPI 200 list from FinanceDataReader, returns top 200 by market cap'''
   kospi = fdr.StockListing('KOSPI')
   kospi = kospi.nlargest(200, 'Marcap').reset_index(drop=True)
   return kospi[['Code', 'Name', 'Dept', 'Marcap']]


def get_korean_fundamentals(kr_universe):
   '''pulls PER, PBR, market_cap, sector for KOSPI stocks via Naver Finance'''
   headers = {'User-Agent': 'Mozilla/5.0'}

   def _scrape(code):
      r = requests.get(f'https://finance.naver.com/item/main.naver?code={code}',
                       headers=headers, timeout=5)
      soup = BeautifulSoup(r.text, 'html.parser')
      table = soup.find('table', {'class': 'per_table'})
      tds = table.find_all('td')
      per = float(tds[0].get_text(strip=True).split('배')[0].replace(',', ''))
      pbr = float(tds[2].get_text(strip=True).split('배')[0].replace(',', ''))
      sector_tag = soup.find('a', href=lambda h: h and 'upjong' in h and 'sise_group_detail' in h)
      sector = sector_tag.get_text(strip=True) if sector_tag else None
      return per, pbr, sector

   records = []
   for _, row in tqdm(kr_universe.iterrows(), total=len(kr_universe), desc='Fetching Korean fundamentals'):
      code = row['Code']
      try:
         per, pbr, sector = _scrape(code)
         records.append({
            'Code':       code,
            'company':    row['Name'],
            'market_cap': row['Marcap'] / 1350,  # KRW → USD
            'per':        per,
            'pbr':        pbr,
            'sector':     sector,
         })
      except Exception:
         continue
   return pd.DataFrame(records)

def get_korean_financials(kr_df, cache_path='raw-data/kr_financials_cache.csv'):
   '''fetches rev_growth, gross_margin, ps_ratio for KOSPI stocks via DART OpenAPI.
   Requires DART_API_KEY in .env or environment.
   Results are cached to cache_path to avoid re-fetching on subsequent runs.
   Merges results into kr_df and returns updated DataFrame.'''
   from dotenv import load_dotenv
   import dart_fss as dart

   load_dotenv()
   api_key = os.getenv('DART_API_KEY')
   if not api_key:
      raise EnvironmentError("DART_API_KEY not set. Get a free key at https://opendart.fss.or.kr/")
   dart.set_api_key(api_key)

   # Load cache if it exists
   cache_file = Path(cache_path)
   if cache_file.exists():
      cached = pd.read_csv(cache_file)
      already_fetched = set(cached['Code'].astype(str))
      to_fetch = kr_df[~kr_df['Code'].astype(str).isin(already_fetched)]
      print(f"Cache hit: {len(already_fetched)} stocks. Fetching {len(to_fetch)} new stocks.")
   else:
      cached = pd.DataFrame()
      to_fetch = kr_df

   if to_fetch.empty:
      return kr_df.merge(cached, on='Code', how='left')

   corp_list = dart.get_corp_list()

   def _parse(val):
      try: return float(str(val).replace(',', ''))
      except: return None

   def _get_row(df, label_col, label):
      rows = df[df[label_col].astype(str) == label]
      return rows.iloc[0] if not rows.empty else None

   new_records = []
   for _, row in tqdm(to_fetch.iterrows(), total=len(to_fetch), desc='Fetching DART financials'):
      code = row['Code']
      try:
         corp = corp_list.find_by_stock_code(code)
         if corp is None:
            continue
         fs = corp.extract_fs(bgn_de='20230101')
         stmts = fs._statements
         # Try IS first, then CIS — pick whichever has 매출액
         is_df = None
         for key in ('is', 'cis'):
            candidate = stmts.get(key)
            if candidate is not None and not candidate.empty:
               lk = [c for c in candidate.columns if 'label_ko' in str(c)]
               if lk and (candidate[lk[0]].astype(str) == '매출액').any():
                  is_df = candidate
                  break
         if is_df is None:
            continue

         label_col = [c for c in is_df.columns if 'label_ko' in str(c)][0]
         excl_set  = {c for c in is_df.columns if any(x in str(c) for x in ['label', 'concept', 'class'])}
         val_cols  = [c for c in is_df.columns if c not in excl_set]
         if len(val_cols) < 2:
            continue

         rev_row = _get_row(is_df, label_col, '매출액')
         gp_row  = _get_row(is_df, label_col, '매출총이익')
         cg_row  = _get_row(is_df, label_col, '매출원가')
         if rev_row is None:
            continue

         rev_curr     = _parse(rev_row[val_cols[0]])
         rev_prev     = _parse(rev_row[val_cols[1]])
         rev_growth   = (rev_curr - rev_prev) / abs(rev_prev) if rev_curr and rev_prev else None
         gp           = _parse(gp_row[val_cols[0]]) if gp_row is not None else None
         if gp is None and cg_row is not None:
            cogs = _parse(cg_row[val_cols[0]])
            gp   = rev_curr - cogs if rev_curr and cogs else None
         gross_margin = gp / rev_curr if gp and rev_curr else None
         ps_ratio     = (row['market_cap'] * 1350) / rev_curr if rev_curr and row['market_cap'] else None

         new_records.append({'Code': code, 'rev_growth': rev_growth,
                              'gross_margin': gross_margin, 'ps_ratio': ps_ratio})
      except Exception:
         continue

   new_df = pd.DataFrame(new_records)
   if not new_df.empty:
      combined = pd.concat([cached, new_df], ignore_index=True)
      cache_file.parent.mkdir(parents=True, exist_ok=True)
      combined.to_csv(cache_path, index=False)
   else:
      combined = cached

   if combined.empty:
      return kr_df
   return kr_df.merge(combined, on='Code', how='left')

US_THEME_MAP = {
    'Technology':             'AI/Tech',
    'Healthcare':             'Genomics/Bio',
    'Consumer Cyclical':      'EV/Consumer',
    'Financial Services':     'Fintech',
    'Industrials':            'Defense/Industrial',
    'Communication Services': 'Internet/SaaS',
    'Energy':                 'Energy/Climate',
}

KR_THEME_MAP = {
    '반도체와반도체장비': 'AI/Tech',
    '전자장비와기기':     'AI/Tech',
    'IT서비스':           'AI/Tech',
    '소프트웨어':         'AI/Tech',
    '통신장비':           'AI/Tech',
    '디스플레이패널':     'AI/Tech',
    '전자제품':           'AI/Tech',
    '제약':               'Genomics/Bio',
    '자동차':             'EV/Consumer',
    '자동차부품':         'EV/Consumer',
    '전기장비':           'EV/Consumer',
    '전기제품':           'EV/Consumer',
    '화장품':             'EV/Consumer',
    '식품':               'EV/Consumer',
    '백화점과일반상점':   'EV/Consumer',
    '섬유,의류,신발,호화품': 'EV/Consumer',
    '항공사':             'EV/Consumer',
    '가정용기기와용품':   'EV/Consumer',
    '호텔,레스토랑,레저': 'EV/Consumer',
    '은행':               'Fintech',
    '증권':               'Fintech',
    '손해보험':           'Fintech',
    '생명보험':           'Fintech',
    '카드':               'Fintech',
    '기계':               'Defense/Industrial',
    '우주항공과국방':     'Defense/Industrial',
    '조선':               'Defense/Industrial',
    '건설':               'Defense/Industrial',
    '철강':               'Defense/Industrial',
    '비철금속':           'Defense/Industrial',
    '해운사':             'Defense/Industrial',
    '항공화물운송과물류': 'Defense/Industrial',
    '건축자재':           'Defense/Industrial',
    '상업서비스와공급품': 'Defense/Industrial',
    '무역회사와판매업체': 'Defense/Industrial',
    '게임엔터테인먼트':   'Internet/SaaS',
    '양방향미디어와서비스': 'Internet/SaaS',
    '무선통신서비스':     'Internet/SaaS',
    '다각화된통신서비스': 'Internet/SaaS',
    '방송과엔터테인먼트': 'Internet/SaaS',
    '광고':               'Internet/SaaS',
    '에너지장비및서비스': 'Energy/Climate',
    '석유와가스':         'Energy/Climate',
    '전기유틸리티':       'Energy/Climate',
    '가스유틸리티':       'Energy/Climate',
    '화학':               'Energy/Climate',
}

def map_themes(df, sector_col, mapping_dict):
   '''maps GICS sector (US) or KRX sector (Korean) to shared theme labels
   →used on both DataFrames before KNN'''
   df = df.copy()
   df['theme'] = df[sector_col].map(mapping_dict)
   return df

def build_feature_matrix(df, features):
   '''selects feature columns, drops rows with nulls, returns clean DataFrame'''
   matrix = df[features + ['Sym' if 'Sym' in df.columns else 'Code']].copy()
   matrix = matrix.dropna(subset=features)
   matrix = matrix.reset_index(drop=True)
   return matrix

def encode_theme(ark_df, kr_df):
   '''one-hot encodes the theme column using the union of both DataFrames themes
   returns ark_df, kr_df with theme_* columns appended'''
   all_themes = pd.concat([ark_df[['theme']], kr_df[['theme']]]).dropna()
   dummies = pd.get_dummies(all_themes['theme'], prefix='theme')
   ark_enc = pd.get_dummies(ark_df['theme'], prefix='theme').reindex(columns=dummies.columns, fill_value=0)
   kr_enc  = pd.get_dummies(kr_df['theme'],  prefix='theme').reindex(columns=dummies.columns, fill_value=0)
   return (pd.concat([ark_df.reset_index(drop=True), ark_enc], axis=1),
           pd.concat([kr_df.reset_index(drop=True),  kr_enc],  axis=1))

def scale_features(ark_df, kr_df, features, weights=None):
    '''fits StandardScaler on ARK, transforms both
    returns ark_scaled, kr_scaled as numpy arrays
    weights: dict of {feature: multiplier} applied after scaling'''
    import numpy as np
    scaler = StandardScaler()
    ark_scaled = scaler.fit_transform(ark_df[features])
    kr_scaled  = scaler.transform(kr_df[features])   # same scaler — do not refit
    if weights:
        w = np.array([weights.get(f, 1.0) for f in features])
        ark_scaled = ark_scaled * w
        kr_scaled  = kr_scaled  * w
    return ark_scaled, kr_scaled

def run_knn(ark_scaled, kr_scaled, k=3):
   '''fits NearestNeighbors on kr_scaled
   queries with ark_scaled
   returns distances, indices'''
   from sklearn.neighbors import NearestNeighbors
   nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
   nn.fit(kr_scaled)
   distances, indices = nn.kneighbors(ark_scaled)
   return distances, indices

def build_output(ark_df, kr_df, distances, indices):
   '''combines ARK stock + 3 Korean matches + distance scores into one DataFrame
   adds ✓/⚠/✗ flag based on distance threshold'''
   def _flag(d):
      if d < 1.5:   return '✓'
      if d < 3.0:   return '⚠'
      return '✗'

   rows = []
   for i, ark_row in ark_df.iterrows():
      for rank, (k_idx, dist) in enumerate(zip(indices[i], distances[i]), start=1):
         kr_row = kr_df.iloc[k_idx]
         rows.append({
            'ark_sym':    ark_row.get('Sym', ark_row.name),
            'kr_code':    kr_row.get('Code'),
            'kr_company': kr_row.get('company'),
            'distance':   round(float(dist), 4),
            'rank':       rank,
            'flag':       _flag(dist),
         })
   return pd.DataFrame(rows)

def export_results(df, path):
   '''writes output DataFrame to Excel'''
   Path(path).parent.mkdir(parents=True, exist_ok=True)
   df.to_excel(path, index=False)
   print(f"Results exported to {path}")

def export_reverse_lookup(output_df, path):
   '''adds a second sheet to the Excel file showing, for each Korean stock,
   all ARK stocks that matched to it — sorted by match frequency descending'''
   freq = output_df.groupby(['kr_code', 'kr_company']).size().reset_index(name='matched_by_n_ark_stocks')

   rows = []
   for _, group in output_df.groupby(['kr_code', 'kr_company'], sort=False):
      for _, r in group.sort_values('distance').iterrows():
         rows.append({
            'kr_code':               r['kr_code'],
            'kr_company':            r['kr_company'],
            'ark_sym':               r['ark_sym'],
            'rank_in_ark_portfolio': r['rank'],
            'distance':              r['distance'],
            'flag':                  r['flag'],
         })

   reverse_df = pd.DataFrame(rows)
   reverse_df = reverse_df.merge(freq, on=['kr_code', 'kr_company'], how='left')
   reverse_df = reverse_df.sort_values(['matched_by_n_ark_stocks', 'kr_code', 'distance'],
                                        ascending=[False, True, True]).reset_index(drop=True)

   # Rank-1 only sheet
   rank1_df = output_df[output_df['rank'] == 1].copy()
   freq1    = rank1_df.groupby(['kr_code', 'kr_company']).size().reset_index(name='matched_by_n_ark_stocks')
   rank1_reverse = rank1_df[['kr_code', 'kr_company', 'ark_sym', 'distance', 'flag']].copy()
   rank1_reverse = rank1_reverse.merge(freq1, on=['kr_code', 'kr_company'], how='left')
   rank1_reverse = rank1_reverse.sort_values(['matched_by_n_ark_stocks', 'kr_code', 'distance'],
                                              ascending=[False, True, True]).reset_index(drop=True)

   with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
      reverse_df.to_excel(writer, sheet_name='Korean Stock → ARK Matches', index=False)
      rank1_reverse.to_excel(writer, sheet_name='Rank 1 Only', index=False)
   print(f"Reverse lookup sheets added to {path}")