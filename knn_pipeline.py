from sklearn.preprocessing import StandardScaler
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import FinanceDataReader as fdr

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
            'Sym': ticker,
            'rev_growth': info.get('revenueGrowth'),
            'gross_margin': info.get('grossMargins'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector')
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

def scale_features(ark_df, kr_df, features):
    '''fits StandardScaler on ARK, transforms both
    returns ark_scaled, kr_scaled as numpy arrays'''
    scaler = StandardScaler()
    ark_scaled = scaler.fit_transform(ark_df[features])
    kr_scaled  = scaler.transform(kr_df[features])   # same scaler — do not refit
    return ark_scaled, kr_scaled

def run_knn(ark_scaled, kr_scaled, k=3):
   '''fits NearestNeighbors on kr_scaled
   queries with ark_scaled
   returns distances, indices'''

def build_output(ark_df, kr_df, distances, indices):
   '''combines ARK stock + 3 Korean matches + distance scores into one DataFrame
   adds ✓/⚠/✗ flag based on distance threshold'''

def export_results(df, path):
   '''writes output DataFrame to Excel'''