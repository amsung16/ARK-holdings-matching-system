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
   '''pulls PER, PBR, market_cap for KOSPI stocks via Naver Finance'''
   headers = {'User-Agent': 'Mozilla/5.0'}

   def _scrape(code):
      r = requests.get(f'https://finance.naver.com/item/main.naver?code={code}',
                       headers=headers, timeout=5)
      soup = BeautifulSoup(r.text, 'html.parser')
      table = soup.find('table', {'class': 'per_table'})
      if not table:
         return None, None
      tds = table.find_all('td')
      per = float(tds[0].get_text(strip=True).split('배')[0].replace(',', ''))
      pbr = float(tds[2].get_text(strip=True).split('배')[0].replace(',', ''))
      return per, pbr

   records = []
   for _, row in tqdm(kr_universe.iterrows(), total=len(kr_universe), desc='Fetching Korean fundamentals'):
      code = row['Code']
      try:
         per, pbr = _scrape(code)
         records.append({
            'Code':       code,
            'company':    row['Name'],
            'market_cap': row['Marcap'],
            'per':        per,
            'pbr':        pbr,
         })
      except Exception:
         continue
   return pd.DataFrame(records)

def map_themes(df, sector_col, mapping_dict):
   '''maps GICS sector (US) or KRX sector (Korean) to shared theme labels
   →used on both DataFrames before KNN'''

def build_feature_matrix(df, features):
   '''selects feature columns, drops rows with nulls, returns clean DataFrame'''

def scale_features(ark_df, kr_df, features):
    '''fits StandardScaler on ARK, transforms both
    returns ark_scaled, kr_scaled as numpy arrays'''

def run_knn(ark_scaled, kr_scaled, k=3):
   '''fits NearestNeighbors on kr_scaled
   queries with ark_scaled
   returns distances, indices'''

def build_output(ark_df, kr_df, distances, indices):
   '''combines ARK stock + 3 Korean matches + distance scores into one DataFrame
   adds ✓/⚠/✗ flag based on distance threshold'''

def export_results(df, path):
   '''writes output DataFrame to Excel'''