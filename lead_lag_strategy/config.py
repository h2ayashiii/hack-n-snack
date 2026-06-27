"""
Configuration constants for the Japan-US sector lead-lag strategy.
Based on: "Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA"
"""

# ── Tickers ──────────────────────────────────────────────────────────────────

US_TICKERS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

JP_TICKERS = [
    '1617.T', '1618.T', '1619.T', '1620.T', '1621.T', '1622.T', '1623.T',
    '1624.T', '1625.T', '1626.T', '1627.T', '1628.T', '1629.T', '1630.T',
    '1631.T', '1632.T', '1633.T',
]

ALL_TICKERS = US_TICKERS + JP_TICKERS

N_US = len(US_TICKERS)  # 11
N_JP = len(JP_TICKERS)  # 17

# ── Cyclical / Defensive labels (paper Section 4.1) ──────────────────────────

US_CYCLICAL  = ['XLB', 'XLE', 'XLF', 'XLRE']
US_DEFENSIVE = ['XLK', 'XLP', 'XLU', 'XLV']

JP_CYCLICAL  = ['1618.T', '1625.T', '1629.T', '1631.T']
JP_DEFENSIVE = ['1617.T', '1621.T', '1627.T', '1630.T']

# ── Model hyper-parameters ───────────────────────────────────────────────────

LAMBDA = 0.9   # shrinkage intensity toward prior (eq. 13)
K      = 3     # number of principal components to retain (top-K)
K0     = 3     # dimension of prior subspace
L      = 60    # rolling estimation window (business days)
Q      = 0.3   # quantile for long-short portfolio construction

# ── Sample periods ───────────────────────────────────────────────────────────

DATA_START       = '2010-01-01'
DATA_END         = '2025-12-31'

PRIOR_START      = '2010-01-01'   # window used to estimate C_full
PRIOR_END        = '2014-12-31'

BACKTEST_START   = '2015-01-01'   # first signal date
BACKTEST_END     = '2025-12-31'

# ── Sector name maps (for display) ──────────────────────────────────────────

US_SECTOR_NAMES = {
    'XLB':  'Materials',
    'XLC':  'Communication Services',
    'XLE':  'Energy',
    'XLF':  'Financials',
    'XLI':  'Industrials',
    'XLK':  'Information Technology',
    'XLP':  'Consumer Staples',
    'XLRE': 'Real Estate',
    'XLU':  'Utilities',
    'XLV':  'Health Care',
    'XLY':  'Consumer Discretionary',
}

JP_SECTOR_NAMES = {
    '1617.T': '食品',
    '1618.T': 'エネルギー資源',
    '1619.T': '建設・資材',
    '1620.T': '素材・化学',
    '1621.T': '医薬品',
    '1622.T': '自動車・輸送機',
    '1623.T': '鉄鋼・非鉄',
    '1624.T': '機械',
    '1625.T': '電機・精密',
    '1626.T': '情報通信・サービスその他',
    '1627.T': '電力・ガス',
    '1628.T': '運輸・物流',
    '1629.T': '商社・卸売',
    '1630.T': '小売',
    '1631.T': '銀行',
    '1632.T': '金融（除く銀行）',
    '1633.T': '不動産',
}

TRADING_DAYS_PER_YEAR = 252
