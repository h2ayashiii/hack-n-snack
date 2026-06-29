# 部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略

**Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA**

中川 慧, 竹本 悠城, 久保 健治, 加藤 真大  
人工知能学会第二種研究会資料 SIG-FIN-036-13

---

## 概要

米国と日本の株式市場は取引時間帯が重複しない。この**時間帯非同期性**を利用し、

> 米国業種ETFの当日（時点 $t$）終値間リターンから、翌営業日（時点 $t+1$）の日本業種ETFの寄引リターンを予測する

という投資仮説を検証するコードです。  
方法論の核心は、日米結合相関行列に対して**部分空間正則化付き主成分分析 (SR-PCA)** を適用し、  
グローバル・国スプレッド・シクリカル/ディフェンシブの3つの経済的ファクターを安定的に推定することです。

---

## 数式の説明

### リターン定義

**Close-to-Close リターン（推定用）:**

$$r^{cc}_{i,t} := \frac{P^{close}_{i,t}}{P^{close}_{i,t-1}} - 1 \quad \text{(eq. 1)}$$

**Open-to-Close リターン（戦略評価用、日本側のみ）:**

$$r^{oc}_{j,t} := \frac{P^{close}_{j,t}}{P^{open}_{j,t}} - 1, \quad j \in \mathcal{J} \quad \text{(eq. 2)}$$

### 標準化リターン

ウィンドウ $\mathcal{W}_t = \{t-L, \ldots, t-1\}$（$L=60$ 営業日）内の平均・標準偏差で標準化:

$$z_{i,\tau} := \frac{r^{cc}_{i,\tau} - \mu_{i,t}}{\sigma_{i,t}}, \quad \tau \in \mathcal{W}_t \quad \text{(eq. 9)}$$

### 事前部分空間の構築

事前固有ベクトル $V_0 \in \mathbb{R}^{N \times K_0}$（$K_0 = 3$）を以下の3軸で構成:

| 番号 | ファクター | 方向 |
|------|-----------|------|
| $v_1$ | グローバル | 全銘柄に等しい重み $v_1 \propto \mathbf{1}$ |
| $v_2$ | 国スプレッド | 米国を正、日本を負（$v_1$ に直交化） |
| $v_3$ | シクリカル・ディフェンシブ | 景気敏感を正、ディフェンシブを負（$v_1, v_2$ に直交化） |

事前ターゲット相関行列:

$$D_0 := \mathrm{diag}(V_0^\top C_{full} V_0) \quad \text{(eq. 10)}$$

$$C_0^{raw} := V_0 D_0 V_0^\top \quad \text{(eq. 11)}$$

$$C_0 := \Delta^{-1/2} C_0^{raw} \Delta^{-1/2}, \quad \Delta := \mathrm{diag}(C_0^{raw}), \quad \text{diag}(C_0)=\mathbf{1} \quad \text{(eq. 12)}$$

### 正則化相関行列

$$C^{reg}_t := (1 - \lambda) C_t + \lambda C_0, \quad \lambda = 0.9 \quad \text{(eq. 13)}$$

### 固有分解と日米業種のブロック分割

$$C^{reg}_t = V_t \Lambda_t V_t^\top, \quad \ell_{1,t} \geq \cdots \geq \ell_{N,t} \quad \text{(eq. 14, 15)}$$

上位 $K=3$ 固有ベクトル $V_t^{(K)} \in \mathbb{R}^{N \times K}$ を米国・日本ブロックに分割:

$$V_{U,t}^{(K)} \in \mathbb{R}^{N_U \times K}, \quad V_{J,t}^{(K)} \in \mathbb{R}^{N_J \times K} \quad \text{(eq. 16)}$$

### リードラグ・シグナル

米国側の標準化リターン $z_{U,t}$ からファクタースコアを得て、日本側へ復元:

$$f_t := \bigl(V_{U,t}^{(K)}\bigr)^\top z_{U,t} \in \mathbb{R}^K \quad \text{(eq. 18)}$$

$$\hat{z}_{J,t+1} := V_{J,t}^{(K)} f_t \in \mathbb{R}^{N_J} \quad \text{(eq. 19)}$$

**低ランク線形予測器としての表現:**

$$\hat{z}_{J,t+1} = B_t^{(K)} z_{U,t}, \quad B_t^{(K)} := V_{J,t}^{(K)} \bigl(V_{U,t}^{(K)}\bigr)^\top \in \mathbb{R}^{N_J \times N_U} \quad \text{(eq. 20, 21)}$$

$B_t^{(K)}$ は $\mathrm{rank}(B_t^{(K)}) \leq K$ を満たす**伝播行列**（米国業種ショックが日本業種へどう波及するかを表す）。

### 最適性（命題2）

共通ファクターモデル $z_{U,t} = V_U^\star g_t + \varepsilon_{U,t}$、$z_{J,t+1} = V_J^\star g_t + \varepsilon_{J,t+1}$ の下で、  
平均二乗誤差を最小化する最良線形予測は:

$$B^\star = \Sigma_{JU} \Sigma_{UU}^{-1} = \frac{1}{1 + \sigma_U^2} V_J^\star (V_U^\star)^\top \quad \text{(eq. 25)}$$

### ロングショートポートフォリオ

上位・下位 $q=0.3$ 分位のシグナルでロング・ショートを構築:

$$w_{j,t+1} := \begin{cases} +\frac{1}{|\mathcal{L}_{t+1}|} & j \in \mathcal{L}_{t+1} \\ -\frac{1}{|\mathcal{S}_{t+1}|} & j \in \mathcal{S}_{t+1} \\ 0 & \text{otherwise} \end{cases} \quad \text{(eq. 5)}$$

$$R_{t+1} := \sum_{j \in \mathcal{J}} w_{j,t+1} r^{oc}_{j,t+1} \quad \text{(eq. 7)}$$

### 評価指標

| 指標 | 式 |
|------|---|
| 年率リターン (AR) | $\mathrm{AR} = \frac{252}{T}\sum_{t=1}^T R_t$ |
| 年率リスク (RISK) | $\mathrm{RISK} = \sqrt{\frac{252}{T-1}\sum_{t=1}^T(R_t - \mu)^2}$ |
| リターン/リスク | $\mathrm{R/R} = \mathrm{AR}/\mathrm{RISK}$ |
| 最大ドローダウン (MDD) | $\mathrm{MDD} = \min_t\left\{0,\; \frac{W_t}{\max_{\tau \leq t} W_\tau}-1\right\}$ |

---

## データ

| 市場 | ETF | 業種数 | 期間 |
|------|-----|--------|------|
| 米国 | Select Sector SPDR ETF (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) | 11 | 2010-2025 |
| 日本 | NEXT FUNDS TOPIX-17 業種別 ETF (1617.T〜1633.T) | 17 | 2010-2025 |

**シクリカル/ディフェンシブ分類:**

| 市場 | シクリカル | ディフェンシブ |
|------|-----------|-------------|
| 米国 | XLB, XLE, XLF, XLRE | XLK, XLP, XLU, XLV |
| 日本 | 1618.T, 1625.T, 1629.T, 1631.T | 1617.T, 1621.T, 1627.T, 1630.T |

---

## ディレクトリ構成

```
lead_lag_strategy/
├── README.md               # 本ファイル
├── requirements.txt        # 依存パッケージ
├── config.py               # 定数・パラメータ設定
├── backtest.py             # バックテスト（検証用）
├── live.py                 # リアル実行用シグナル生成
├── data/
│   ├── __init__.py
│   └── fetcher.py          # データ取得・キャッシュ・リターン計算
├── model/
│   ├── __init__.py
│   ├── pca.py              # 部分空間正則化PCA（核心アルゴリズム）
│   ├── signal.py           # 全戦略のシグナル計算
│   └── portfolio.py        # ロングショートポートフォリオ構築
└── evaluation/
    ├── __init__.py
    └── metrics.py          # パフォーマンス指標・ファクター回帰
```

---

## セットアップ

```bash
# リポジトリのルートで実行
cd hack-n-snack

# 依存パッケージのインストール
pip install -r lead_lag_strategy/requirements.txt
```

---

## 使い方

### 1. バックテスト（論文の実証分析を再現）

```bash
# デフォルト設定（2015-2025年）
python -m lead_lag_strategy.backtest

# 期間・パラメータを指定
python -m lead_lag_strategy.backtest \
    --start 2015-01-01 \
    --end 2025-12-31 \
    --window 60 \
    --K 3 \
    --q 0.3 \
    --save-dir results

# プロット不要の場合
python -m lead_lag_strategy.backtest --no-plot
```

**出力ファイル（results/ 以下）:**

| ファイル | 内容 |
|---------|------|
| `performance_summary.csv` | Table 2: AR, RISK, R/R, MDD |
| `ff3_regression.csv` | Table 3: Fama-French 3ファクター回帰 |
| `carhart4_regression.csv` | Table 4: Carhart 4ファクター回帰 |
| `cumulative_returns.png` | Figure 2: 累積リターン推移 |
| `strategy_returns.csv` | 日次戦略リターン |

### 2. リアル実行（今日のシグナル生成）

```bash
# 本日のシグナル（推奨: PCA_SUB戦略）
python -m lead_lag_strategy.live

# 特定日付のシグナル
python -m lead_lag_strategy.live --date 2025-06-26

# JSON出力（システム連携向け）
python -m lead_lag_strategy.live --json

# 戦略を変更
python -m lead_lag_strategy.live --strategy MOM
python -m lead_lag_strategy.live --strategy PCA_PLAIN

# 伝播行列の可視化も出力
python -m lead_lag_strategy.live --propagation
```

**出力例:**
```
============================================================
  Lead-Lag Signal  |  Strategy: PCA_SUB
  Signal Date: 2025-06-26
  (Trade JP open-to-close on the NEXT trading day)
============================================================

  LONG  (+20.0% each):
    1618.T     エネルギー資源
    1623.T     鉄鋼・非鉄
    ...

  SHORT (-20.0% each):
    1617.T     食品
    1627.T     電力・ガス
    ...
```

### 3. Python API

```python
from lead_lag_strategy.data.fetcher import download_ohlcv, build_returns
from lead_lag_strategy.model.pca import SubspaceRegularisedPCA
from lead_lag_strategy.model.signal import compute_all_signals
from lead_lag_strategy.model.portfolio import compute_all_strategy_returns
from lead_lag_strategy.evaluation.metrics import summary_table
from lead_lag_strategy.config import US_TICKERS, JP_TICKERS

import pandas as pd

# データ取得
ohlcv   = download_ohlcv(start="2009-12-01", end="2025-12-31")
returns = build_returns(ohlcv)

us_cc = returns["us_cc"]
jp_cc = returns["jp_cc"]
jp_oc = returns["jp_oc"]

cc_all = pd.concat([us_cc, jp_cc], axis=1)
cc_all.columns = US_TICKERS + JP_TICKERS

# シグナル計算
signals = compute_all_signals(cc_all)

# ポートフォリオリターン計算
strategy_rets = compute_all_strategy_returns(signals, jp_oc)

# パフォーマンス評価
print(summary_table(strategy_rets))

# SR-PCA モデルを直接使う
model = SubspaceRegularisedPCA(K=3, lam=0.9, window=60)
model.fit_prior(cc_all, prior_start="2010-01-01", prior_end="2014-12-31")

t = cc_all.index[-1]
signal = model.compute_signal(cc_all, t)   # shape (17,) = JP sectors
B_t    = model.compute_propagation_matrix(cc_all, t)  # shape (17, 11)
```

---

## 戦略の説明

| 戦略 | 説明 |
|------|------|
| **PCA_SUB** | 部分空間正則化付きPCA（**提案手法**）$\lambda=0.9$ |
| **PCA_PLAIN** | 正則化なしPCA（$\lambda=0$）|
| **MOM** | 日本側単純モメンタム（eq. 31）|
| **DOUBLE** | MOM × PCA_SUB の2段ソート |

### 論文の実証結果（参考）

| 戦略 | AR (%) | RISK (%) | R/R | MDD (%) |
|------|--------|----------|-----|---------|
| MOM | 5.63 | 10.59 | 0.53 | 16.97 |
| PCA_PLAIN | 6.24 | 9.94 | 0.62 | 23.65 |
| **PCA_SUB** | **23.79** | **10.70** | **2.22** | **9.58** |
| DOUBLE | 18.86 | 11.16 | 1.69 | 12.10 |

---

## 主要パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `K` | 3 | 抽出する主成分数 |
| `LAMBDA` | 0.9 | 正則化強度（事前へのシュリンケージ） |
| `L` | 60 | ローリングウィンドウ（営業日） |
| `Q` | 0.3 | ロング/ショート分位点 |
| `PRIOR_START` | 2010-01-01 | Cfull推定開始 |
| `PRIOR_END` | 2014-12-31 | Cfull推定終了 |
| `BACKTEST_START` | 2015-01-01 | バックテスト開始 |

---

## 実装上の注意

- **日付アライメント**: 米国の時点 $t$ のClose-to-Closeリターンを使って、日本の時点 $t+1$ のOpen-to-Closeリターンを予測。`jp_oc.shift(-1)` で対応。
- **キャッシュ**: 初回ダウンロード後は `data/cache/ohlcv.parquet` にキャッシュ保存。差分取得で2回目以降は高速。
- **NaN処理**: 株式分割・上場前・祝日の欠損は `dropna(how="all")` で適切に処理。
- **Cfull推定**: 先験ターゲット相関行列 $C_0$ の推定には2010-2014年の全期間データを使用（ルックアヘッドバイアス回避）。

---

## 参考文献

1. Nakagawa, K., Takemoto, Y., Kubo, K., & Kato, M. (2026). Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA. *SIG-FIN-036-13*, JSAI.
2. 中川慧, 加藤真大, 今村光良 (2025). 事前エクスポージャー情報を活用した部分空間正則化付き主成分分析. *SIG-FIN-035*.
3. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of financial economics*, 33(1), 3-56.
4. Carhart, M. M. (1997). On persistence in mutual fund performance. *The Journal of finance*, 52(1), 57-82.
