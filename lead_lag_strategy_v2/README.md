# 部分空間正則化PCAによる日米セクターのリードラグ戦略

論文 **"Lead-lag strategies for Japanese and U.S. sectors using subspace
regularization PCA"**（中川慧・竹本悠城・久保健治・加藤真大、人工知能学会
金融情報学研究会 SIG-FIN-036-13）で検証されているアルゴリズムを Python で
再実装し、ロジックの検証とリアルタイム断面シグナルの出力を行うリポジトリです。

---

## 1. アルゴリズムの概要

### 1.1 仮説（リードラグ／情報伝播）

米国市場と日本市場は取引時間が重ならない。先に引ける米国市場に**完全に
反映された情報**は、後から開く日本市場に、その**寄り付き〜日中**にかけて
遅れて織り込まれる、という仮説を検証する。

- **情報集合**: 米国セクターETF（11本）の当日 **Close-to-Close** リターン
- **予測対象**: 日本セクターETF（17本）の翌日 **Open-to-Close** リターン

この日米セクター間の予測関係を、両市場を結合した相関行列に対する
**部分空間正則化PCA (subspace regularized PCA)** で安定的に抽出する。
得られる予測子は「米国リターンから日本リターンへの低ランク線形写像」として
書け、理想化モデルの下では**最良線形予測**に一致することが示される（命題2）。

### 1.2 ユニバース（4.1節）

| | 本数 | 銘柄 |
|---|---|---|
| 米国 (U) | 11 | Select Sector SPDR ETF（XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY） |
| 日本 (J) | 17 | NEXT FUNDS TOPIX-17 ETF（1617.T 〜 1633.T） |

シクリカル／ディフェンシブの分類（第3の事前ベクトルに使用）:

- 米国シクリカル: XLB, XLE, XLF, XLRE / ディフェンシブ: XLK, XLP, XLU, XLV
- 日本シクリカル: 1618.T, 1625.T, 1629.T, 1631.T / ディフェンシブ: 1617.T, 1621.T, 1627.T, 1630.T

### 1.3 リターン定義（式1–2）

```
Close-to-Close :  rcc_{i,t} = P^close_{i,t} / P^close_{i,t-1} - 1
Open-to-Close  :  roc_{j,t} = P^close_{j,t} / P^open_{j,t}   - 1   (j ∈ J)
```

### 1.4 ローリング標準化（式8–9）

推定ウィンドウ `W_t = {t-L, ..., t-1}`（既定 `L = 60`）について

```
mu_{i,t}    = (1/L) Σ_{τ∈W_t} rcc_{i,τ}
sigma_{i,t} = sqrt( (1/L) Σ_{τ∈W_t} (rcc_{i,τ} - mu_{i,t})^2 )
z_{i,τ}     = (rcc_{i,τ} - mu_{i,t}) / sigma_{i,t}
```

標準化リターン行列 `Z_t ∈ R^{L×N}`、相関行列 `C_t = (1/L) Z_t^T Z_t ∈ R^{N×N}`。

### 1.5 部分空間事前 C0 の構成（式10–12）

`K0 = 3` 本の直交事前ベクトル `V0 = [v1, v2, v3]` を作る:

1. **グローバル因子**: `v1 ∝ 1`
2. **日米スプレッド因子**: `v2 ∝ (1_{NU}, -1_{NJ})`、`v1` に直交化
3. **シクリカル−ディフェンシブ因子**: シクリカル正・ディフェンシブ負、`v1,v2` に直交化

事前推定期間 **2010-01-01〜2014-12-31**（論文4.3節）の標準化リターンから
得た相関行列 `C_full` を用いて

```
D0     = diag( V0^T C_full V0 )          (式10)
C^raw_0 = V0 D0 V0^T                       (式11)
C0     = Δ^{-1/2} C^raw_0 Δ^{-1/2},  Δ = diag(C^raw_0)   (式12)
```

`C0` の対角は 1（相関行列として正規化）。

> 実データでは XLRE（2015-10 上場）と XLC（2018-06 上場）は事前推定期間に
> 存在しないため、`C_full` はペアワイズ完全観測で推定し、観測のない銘柄の
> 事前固有値への寄与は観測済みサブブロック上のレイリー商で代替する
> （`common.build_C0`）。欠損がなければ式(10)そのものに帰着する。

### 1.6 正則化PCA（式13–16）

各時点の相関行列 `C_t` を事前 `C0` に縮約:

```
C^reg_t = (1 - λ) C_t + λ C0,   λ ∈ [0,1]  (既定 λ = 0.9)   (式13)
C^reg_t = V_t Λ_t V_t^T                                       (式14)
```

固有値降順で上位 `K = 3` の固有ベクトル `V^(K)_t ∈ R^{N×K}` を取り、
米国ブロックと日本ブロックに分割:

```
V^(K)_{U,t} ∈ R^{NU×K},   V^(K)_{J,t} ∈ R^{NJ×K}            (式16)
```

### 1.7 リードラグ・シグナル（式17–21）

当日 t の米国ショック（標準化）から因子スコアを抽出し、日本側ローディングで翌日を予測:

```
z_{U,t}      = (rcc_{u,t} - mu_{u,t}) / sigma_{u,t}            (式17)
f_t          = (V^(K)_{U,t})^T z_{U,t}                         (式18)  因子スコア
zhat_{J,t+1} = V^(K)_{J,t} f_t                                 (式19)  予測
             = B^(K)_t z_{U,t},   B^(K)_t = V^(K)_{J,t} (V^(K)_{U,t})^T   (式20–21)
```

`B^(K)_t` は **rank ≤ K** の低ランク線形予測子（命題1）。

### 1.8 理想化モデルと最良線形予測（命題2、式23–26）

固定ローディング `V*_U, V*_J` と共通因子 `g_t`（`E[g]=0, Cov=I_K`）の下で

```
z_{U,t}   = V*_U g_t + ε_{U,t}
z_{J,t+1} = V*_J g_t + ε_{J,t+1}      （米国因子が翌日日本へ波及）
```

二乗誤差 `R(B)=E‖z_{J,t+1} - B z_{U,t}‖²` を最小化する最良線形予測は

```
B* = Σ_{JU} Σ_{UU}^{-1} = 1/(1+σ_U²) · V*_J V*_U^T            (式25)
```

すなわち提案予測子 `B^(K)_t` は、母数 `(V*_U, V*_J)` を正則化PCAで推定した
`B*` の近似になっている。

### 1.9 ポートフォリオ（式3–7）

シグナル `s_{j,t} = zhat_{j,t+1}` に基づき、上位 `q`（既定 0.3）をロング、
下位 `q` をショートする等ウェイト・ダラーニュートラル:

```
w_{j,t+1} = +1/|L_{t+1}|  (j ∈ Top-q),  -1/|S_{t+1}|  (j ∈ Bottom-q),  0 otherwise
Σ w = 0,   Σ |w| = 2
R_{t+1} = Σ_j w_{j,t+1} roc_{j,t+1}
```

### 1.10 評価指標（式27–30）

```
AR   = (a/T) Σ R_t                               年率リターン
RISK = sqrt( a/(T-1) Σ (R_t - μ)^2 )             年率リスク
R/R  = AR / RISK                                  リスク調整後リターン
MDD  = min_t ( W_t / max_{τ≤t} W_τ - 1 ),  W_t = Π (1+R)   最大ドローダウン
```

> 論文の式 (27)–(28) は年率化係数を 12（月次想定）としている。本実装では
> 日次 Open-to-Close を扱うため `periods_per_year` を引数化し既定 252 とした。

### 1.11 ベースライン（4.3節）と論文の結論（表2）

| 戦略 | 内容 |
|---|---|
| MOM | 日本側 cc リターンのトレイリング平均 `m_{j,t}` をシグナルに |
| PCA_PLAIN | `λ=0`（正則化なし）の素のPCA |
| PCA_SUB | 提案手法（部分空間正則化PCA） |
| DOUBLE | MOM と PCA_SUB の 2×2 ダブルソート（中央値分割、High×High ロング・Low×Low ショート）。いずれかのレッグが空の日はノーポジション（ダラーニュートラル維持） |

論文の実証結果（表2）では **PCA_SUB** が AR 23.79% / RISK 10.70% / R/R 2.22 /
MDD -9.58% と、リスク調整後リターンと最大ドローダウンの両面で最良だった。

---

## 2. ファイル構成

| ファイル | 役割 |
|---|---|
| `common.py` | 共通ロジック（リターン変換・標準化・C0構成・正則化PCA・シグナル・ポートフォリオ・指標・バックテストループ） |
| `verify_logic.py` | **ロジック検証**: 理想化モデルから合成データを生成し、命題1–2と表2の結論を再現。多面的な図を `output/verify_logic.png` に出力 |
| `realtime_run.py` | **リアルタイム実行**: 最新データ（yfinance、失敗時は合成）から現時点の因子構造・シグナル・翌日のロング/ショート建玉を出力。チャートを `output/` に保存 |
| `daily_report.py` | **日次レポート**: シグナルをメール（HTML/テキスト、SMTP）と Discord（Embed、Webhook）のどちらか／両方で配信（第5節） |
| `../.github/workflows/daily-signal.yml` | **日次自動実行**: GitHub Actions の cron で毎営業日 `daily_report.py` を実行 |
| `output/` | 生成されるチャート・図の出力先（git 管理外） |
| `README.md` | 本ファイル |

---

## 3. セットアップ

[uv](https://docs.astral.sh/uv/) を使ってパッケージを管理します。

### uv のインストール（未導入の場合）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 仮想環境の作成とパッケージのインストール

```bash
cd lead_lag_strategy_v2

# 仮想環境を作成してパッケージをインストール（基本パッケージ: numpy・pandas・matplotlib・scipy）
uv sync

# yfinance も含めてインストール（realtime_run.py のライブ取得を使う場合）
uv sync --extra live
```

`uv sync` は `pyproject.toml` と `uv.lock` に基づいて `.venv/` を自動作成します。

### 仮想環境の有効化

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

有効化すると、以降のターミナルセッションでは `python` コマンドが `.venv` 内のインタープリタを指します。

Python 3.11+ で動作確認済み（`.python-version` により 3.12 を既定使用）。

---

## 4. 実行方法

### 4.1 ロジック検証

仮想環境を有効化した状態で実行してください（セクション3参照）。

```bash
python verify_logic.py                       # 既定（seed=0, 1500日）→ output/verify_logic.png
python verify_logic.py --seed 1 --days 2000  # 別シード・期間
python verify_logic.py --out my_fig.png      # 図の出力先を変更
```

標準出力に以下を表示し、図を保存する:

- **命題1–2**: 推定した `B^(K)` と最良予測 `B*` のコサイン類似度、`rank(B*) ≤ K` の確認
- **正則化の効果**: ウィンドウ長 `L` を変えたときの `B*` への近似精度（短い `L` で正則化が優位）
- **表2の再現**: MOM / PCA_PLAIN / PCA_SUB / DOUBLE の AR・RISK・R/R・MDD と、PCA_SUB が最良であること

図（6パネル）: 真の `B*`／正則化PCAの `B^(K)`／素のPCAの `B^(K)`
のヒートマップ比較、ウィンドウ長 vs 近似精度、累積リターン、R/R 棒グラフ。

### 4.2 リアルタイム実行

yfinance によるライブ取得を使う場合は `uv sync --extra live` でインストール後、仮想環境を有効化してください。

```bash
# 最新日（デフォルト）
python realtime_run.py

# 特定の日付を指定
python realtime_run.py --date 2024-11-01

# 日付範囲を指定（複数日ヒートマップを出力）
python realtime_run.py --start-date 2024-10-01 --end-date 2024-11-30

# その他オプション
python realtime_run.py --no-chart               # テキストのみ（チャート不要な場合）
python realtime_run.py --offline                # ネットワークを使わず合成データで実行
python realtime_run.py --watch 300              # 300秒ごとに更新（単一日モード専用）
python realtime_run.py --L 60 --lam 0.9 --K 3 --q 0.3   # パラメータ指定
python realtime_run.py --output-dir /tmp/charts # 出力先ディレクトリを変更
```

#### 出力チャート

| モード | チャート形式 | ファイル名 |
|---|---|---|
| 単一日（`--date` または既定） | 横棒グラフ（銘柄ごとの予測リターン） | `output/realtime_signal_YYYY-MM-DD.png` |
| 日付範囲（`--start-date`/`--end-date`） | ヒートマップ（JP銘柄 × 日付、▲=LONG / ▼=SHORT） | `output/realtime_signal_START_END.png` |

チャートはすべて `output/` フォルダに保存され、git 管理対象外です。

教師あり学習ではなく PCA（教師なし）であるため、リアルタイム出力は
「現時点のモデル状態」を提示する:

- 当日の米国ショックから抽出した **共通因子スコア `f_t`**
- 正則化相関行列の **固有値スペクトル**（上位K本が保持される様子）
- **予測標準化日本リターン `zhat_{J,t+1}`** のランキング
- 翌日セッション向けの **ロング/ショート建玉**（ダラーニュートラル）

`yfinance` で最新価格を取得できない環境では、理想化モデルの合成ウィンドウへ
自動フォールバックする（出力バナーにデータソースを明示）。

---

## 5. 日次自動実行と通知配信（メール／Discord）

毎営業日、米国クローズ後・日本オープン前にシグナルを生成し、その内容を
**メールと Discord のどちらか、または両方**で配信する仕組みです。
`daily_report.py`（生成と配信）と `.github/workflows/daily-signal.yml`
（GitHub Actions の定期実行）の2つで構成されます。

シグナルの計算は配信チャネルに関係なく1回だけ行われ（`realtime_run.py`
と同じ経路）、その結果を各チャネル向けに描画してから送るだけです。

### 5.1 配信内容

| チャネル | 内容 |
|---|---|
| メール | シグナル日・パラメータ・データソース、翌セッションのロング/ショート全銘柄と `zhat`・ウェイト、共通因子スコア `f_t`（式18）、直近20営業日の実現パフォーマンス、チャート（インライン画像）、日本17業種の全ランキング。HTML とプレーンテキストの `multipart/alternative` |
| Discord | 同じシグナルの要約を1つの Embed として投稿。上位/下位 `q`（既定 top-5/bottom-5）の LONG/SHORT、共通因子スコア、実現パフォーマンス、チャート画像を添付。Discord の Embed 制限（フィールド1024文字・25個まで）に収まるよう、日本17業種の全ランキングは省略（メールまたは Artifacts 参照） |

どちらも合成データにフォールバックした場合は警告（メール: 件名と本文、
Discord: Embed 冒頭と色をオレンジに変更）が入ります。

### 5.2 配信チャネルの選択

```bash
python daily_report.py                          # 既定: メールのみ
python daily_report.py --channels discord        # Discord のみ
python daily_report.py --channels email,discord  # 両方
```

環境変数 `REPORT_CHANNELS`（例: `email,discord`）でも指定でき、
`--channels` が優先されます。両方とも未指定の場合は `email` のみ
（後方互換）。

### 5.3 手動実行

```bash
# 生成のみ（SMTP/Discordに接続しない）
# → output/daily_report_YYYY-MM-DD.{eml,html} と/または
#   output/discord_payload_YYYY-MM-DD.json
python daily_report.py --dry-run

# 特定日を指定してバックフィル／テスト（ネットワーク不要）
python daily_report.py --date 2024-11-01 --offline --dry-run --channels discord

# 実際に配信（環境変数の設定が必要、5.4/5.5参照）
python daily_report.py --channels email,discord
```

主なオプション:

| オプション | 意味 |
|---|---|
| `--channels` | 配信チャネル（`email` / `discord` / `email,discord`、既定 `email`） |
| `--dry-run` | SMTP/Discord に接続せず、選択したチャネルの内容をファイル出力 |
| `--require-live` | ライブ価格が取れない場合に合成データへフォールバックせず異常終了 |
| `--skip-if-stale N` | 最新シグナル日が N 日より古ければ何も配信せず正常終了（`--date` 指定時は無視） |
| `--trailing-days N` | 実現パフォーマンス集計期間（既定20、`0` で無効） |
| `--to` / `--from` | メール宛先・差出人（環境変数より優先） |
| `--discord-webhook` | Discord Webhook URL（環境変数より優先） |
| `--no-chart` | チャートの生成と添付を省略（両チャネル共通） |

### 5.4 メールアドレスと SMTP の設定

**既定値はモック**です。`example.com` は RFC 2606 が文書用に予約している
ドメインなので、未設定のまま実行しても実在のメールボックスには決して
届きません。

| 環境変数 | 既定値 | 用途 |
|---|---|---|
| `REPORT_TO` | `lead-lag-signals@example.com` | 宛先（カンマ区切りで複数可） |
| `REPORT_FROM` | `lead-lag-bot@example.com` | 差出人 |
| `SMTP_HOST` | （未設定） | **未設定ならドライラン**にフォールバック |
| `SMTP_PORT` | `587` | `587`/`25` は STARTTLS、`465` は暗黙TLS |
| `SMTP_USER` / `SMTP_PASSWORD` | （未設定） | 認証が必要な場合のみ |
| `SMTP_STARTTLS` | `1` | `0` で STARTTLS を無効化（ローカルテスト用） |

実運用に切り替えるには、GitHub の
**Settings → Secrets and variables → Actions** で

- **Variables** に `REPORT_TO`（実際の宛先）、必要なら `REPORT_FROM`
- **Secrets** に `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD`

を登録します。`SMTP_HOST` を登録するまではワークフローはドライランのまま
動作し、生成されたメールは実行成果物（Artifacts）として確認できます。

### 5.5 Discord Webhook の設定

Discord への投稿は **Webhook URL** だけで完結し、Bot の作成やサーバーへの
招待は不要です。

**① 投稿先チャンネルに Webhook を作成する**

1. Discord サーバーで、投稿したいテキストチャンネルの設定（歯車アイコン）
   を開く
2. 左メニューの **連携サービス（Integrations）** → **ウェブフック
   （Webhooks）** → **新しいウェブフック（New Webhook）**
3. 名前・アイコンを好みで設定（メッセージ送信者名は `daily_report.py`
   が `username: "Lead-Lag PCA Bot"` で上書きするため、ここでの名前は
   実質的に使われません）
4. **ウェブフック URL をコピー（Copy Webhook URL）** をクリックし、
   `https://discord.com/api/webhooks/<id>/<token>` 形式の URL を控える

> **この URL 自体が認証情報です。** URL を知っていれば誰でもそのチャンネルに
> 投稿できてしまうため、コードにハードコードしたり公開リポジトリにコミット
> したりせず、必ず Secrets 経由で渡してください。漏洩した場合は Discord の
> Webhook 設定画面から再生成（Regenerate）すれば無効化できます。

**② 環境変数を設定する**

| 環境変数 | 既定値 | 用途 |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | （未設定） | **未設定ならドライラン**（`output/discord_payload_*.json` に出力）にフォールバック |

ローカルで試す場合:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxxx/yyyy"
python daily_report.py --offline --channels discord   # 合成データで投稿テスト
```

実運用に切り替えるには、GitHub の
**Settings → Secrets and variables → Actions → Secrets** に
`DISCORD_WEBHOOK_URL` を登録するだけで、`.github/workflows/daily-signal.yml`
は毎営業日 Discord へ投稿するようになります（5.6節参照。この
ワークフローは `REPORT_CHANNELS` 未設定時のデフォルトを `discord` に
していて、Secret 追加だけで有効化する運用にしています）。
メールと併用したい場合のみ **Variables** の `REPORT_CHANNELS` に
`email,discord` を設定してください。

**技術的な補足**: Discord Webhook API へ `multipart/form-data` で
POST し、JSON の Embed（`payload_json`）とチャート画像（`files[0]`）を
1リクエストで送っています。追加の依存パッケージ（`requests` など）は
使わず、標準ライブラリの `urllib` のみで実装しています。

### 5.6 GitHub Actions による自動実行

`.github/workflows/daily-signal.yml` が `cron: "30 22 * * 1-5"`（UTC）で
毎営業日実行します。

```
22:30 UTC = 18:30 ET (EST) / 17:30 ET (EDT)  … 米国クローズの1.5〜2.5時間後
          = 07:30 JST（翌日）                 … 日本オープンの1.5時間前
```

論文のタイミング規約（時点 t の米国クローズ → 翌営業日 t+1 の日本
Open-to-Close）に沿った時間帯です。

- **スケジュール実行**では `--require-live --skip-if-stale 0` が付きます。
  合成データを実シグナルとして配信しないための安全策であり、また
  米国休場などで日米双方が取引していない日には「新しい米国ショックが
  存在しない」ため、前日の建玉を再送せず正常終了します。使用する
  チャネルは Variables の `REPORT_CHANNELS` に従いますが、**このワーク
  フローは未設定時のデフォルトを `discord` にしている**（`daily_report.py`
  単体の既定 `email` とは異なる）ため、`DISCORD_WEBHOOK_URL` の Secret
  さえ設定すれば追加設定なしで Discord 配信が有効になります。
- **手動実行**（Actions タブ → Run workflow）では `date` / `dry_run` /
  `offline` / `channels` を指定でき、ネットワークなしでパイプライン
  全体を試せます（例: `channels=discord` で Discord 投稿だけテスト）。
- 生成物は常に Artifacts（`daily-signal-<run_id>`、保持30日）へアップロード
  されるため、配信の成否にかかわらず内容を確認できます。

**課金枠について**: GitHub Actions のスケジュール実行は、パブリック
リポジトリでは無料、プライベートリポジトリでも Free プランの月2,000分の
無料枠内で動作します（本ジョブは1回あたり数分）。Discord Webhook の
利用自体にも料金は発生しません。

**既知の注意点**:

- GitHub の cron はベストエフォートで、混雑時は数分〜1時間程度遅延します。
  上記の時間帯には十分な余裕を取ってあります。
- **リポジトリが60日間非アクティブだとスケジュールが自動停止**します。
  Actions タブから再有効化するか、コミットを push してください。

---

## 6. 検証結果（合成データ）

`verify_logic.py` の実行例（seed=0）:

```
[Claim 2] Mean cosine similarity of B^(K) to B* vs window length:
  L plain   sub
 20 0.242 0.583     ← 短いウィンドウでは素のPCAは不安定、正則化が大きく優位
 60 0.455 0.584
180 0.714 0.584     ← 長いウィンドウでは素のPCAが追いつく（バイアス・分散トレードオフ）

[Claim 3] Long/short performance on synthetic data (annualised, 252d):
              AR  RISK    RR    MDD
MOM        -1.74 11.85 -0.15 -28.31
PCA_PLAIN  48.41 12.49  3.88  -6.88
PCA_SUB    74.04 13.69  5.41  -6.32     ← R/R 最良・MDD 最小（論文の結論と整合）
DOUBLE     47.27 14.27  3.31  -7.11
```

合成データ上でも、論文の定性的結論（**PCA_SUB がリスク調整後で最良、
正則化が短ウィンドウでの推定安定化に寄与、`B^(K)` が低ランクの最良予測 `B*`
を回復**）が再現される。実数値は乱数シードに依存する。

---

## 7. 開発情報・実装メモ

- **設計方針**: `common.py` が論文の式番号と1対1に対応する純粋関数群を提供し、
  検証スクリプトとリアルタイムスクリプトはこれを呼び出すだけにしている。
  バックテストループ（`run_backtest`）は合成データ・実価格の双方に共通。
- **相関行列**: `C_t` は標準化リターンから `(1/L) Z^T Z` で構成し、数値的に
  対称化・単位対角へ正規化している。
- **固有分解**: 対称行列なので `numpy.linalg.eigh` を使用。PCA の符号不定性は
  予測子 `B^(K) = V_J V_U^T` を取ることで（積の形で）相殺される。
- **タイミング**: シグナルは時点 t までの情報のみを使用（ウィンドウは `{t-L,…,t-1}`、
  当日ショックは `rcc_t`）、実現リターンは `roc_{J,t+1}`。ルックアヘッドはない。
- **実データの取り扱い**（`realtime_run.py`、論文4.1・4.3節に対応）:
  - 日米**双方の市場が実際に取引した営業日のみ**を使用する（各市場のETFの
    過半数に当日値が付いた日を共通営業日とみなす）。片方の市場だけが
    休場の日は除外され、リターン0が捏造されることはない。
  - XLRE（2015-10 上場）・XLC（2018-06 上場）の上場前は NaN のまま保持し、
    2010年からの履歴を切り捨てない。事前行列 `C0` は論文どおり
    **2010-01-01〜2014-12-31** のデータから推定される（期間内のデータが
    不足する場合のみ警告付きで先頭400営業日にフォールバック）。
  - シグナル計算日の推定ウィンドウにデータが揃わない米国銘柄（上場前の
    XLC など）は、その日の推定ユニバースから除外して結合PCAを実行する
    （出力に使用銘柄数を明示）。日本側17銘柄は常に必須。
  - `fetch_prices` は `yf.download(..., threads=False)` で銘柄を逐次
    取得する。yfinance は既定でスレッド並行取得を行うが、その際に
    各スレッドが共通のタイムゾーンキャッシュ（`~/.cache/py-yfinance/`
    配下の単一 sqlite ファイル）へ同時書き込みを試みることがあり、
    `sqlite3.OperationalError: database is locked` が一部銘柄だけ
    無言で失敗する形で表面化する（例外は上がらず、その銘柄のデータが
    欠損するだけ）。直近の実行分もこれが原因で
    `ValueError: Missing Japanese data` として失敗していた。逐次取得に
    加え、取得後に直近ウィンドウ（`tail(65)`）の欠損チェックを行い、
    欠損があれば例外を送出してリトライ（最大3回、指数バックオフ）する
    ことで、この一過性の失敗を自己修復する。
- **年率化**: 既定 252（日次）。論文の月次想定に合わせる場合は
  `performance_metrics(..., periods_per_year=12)`。
- **再現性**: `verify_logic.py` は `--seed` で固定可能。`realtime_run.py` の
  合成フォールバックは実行時刻ベースのシードを使い、系列は当日まで生成される
  （ライブデータと同じ日付レンジ・形状を持たせるため）。
- **通知配信**: `daily_report.py` は `realtime_run.py` の
  `get_data` / `build_prior` / `snapshot_at` をそのまま再利用するため、
  シグナルの計算経路は手動実行時と完全に同一。メール（描画・MIME組み立て
  ・SMTP）と Discord（Embed組み立て・`multipart/form-data` POST、`urllib`
  のみで実装し追加依存なし）は互いに独立したチャネルとして後段に追加して
  おり、`--channels` で選択する。合成フォールバック時はどちらのチャネル
  にも `SYNTHETIC` 警告が入り、CI のスケジュール実行では `--require-live`
  により、そもそも合成データでの配信が起きない。Discord の Webhook URL
  はそれ自体が認証情報のため、ログには常にマスクした値のみ出力する
  （`_mask_webhook`）。

### 既知の制約

- 本リポジトリは論文の**ロジック再現**が目的であり、論文の実証数値（表2）を
  bit-exact に再現するものではない（実データの取得期間・調整・欠損処理に依存）。
- ライブ実行は ETF 終値の更新タイミングに依存する。日本側の翌日 Open-to-Close は
  当然ながらシグナル生成時点では未確定（予測対象）。
- 取引コスト・スリッページ・建玉制約は考慮していない。

---

## 8. 参考

中川慧・竹本悠城・久保健治・加藤真大,
"Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA",
人工知能学会 第二種研究会資料 金融情報学研究会 SIG-FIN-036-13, pp.76–83.
