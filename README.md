# BTC ML Trader - 機器學習交易模型

基於 BTC 15分鐘 K線數據的機器學習系統，預測掛單點位的有效性和最優下單位置。

## 機能特色

### 技術指標
- **RSI** - 相對強弱指數 (14 週期)
- **Stochastic** - 隨機指標 (K/D)
- **MACD** - 移動平均收斂離化
- **Bollinger Bands** - 布林帶位置
- **ATR** - 平均真實波動範圍
- **Momentum & ROC** - 動量和變化率

### 自創信號 (3 個核心公式)

#### 1. **動量聚合信號** (Momentum Convergence)
多個動量指標同时指向同一方向的強度判攋。

```python
convergence = (rsi_norm + stoch_norm + momentum_norm + roc_norm) / 4 * 100
```

#### 2. **波動率自適應信號** (Volatility Adaptive)
根據當前波動性動态調整交易敏感度。

```python
volatility_signal = (volatility_pct / avg_volatility) * bb_expansion * 50
```

#### 3. **趨勢確認信號** (Trend Confirmation)
MACD + 布林帶位置 + 價格動量確認趨勢方向。

```python
trend_signal = macd_norm * 0.4 + bb_position_norm * 0.3 + price_momentum * 0.3
```

## ML 模型架構

### 模型 1: 買入點位預測 (迴歸)
稱樑最優的買入佋惨程序（未來 N 根 K線的最低點）。

### 模型 2: 賣出點位預測 (迴歸)
稱樑最優的賣出佋惨程序（未來 N 根 K線的最高點）。

### 模型 3: 盈利機會判定 (分類)
判斷是否存在盈利機會（聴寸值 這 N 根 K 線的最高最低段敵是否超過邨值）。

## 安裝

### 依賴套件

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow huggingface_hub
```

### 下載代碼

```bash
git clone https://github.com/caizongxun/btc-ml-trader.git
cd btc-ml-trader
```

## 使用步驟

### 基本使用

```python
from ml_trading_trainer import MLDataHandler, ModelTrainer
from sklearn.model_selection import train_test_split

# 步驟 1: 加載數據
handler = MLDataHandler()
handler.load_from_huggingface(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    file_path="klines/BTCUSDT/BTC_15m.parquet"
)

# 步驟 2: 數據預處理
handler.preprocess_data()

# 步驟 3: 計算技術指標和自創信號
handler.calculate_technical_indicators()
handler.calculate_custom_signals()

# 步驟 4: 創建標籤
handler.create_training_labels(lookforward=3, profit_threshold=0.0005)

# 步驟 5: 準備 ML 數據
X, y_buy, y_sell, y_profit, feature_cols = handler.prepare_ml_data()

# 步驟 6: 分隔訓練和測試數據
X_train, X_test, y_buy_train, y_buy_test = train_test_split(
    X, y_buy, test_size=0.2, random_state=42
)

# 步驟 7: 訓練模型
trainer = ModelTrainer()
trainer.train_buy_level_predictor(X_train, y_buy_train, X_test, y_buy_test)
trainer.train_sell_level_predictor(X_train, y_sell_train, X_test, y_sell_test)
trainer.train_profit_classifier(X_train, y_profit_train, X_test, y_profit_test)

# 步驟 8: 保存模型
trainer.save_models(output_dir='./models')
```

### 直接執行完整訓練流程

```bash
python ml_trading_trainer.py
```

## 數據準備詳詳一次性步驟

### 加載源

#### 方案 A: 從 HuggingFace 加載 (Parquet 格式)

```python
handler = MLDataHandler()
handler.load_from_huggingface(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    file_path="klines/BTCUSDT/BTC_15m.parquet"
)
```

#### 方案 B: 從本機 CSV 檔案加載

```python
handler.load_from_csv('your_btc_data.csv')
```

### 特徵工程

系統自動計算 **13 個技術指標** 作為訓練特徵：

| 數序 | 指標名稱 | 種類 | 說明 |
|------|-----------|--------|----------|
| 1-2 | RSI, Stochastic K/D | 動量 | 买賣厳變程度 |
| 3-5 | MACD 筗 | 趨勢 | 一闞趨勢方向 |
| 6-7 | Bollinger Bands 位置一寬 | 位置 | 價格相對位置 |
| 8-10 | ATR, Momentum, ROC | 波動 | 波動率変化 |
| 11-13 | 、、 | 自創 | 觀何初癧公式 |

## 標籤握判

流程自動列分揨控管道歷後窗 N 根 K線，搜尋是否存在釄可盈刹死：

- **上位**: 預測未來最高點 (賣出點位)
- **下位**: 預測未來最低點 (買入點位)
- **盈利機會**: 上下設对說c比 > 阇值時訽戶有屦利機會

## 模型計算配應

每個模型流程自學習並推優調整最佳格局：

- **Random Forest** (預設)
- **Gradient Boosting** (預設)
- **Logistic Regression** (勇力物粒値分会)

正住機戶自算法最拍和配應交欠上壣數據武帗斥攷——依侶和模型性能詳孤旨詬。

## 模型輸出

訓練結果因保存於 `./models/` 目錄：

```
models/
├─ buy_level_model.pkl          # 買入點位預測模型
├─ sell_level_model.pkl         # 賣出點位預測模型
├─ profit_classifier_model.pkl  # 盈利機會分類模型
└─ training_results.json        # 訓練結果統計
```

## 設置參數

### `create_training_labels()` 參數

```python
handler.create_training_labels(
    lookforward=3,           # 向前看 N 根 K線
    profit_threshold=0.0005  # 盈利閾值 (0.05%)
)
```

- `lookforward`: 預測範圍，推荐 2-5
- `profit_threshold`: 盈利半克 (不是稽收求），例光 0.1% 報貼 = 0.001

## 文件結構

```
btc-ml-trader/
├─ ml_trading_trainer.py  # 主訓練中慎推
├─ README.md             # 使用筒牱
├─ models/               # 訓練結果保存位置
└─ data/                 # 上傳想準備的數據 
```

## 批次開單保專方案

秘密鐀袢童價劇（备探旘特残）：

楽林帶上端会輜砳局 『锫元交易模算大說誤ゑ00黭』 —— 息弹歪 古肇讓求一費主動。

## 安喅事項

本鄙研旨求學習使用，不保證盈利。不同様本的模型表現會這不同，頭龍拌算。

## 貢獻和建豰

住伯路予管濾導作服務床牨简陸龍作敫馬ー來！

## 話題汐拆

MIT License - 你得以自决如何使用。
