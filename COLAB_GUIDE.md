# Google Colab 使用指南

## 快速開始

### 方法 1: 直接在 Colab 中打開 Notebook

點擊下方連結直接在 Google Colab 中打開：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caizongxun/btc-ml-trader/blob/main/colab_trainer.ipynb)

### 方法 2: 手動上傳

1. 訪問 [Google Colab](https://colab.research.google.com/)
2. 選擇「GitHub」標籤
3. 輸入 `caizongxun/btc-ml-trader`
4. 選擇 `colab_trainer.ipynb` 文件

## Notebook 內容說明

### 步驟總覽

| 步驟 | 功能 | 說明 |
|------|------|------|
| 0 | 環境配置 | 安裝必要的 Python 依賴套件 |
| 1-3 | 定義類別 | 定義指標計算、信號生成類別 |
| 4-5 | 數據處理 | 加載、清理、計算特徵 |
| 6 | 模型訓練 | 訓練三個預測模型 |
| 7-9 | 執行訓練 | 加載數據、預處理、訓練 |
| 10 | 實時預測 | 對測試集進行預測 |
| 11-12 | 回測驗證 | 驗證預測的準確性和收益 |
| 13-15 | 結果分析 | 可視化展示、生成報告 |

## 核心輸出

### 1. 實時預測結果

每個預測包含：

```
預測 1:
  時間: 2024-01-01 15:15:00
  現價: 42350.50
  預測買入: 42340.12345678
  預測賣出: 42365.87654321
  盈利概率: 75.32%
```

### 2. 回測結果表格

| 時間 | 現價 | 預測買入 | 預測賣出 | 實際低點 | 實際高點 | 收益率 | 結果 |
|------|------|---------|---------|---------|---------|--------|------|
| ... | ... | ... | ... | ... | ... | +0.15% | SUCCESS |

**結果說明：**
- **SUCCESS**: 預測買點和賣點都在未來 3 根 K 線內被觸及，盈利完全實現
- **PARTIAL**: 只有買點被觸及，收益為實際最高點與預測買入的差額
- **FAILED**: 預測買點未被觸及，交易失敗

### 3. 統計數據

#### 模型性能

```
【訓練階段統計】
  總樣本數: 1500
  訓練樣本: 1200
  測試樣本: 300
  特徵維度: 14

【模型性能】
  買入點位模型:
    類型: Gradient Boosting
    R² 分數: 0.7832
  
  賣出點位模型:
    類型: Random Forest
    R² 分數: 0.7645
  
  盈利判定模型:
    類型: Random Forest
    準確率: 68.50%
```

#### 回測成績

```
【回測結果】
  總交易次數: 297
  成功交易: 89 (29.93%)
  部分成功: 124 (41.75%)
  失敗交易: 84 (28.29%)

【收益統計】
  平均收益率: +0.1245%
  最大收益率: +0.8932%
  最小收益率: -0.2341%
  總累積收益: +37.01%

【預測精度】
  買入點位準確率: 71.72%
  賣出點位準確率: 67.00%
```

## 理解預測點位

### 買入點位 (Buy Level)

預測模型學習的是未來 3 根 K 線內的**最低點位**。

**含義**：
- 如果現價是 42350，預測買入是 42340
- 表示模型認為未來會跌到 42340，是買入好機會

**準確性**：
- 如果實際低點確實 ≤ 42340，則「買點命中」
- 買點命中率越高，說明模型越準

### 賣出點位 (Sell Level)

預測模型學習的是未來 3 根 K 線內的**最高點位**。

**含義**：
- 如果現價是 42350，預測賣出是 42365
- 表示模型認為未來會漲到 42365，是賣出好機會

**準確性**：
- 如果實際高點確實 ≥ 42365，則「賣點命中」
- 賣點命中率越高，說明模型越準

### 盈利概率 (Profit Probability)

分類模型預測的是買賣點位間的價差是否能超過閾值。

**含義**：
- 75% 概率表示有 3/4 的把握能盈利
- 50% 概率表示勢均力敵，風險收益比較低

## 可視化結果

Notebook 會自動生成 4 張圖表：

### 圖 1: 交易成功率分布 (圓餅圖)
- 綠色：成功交易占比
- 橙色：部分成功占比
- 紅色：失敗交易占比

### 圖 2: 收益率分布 (直方圖)
- 顯示所有交易的收益率分布
- 紅色虛線表示平均收益
- 越聚集說明模型越穩定

### 圖 3: 累積收益曲線 (折線圖)
- 紫色線表示逐次累計的收益
- 向上斜率越陡說明盈利越快
- 向下回調說明有虧損期間

### 圖 4: 預測精度對比 (柱狀圖)
- 藍色：買入點位準確率
- 紅色：賣出點位準確率
- 綠色：整體成功率

## 下載結果

### 方式 1: CSV 匯出

Notebook 會自動生成 `btc_predictions_backtest.csv`，包含：

```
timestamp,current_price,pred_buy,pred_sell,actual_low,actual_high,profit_pct,result
2024-01-01 15:15:00,42350.50,42340.12,42365.87,42335.00,42370.00,+0.1523,SUCCESS
...
```

你可以：
- 在 Colab 左側面板下載
- 導入 Excel 進一步分析
- 製作自己的交易日誌

### 方式 2: 保存到 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# 複製預測結果
import shutil
shutil.copy('btc_predictions_backtest.csv', 
            '/content/drive/My Drive/btc_predictions_backtest.csv')
```

## 常見問題

### Q1: 為什麼買點和賣點都沒有被觸及？

**A**: 這說明模型預測的點位不夠準。可能原因：
- 訓練樣本不足（建議 > 5000）
- 模型複雜度設置不當
- 市場特徵最近發生變化

**改進方法**：
- 增加訓練數據量
- 調整 `profit_threshold` 參數
- 定期重新訓練模型

### Q2: 為什麼平均收益率這麼低？

**A**: 15 分鐘周期內的波幅本身就小。可以：
- 改用 1 小時或 4 小時數據
- 調整 `lookforward` 參數（看更遠的未來）
- 增加點位預測的精度

### Q3: 能否用來實際交易？

**A**: 目前版本適合：
- ✅ 學習和研究
- ✅ 策略測試
- ✅ 指標驗證
- ❌ 直接實盤交易（需更多驗證和風控）

**建議**：
1. 先在模擬盤測試
2. 從小額開始
3. 定期更新模型
4. 設置止損止盈

### Q4: 如何優化模型？

**方法 1: 增加訓練數據**
```python
# 下載更多歷史數據
handler.load_from_huggingface(
    file_path="klines/BTCUSDT/BTC_1h.parquet"  # 改用 1h 數據
)
```

**方法 2: 調整模型參數**
```python
# 在 train_buy_level_predictor 中修改
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,    # 增加樹的數量
        max_depth=20,        # 增加樹的深度
        min_samples_split=5  # 減少分割要求
    )
}
```

**方法 3: 添加新特徵**
```python
# 在 calculate_technical_indicators 中添加
self.data['vwap'] = calculate_vwap(...)  # 成交量加權均價
self.data['obv'] = calculate_obv(...)    # 能量潮
```

## GPU 加速

Colab 免費版本有 GPU 可用，可加速訓練：

1. 點擊「執行期間」→「變更執行期間類型」
2. 選擇 GPU (Tesla T4 或更好)
3. 重新執行即可

**預期加速**：
- 無 GPU：~ 2-3 分鐘
- 有 GPU：~ 30-40 秒

## 進階用法

### 實時交易訊號

創建一個持續運行的預測器：

```python
import time
from datetime import datetime

while True:
    # 每 15 分鐘更新一次
    latest_data = fetch_latest_price()
    signal = engine.predict_signal(latest_data)
    
    if signal['profit_probability'] > 0.7:
        print(f"[{datetime.now()}] 買入信號: {signal['buy_level']}")
        # 發送交易指令...
    
    time.sleep(900)  # 15 分鐘
```

### 定時更新訓練

定期重新訓練模型以適應市場變化：

```python
# 每周末更新一次
from apscheduler.schedulers.background import BackgroundScheduler

def retrain_models():
    handler.load_from_huggingface()  # 加載最新數據
    handler.preprocess_data()
    handler.calculate_technical_indicators()
    # 重新訓練...
    trainer.save_models()

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_models, 'cron', day_of_week='6', hour=0)  # 周六午夜
scheduler.start()
```

## 支持和反饋

如有問題或改進建議，歡迎：
- 在 GitHub 提交 Issue
- Fork 並發送 Pull Request
- 在討論區分享經驗

## 免責聲明

本項目僅供教育和研究使用。過往表現不代表未來結果。

**使用本工具進行實際交易時，風險自負。**
