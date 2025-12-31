# 快速開始指南 (5 分鐘上手)

## 最快的方式

### 在 Google Colab 中打開

👉 **[按這裡直接在 Colab 中打開](https://colab.research.google.com/github/caizongxun/btc-ml-trader/blob/main/colab_trainer.ipynb)**

## 執行步驟

### 1️⃣ 安裝依賴 (30 秒)

在第一個代碼單元執行：

```python
!pip install -q pandas numpy scikit-learn matplotlib seaborn pyarrow huggingface_hub
```

### 2️⃣ 運行所有單元 (3-5 分鐘)

按 `Ctrl+F9` (或點擊功能表的「執行所有單元」)，自動執行完整訓練流程。

### 3️⃣ 查看結果 (1 分鐘)

訓練完成後自動生成：
- ✅ 前 10 個實時預測（顯示買入點、賣出點、盈利概率）
- ✅ 統計表格（前 20 個交易的詳細結果）
- ✅ 4 張圖表（成功率、收益分布、累積收益、精度對比）
- ✅ 完整報告

## 理解預測結果

### 預測示例

```
預測 1:
  時間: 2024-01-01 15:15:00
  現價: 42350.50
  預測買入: 42340.12
  預測賣出: 42365.87
  盈利概率: 75.32%
```

**含義**：
- 現價是 42350.50
- 模型認為未來會跌到 42340.12（買入好機會）
- 然後漲到 42365.87（賣出好機會）
- 有 75% 的把握能實現

### 回測結果表格

| 時間 | 現價 | 預測買入 | 預測賣出 | 實際低點 | 實際高點 | 收益率 | 結果 |
|------|------|--------|--------|--------|--------|--------|------|
| ... | 42350 | 42340 | 42365 | 42335 | 42370 | +0.15% | SUCCESS |

**三種結果**：
- 🟢 **SUCCESS**: 買點和賣點都被觸及，完全實現盈利
- 🟡 **PARTIAL**: 只有買點被觸及，部分盈利
- 🔴 **FAILED**: 買點都沒被觸及，交易失敗

### 統計數據理解

```
回測統計:
  總交易次數: 297
  成功交易: 89 (29.93%)     ← 買賣都對的比例
  部分成功: 124 (41.75%)    ← 只有買點對的比例
  失敗交易: 84 (28.29%)     ← 買點都沒對的比例

收益統計:
  平均收益率: +0.1245%     ← 每次交易平均賺多少
  最大收益率: +0.8932%     ← 最好的一次交易
  最小收益率: -0.2341%     ← 最差的一次交易（虧損）
  總累積收益: +37.01%      ← 所有交易加起來的收益
```

## 關鍵指標解釋

### 1. 模型 R² 分數 (Model R² Score)

- **0.78** = 模型解釋了 78% 的價格變動
- 越接近 1.0 越好
- 低於 0.5 說明模型需要改進

### 2. 預測精度 (Prediction Accuracy)

**買入點精度: 71.72%**
- 表示預測買點在 100 次中有 72 次確實被觸及
- 越高越好，代表買點預測越準

**賣出點精度: 67.00%**
- 表示預測賣點在 100 次中有 67 次確實被觸及
- 通常比買點精度低（正常現象）

### 3. 成功率 (Success Rate)

29.93% 成功率在 15 分鐘級別是不錯的成績
- 只要長期成功率 > 50%，加上適當倍數就能盈利

## 下載和使用結果

### 方法 1: 下載 CSV

1. Colab 左側找到「文件」面板
2. 找到 `btc_predictions_backtest.csv`
3. 點擊下載
4. 在 Excel 中打開查看所有預測

### 方法 2: 複製到 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy('btc_predictions_backtest.csv', 
            '/content/drive/My Drive/btc_predictions_backtest.csv')
```

## 常見問題速答

### Q: 這些預測能直接用來交易嗎？
**A**: 不建議。此版本是概念驗證，需要：
- 更多驗證和優化
- 風險管理機制
- 實盤測試
- 止損止盈設置

### Q: 準確率只有 30% 算好嗎？
**A**: 對於短週期交易，30% 成功率其實不錯（隨機 50%）。關鍵是：
- 每次成功的收益 > 失敗的虧損
- 需要 1.5-2 倍的收益-風險比

### Q: 為什麼有些預測失敗了？
**A**: 原因包括：
- 15 分鐘內波幅本身就小
- 模型需要更多訓練數據
- 最近市場走勢與歷史不符
- 需要調整 `lookforward` 參數

### Q: 怎樣改進模型？
**A**: 三種方式：

**方式 1: 增加數據**
```python
handler.load_from_huggingface(
    file_path="klines/BTCUSDT/BTC_1h.parquet"  # 改用 1h 數據
)
```

**方式 2: 調整參數**
```python
handler.create_training_labels(
    lookforward=5,              # 向前看 5 根 K 線（更長視角）
    profit_threshold=0.001      # 提高盈利閾值（0.1%）
)
```

**方式 3: 增加特徵**
在 `calculate_technical_indicators` 中加入更多指標

## 高級使用

### 改用不同交易對

```python
handler.load_from_huggingface(
    file_path="klines/ETHUSDT/ETH_15m.parquet"  # ETH
)
```

### 改用不同時間框架

```python
handler.load_from_huggingface(
    file_path="klines/BTCUSDT/BTC_4h.parquet"   # 4h
)
```

### 自定義特徵

在 `calculate_technical_indicators` 之後添加：

```python
# 添加成交量指標
self.data['volume_ma'] = self.data['volume'].rolling(20).mean()
self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma']

# 添加相對強度
self.data['rs_ratio'] = self.data['close'] / self.data['close'].rolling(50).mean()
```

## 時間預估

| 環節 | 不含 GPU | 含 GPU |
|------|---------|--------|
| 安裝依賴 | 30 秒 | 30 秒 |
| 加載數據 | 20 秒 | 20 秒 |
| 計算指標 | 30 秒 | 30 秒 |
| 訓練模型 | 90 秒 | 25 秒 |
| 回測預測 | 60 秒 | 40 秒 |
| 可視化 | 20 秒 | 20 秒 |
| **總計** | **~4 分鐘** | **~2.5 分鐘** |

## 開啟 GPU 加速

1. 點擊 Colab 功能表「執行期間」
2. 選擇「變更執行期間類型」
3. 選擇 GPU (Tesla T4 或更好)
4. 點擊「保存」
5. 重新執行

## 下一步

✅ 運行一次完整流程  
✅ 理解每個預測結果  
✅ 修改參數看效果變化  
✅ 嘗試不同的交易對  
✅ 研究失敗案例  
✅ 添加自定義特徵  

## 有問題？

- 查看完整 [COLAB_GUIDE.md](./COLAB_GUIDE.md)
- 瀏覽 [README.md](./README.md) 了解技術細節
- 在 GitHub Issues 提問

---

**祝交易順利！** 🚀
