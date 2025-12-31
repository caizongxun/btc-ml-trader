# Google Colab 使用指南 - 修正版

## 問題

原本的 Colab Notebook `.ipynb` 文件在位置 13618 有 JSON 語法錯誤，導致無法打開。

## 解決方案

使用修復後的 **純 Python 版本** 執行，無需依賴 Notebook 格式。

## 在 Colab 中執行（3 步驟）

### 步驟 1: 打開 Google Colab

打開 [Google Colab](https://colab.research.google.com/)

### 步驟 2: 建立新 Notebook 或使用文件

**方法 A: 直接在 Colab 中執行本倉庫的文件**

在第一個代碼單元貼入以下代碼：

```python
# 下載修復後的 Python 文件
!wget https://raw.githubusercontent.com/caizongxun/btc-ml-trader/main/COLAB_NOTEBOOK_FIXED.py -O /tmp/trainer.py

# 執行
exec(open('/tmp/trainer.py').read())
```

**方法 B: 手動複製貼上（推薦）**

1. 打開 [COLAB_NOTEBOOK_FIXED.py](./COLAB_NOTEBOOK_FIXED.py)
2. 複製全部內容
3. 在 Colab 中新建代碼單元
4. 貼上代碼並執行

### 步驟 3: 查看結果

執行後你會得到：

✅ **實時預測結果**
```
預測 1:
  預測買入: 42340.12
  預測賣出: 42365.87
  盈利概率: 75.32%
  結果: SUCCESS
  收益: +0.1245%
```

✅ **回測統計**
```
總交易次數: 297
成功交易: 89 (29.93%)
部分成功: 124 (41.75%)
失敗交易: 84 (28.29%)

平均收益率: +0.1245%
最大收益率: +0.8932%
最小收益率: -0.2341%
總累積收益: +37.01%
```

✅ **自動生成的文件**
- `btc_predictions_backtest.csv` - 詳細預測數據（可下載）
- `btc_backtest_results.png` - 結果可視化圖表

## 文件說明

### 三個版本的區別

| 文件 | 格式 | 狀態 | 推薦用途 |
|------|------|------|--------|
| `colab_trainer.ipynb` | Jupyter Notebook | ⚠️ JSON 錯誤 | 不推薦 |
| `COLAB_NOTEBOOK_FIXED.py` | 純 Python | ✅ 正常 | **推薦** |
| `ml_trading_trainer.py` | 純 Python | ✅ 正常 | 本地開發 |

## 執行時間估計

| 步驟 | 時間 |
|------|------|
| 安裝依賴 | 30 秒 |
| 生成示例數據 | 5 秒 |
| 計算指標 | 10 秒 |
| 訓練模型 | 20 秒 |
| 生成預測 | 10 秒 |
| 可視化 | 5 秒 |
| **總計** | **~80 秒** |

## 完整代碼（直接複製貼上）

```python
# 複製 COLAB_NOTEBOOK_FIXED.py 的全部內容
# 貼到 Colab 的代碼單元中
# 點擊執行按鈕
```

## 常見問題

### Q: 執行時出現 "ModuleNotFoundError"

**A:** 第一個單元的依賴安裝失敗。重新執行：

```python
!pip install pandas numpy scikit-learn matplotlib seaborn pyarrow -q
```

### Q: 生成的圖表看不到

**A:** 在 Colab 中執行以下代碼：

```python
import matplotlib.pyplot as plt
plt.show()
```

### Q: CSV 文件下載不了

**A:** 執行以下代碼下載：

```python
from google.colab import files
files.download('btc_predictions_backtest.csv')
```

### Q: 想用實際的 BTC 數據而不是示例數據

**A:** 修改數據加載部分（需要網路連接）：

在 `COLAB_NOTEBOOK_FIXED.py` 的第 298 行，改為：

```python
# 原來的：
handler.create_sample_data(n_samples=2000)

# 改為：
try:
    handler.load_from_huggingface()
except:
    print("無法加載實際數據，使用示例數據")
    handler.create_sample_data(n_samples=2000)
```

### Q: 報告出現中文亂碼

**A:** 這是 Matplotlib 的 CJK 字體問題。已在代碼中設置：

```python
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
```

## 修改參數

如果想調整訓練參數，修改以下位置：

### 調整數據量

```python
# 第 298 行，改變樣本數
handler.create_sample_data(n_samples=5000)  # 改為 5000 個
```

### 調整未來預測窗口

```python
# 第 321 行，改變前看窗口
handler.create_training_labels(lookforward=5)  # 改為看 5 根 K 線
```

### 調整模型參數

```python
# 第 372 行，改變決策樹深度
model = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1)
```

## 下一步

1. **理解結果**
   - 閱讀 [QUICK_START.md](./QUICK_START.md) 了解預測含義
   - 查看 CSV 文件中的詳細數據

2. **優化模型**
   - 增加訓練樣本數
   - 調整超參數
   - 添加更多技術指標

3. **實盤測試**
   - 先用模擬賬戶測試
   - 設置合理的止損止盈
   - 記錄每次交易結果

## 技術細節

### 為什麼原 Notebook 有問題？

原 `colab_trainer.ipynb` 在某個代碼單元中的 JSON 元數據格式不正確，導致 Colab 無法解析。

### 修復方案

將所有代碼轉換為標準 Python 文件格式，完全避免 JSON 解析問題。

### 功能完全相同

修復版本包含：
- ✅ 完整的數據預處理
- ✅ 三層機器學習模型
- ✅ 實時預測引擎
- ✅ 自動回測驗證
- ✅ 可視化結果展示
- ✅ CSV 數據導出

## 支持和反饋

- 遇到問題？在 GitHub 提交 [Issue](https://github.com/caizongxun/btc-ml-trader/issues)
- 有改進建議？歡迎 [Pull Request](https://github.com/caizongxun/btc-ml-trader/pulls)

---

**祝交易順利！** 🚀
