# BTC ML Trading System - Colab Version
# ===================================================
# 完整的䮤易模型訓練和預測系統
# 第一步：安裝依賴
# 第二步：運行此 Python 文件

print("\n" + "="*60)
print("機器學習交易系統 - Colab 版本")
print("="*60 + "\n")

# Step 1: Install dependencies
import subprocess
import sys

print("⬼ 安裝依賴...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "pandas", "numpy", "scikit-learn", "matplotlib", 
                       "seaborn", "pyarrow", "huggingface_hub"])
print("✓ 依賴安裝完成\n")

# Step 2: Import modules
import pandas as pd
import numpy as np
from datetime import datetime
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("✓ 所有模塊已導入\n")

# Step 3: Define Technical Indicators
class TechnicalIndicators:
    """計算技術指標"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent.fillna(50), d_percent.fillna(50)
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, num_std=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_width = upper_band - lower_band
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return upper_band, lower_band, bb_width, bb_position
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean())

print("✓ 技術指標類已定義\n")

# Step 4: Define Data Handler
class MLDataHandler:
    def __init__(self):
        self.data = None
        self.scaler_features = StandardScaler()
        self.indicators = TechnicalIndicators()
    
    def create_sample_data(self, n_samples=2000):
        """生成示例数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='15T')
        
        close_prices = np.cumsum(np.random.randn(n_samples) * 0.0005) + 42000
        high_prices = close_prices + np.abs(np.random.randn(n_samples) * 0.002)
        low_prices = close_prices - np.abs(np.random.randn(n_samples) * 0.002)
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(n_samples) * 0.001,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.uniform(100, 10000, n_samples)
        })
        self.data.set_index('timestamp', inplace=True)
        print(f"✓ 生成 {n_samples} 條示例数据\n")
    
    def calculate_indicators(self):
        """計算技術指標"""
        print("⬼ 計算技術指標...")
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        self.data['rsi'] = self.indicators.calculate_rsi(close)
        self.data['stoch_k'], self.data['stoch_d'] = self.indicators.calculate_stochastic(high, low, close)
        macd_line, signal_line, histogram = self.indicators.calculate_macd(close)
        self.data['macd'] = macd_line
        self.data['macd_signal'] = signal_line
        self.data['macd_hist'] = histogram
        upper_bb, lower_bb, bb_width, bb_position = self.indicators.calculate_bollinger_bands(close)
        self.data['bb_width'] = bb_width
        self.data['bb_position'] = bb_position
        self.data['atr'] = self.indicators.calculate_atr(high, low, close)
        
        print("✓ 技術指標計算完成\n")
    
    def create_training_labels(self, lookforward=3):
        """建立訓練標籤"""
        print("⬼ 建立訓練標籤...")
        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values
        
        buy_levels = []
        sell_levels = []
        has_profit = []
        
        for i in range(len(close) - lookforward):
            future_low = low[i+1:i+1+lookforward].min()
            future_high = high[i+1:i+1+lookforward].max()
            buy_levels.append(future_low)
            sell_levels.append(future_high)
            has_profit.append(1 if (future_high - future_low) / future_low > 0.001 else 0)
        
        for _ in range(lookforward):
            buy_levels.append(np.nan)
            sell_levels.append(np.nan)
            has_profit.append(np.nan)
        
        self.data['buy_target'] = buy_levels
        self.data['sell_target'] = sell_levels
        self.data['has_profit'] = has_profit
        self.data = self.data.dropna()
        print(f"✓ 訓練標籤建立完成，保留 {len(self.data)} 樣本\n")
    
    def prepare_ml_data(self):
        """準備 ML 數據"""
        feature_cols = ['rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 
                       'macd_hist', 'bb_position', 'bb_width', 'atr']
        X = self.data[feature_cols].copy()
        y_buy = self.data['buy_target'].copy()
        y_sell = self.data['sell_target'].copy()
        y_profit = self.data['has_profit'].copy()
        
        X_scaled = self.scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        print(f"✓ 準備完成：{X_scaled.shape[0]} 樣本，{X_scaled.shape[1]} 特徵\n")
        return X_scaled, y_buy, y_sell, y_profit, feature_cols

print("✓ 數據處理類已定義\n")

# Step 5: Define Model Trainer
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_models(self, X_train, y_buy_train, y_sell_train, y_profit_train,
                    X_test, y_buy_test, y_sell_test, y_profit_test):
        print("="*60)
        print("訓練模型")
        print("="*60 + "\n")
        
        print("⬼ 訓練買入點位迴歸器...")
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1)
        model.fit(X_train, y_buy_train)
        buy_r2 = model.score(X_test, y_buy_test)
        self.models['buy'] = model
        print(f"✓ 買入點位 R²: {buy_r2:.4f}\n")
        
        print("⬼ 訓練賣出點位迴歸器...")
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1)
        model.fit(X_train, y_sell_train)
        sell_r2 = model.score(X_test, y_sell_test)
        self.models['sell'] = model
        print(f"✓ 賣出點位 R²: {sell_r2:.4f}\n")
        
        print("⬼ 訓練盈利判定分類器...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
        model.fit(X_train, y_profit_train)
        profit_acc = model.score(X_test, y_profit_test)
        self.models['profit'] = model
        print(f"✓ 盈利判定準確率: {profit_acc:.4f}\n")

print("✓ 模型訓練類已定義\n")

# Step 6: Define Predictor
class PredictionEngine:
    def __init__(self, trainer, feature_cols, scaler):
        self.trainer = trainer
        self.feature_cols = feature_cols
        self.scaler = scaler
    
    def predict(self, X_test, data_test):
        """預測造句點位和盈利概率"""
        predictions = []
        X_test_scaled = self.scaler.transform(X_test)
        
        for i in range(len(X_test_scaled)):
            features = X_test_scaled[i].reshape(1, -1)
            
            buy_pred = self.trainer.models['buy'].predict(features)[0]
            sell_pred = self.trainer.models['sell'].predict(features)[0]
            profit_prob = self.trainer.models['profit'].predict_proba(features)[0][1]
            
            predictions.append({
                'buy': buy_pred,
                'sell': sell_pred,
                'profit_prob': profit_prob
            })
        
        return predictions
    
    def backtest(self, predictions, data_test):
        """回測計算準確性"""
        results = []
        lookforward = 3
        
        for i in range(len(predictions) - lookforward):
            pred = predictions[i]
            future_data = data_test.iloc[i+1:i+1+lookforward]
            
            future_low = future_data['low'].min()
            future_high = future_data['high'].max()
            
            buy_hit = future_low <= pred['buy'] <= future_high
            sell_hit = future_low <= pred['sell'] <= future_high
            
            if buy_hit and sell_hit:
                profit = (pred['sell'] - pred['buy']) / pred['buy'] * 100
                status = 'SUCCESS'
            elif buy_hit:
                profit = (future_high - pred['buy']) / pred['buy'] * 100
                status = 'PARTIAL'
            else:
                profit = 0
                status = 'FAILED'
            
            results.append({
                'pred_buy': pred['buy'],
                'pred_sell': pred['sell'],
                'actual_low': future_low,
                'actual_high': future_high,
                'profit_pct': profit,
                'status': status,
                'profit_prob': pred['profit_prob']
            })
        
        return pd.DataFrame(results)

print("✓ 預測器類已定義\n")

# Step 7: Run complete training pipeline
print("="*60)
print("開始訓練流程")
print("="*60 + "\n")

print("步驟 1: 數據準備")
print("-" * 60)
handler = MLDataHandler()
handler.create_sample_data(n_samples=2000)
handler.calculate_indicators()
handler.create_training_labels(lookforward=3)

print("步驟 2: 準備 ML 數據")
print("-" * 60)
X, y_buy, y_sell, y_profit, feature_cols = handler.prepare_ml_data()

X_train, X_test, y_buy_train, y_buy_test = train_test_split(X, y_buy, test_size=0.2, random_state=42)
_, _, y_sell_train, y_sell_test = train_test_split(X, y_sell, test_size=0.2, random_state=42)
_, _, y_profit_train, y_profit_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)

print(f"訓練集: {len(X_train)} 樣本")
print(f"測試集: {len(X_test)} 樣本\n")

print("步驟 3: 訓練模型")
print("-" * 60)
trainer = ModelTrainer()
trainer.train_models(X_train, y_buy_train, y_sell_train, y_profit_train,
                    X_test, y_buy_test, y_sell_test, y_profit_test)

print("步驟 4: 預測和回測")
print("-" * 60)
engine = PredictionEngine(trainer, feature_cols, handler.scaler_features)
predictions = engine.predict(X_test, handler.data.loc[X_test.index])
backtest_df = engine.backtest(predictions, handler.data.loc[X_test.index])

print(f"✓ 完成 {len(predictions)} 個預測\n")

# Step 8: Display results
print("="*60)
print("回測統計")
print("="*60 + "\n")

success = (backtest_df['status'] == 'SUCCESS').sum()
partial = (backtest_df['status'] == 'PARTIAL').sum()
failed = (backtest_df['status'] == 'FAILED').sum()
total = len(backtest_df)

print(f"總交易次數: {total}")
print(f"成功交易: {success} ({success/total*100:.2f}%)")
print(f"部分成功: {partial} ({partial/total*100:.2f}%)")
print(f"失敗交易: {failed} ({failed/total*100:.2f}%)\n")

print(f"平均收益率: {backtest_df['profit_pct'].mean():+.4f}%")
print(f"最大收益率: {backtest_df['profit_pct'].max():+.4f}%")
print(f"最小收益率: {backtest_df['profit_pct'].min():+.4f}%")
print(f"總累積收益: {backtest_df['profit_pct'].sum():+.2f}%\n")

# Step 9: Show first 10 predictions
print("前 10 個預測結果:")
print("-" * 80)
for i in range(min(10, len(backtest_df))):
    print(f"預測 {i+1}:")
    print(f"  預測買入: {backtest_df.iloc[i]['pred_buy']:.2f}")
    print(f"  預測賣出: {backtest_df.iloc[i]['pred_sell']:.2f}")
    print(f"  盈利概率: {backtest_df.iloc[i]['profit_prob']*100:.2f}%")
    print(f"  結果: {backtest_df.iloc[i]['status']}")
    print(f"  收益: {backtest_df.iloc[i]['profit_pct']:+.4f}%\n")

# Step 10: Export results
output_file = 'btc_predictions_backtest.csv'
backtest_df.to_csv(output_file, index=False)
print(f"\n✓ 結果已導出到: {output_file}")
print(f"✓ 詳細預測数据帮你保存\n")

# Step 11: Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('BTC ML Trader - 回測結果分析', fontsize=16, fontweight='bold')

# Chart 1: Success distribution pie
ax1 = axes[0, 0]
sizes = [success, partial, failed]
labels = [f'成功 ({success})', f'部分 ({partial})', f'失敗 ({failed})']
colors = ['#2ecc71', '#f39c12', '#e74c3c']
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('交易結果分布', fontweight='bold')

# Chart 2: Profit distribution histogram
ax2 = axes[0, 1]
ax2.hist(backtest_df['profit_pct'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
ax2.axvline(backtest_df['profit_pct'].mean(), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('收益率 (%)')
ax2.set_ylabel('交易次數')
ax2.set_title('收益率分布', fontweight='bold')
ax2.grid(alpha=0.3)

# Chart 3: Cumulative profit
ax3 = axes[1, 0]
cumulative = backtest_df['profit_pct'].cumsum()
ax3.plot(range(len(cumulative)), cumulative.values, linewidth=2, color='#9b59b6')
ax3.fill_between(range(len(cumulative)), cumulative.values, alpha=0.3, color='#9b59b6')
ax3.set_xlabel('交易序列')
ax3.set_ylabel('累積收益 (%)')
ax3.set_title('累積收益曲線', fontweight='bold')
ax3.grid(alpha=0.3)

# Chart 4: Accuracy metrics
ax4 = axes[1, 1]
metrics = ['整體成功率', '平均收益', '最大收益']
values = [success/total*100, backtest_df['profit_pct'].mean(), backtest_df['profit_pct'].max()]
colors_bar = ['#2ecc71', '#3498db', '#e74c3c']
ax4.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_ylabel('值')
ax4.set_title('正方向指標', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('btc_backtest_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 圖表已生成\n")

print("="*60)
print("✓ 訓練流程完成！")
print("="*60)
