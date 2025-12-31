#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC ML Trading System - Colab Version
完整的交易模型訓練和預測系統 - 支持實際 CSV 數據
步驟 1：安裝依賴
步驟 2：運行此 Python 文件
"""

import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("機器學習交易系統 - Colab 版本")
print("="*60 + "\n")

# Install dependencies
print("⬼ 安裝依賴...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "pandas", "numpy", "scikit-learn", "matplotlib"])
print("✓ 依賴安裝完成\n")

# Technical Indicators
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

# Data Handler
class MLDataHandler:
    def __init__(self):
        self.data = None
        self.scaler_features = StandardScaler()
        self.indicators = TechnicalIndicators()
    
    def load_csv_data(self, csv_path):
        """從 CSV 加載實際交易數據"""
        try:
            df = pd.read_csv(csv_path)
            
            # 自動檢測列名
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'time' in col_lower or 'date' in col_lower:
                    col_mapping['timestamp'] = col
                elif col_lower in ['open', 'o']:
                    col_mapping['open'] = col
                elif col_lower in ['high', 'h']:
                    col_mapping['high'] = col
                elif col_lower in ['low', 'l']:
                    col_mapping['low'] = col
                elif col_lower in ['close', 'c']:
                    col_mapping['close'] = col
                elif col_lower in ['volume', 'vol', 'v']:
                    col_mapping['volume'] = col
            
            df = df.rename(columns=col_mapping)
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"缺少必要列: {col}")
            
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=required_cols)
            self.data = df[required_cols].reset_index(drop=True)
            
            print(f"✓ 成功加載 CSV 數據: {len(self.data)} 根 K線")
            print(f"  價格範圍: {self.data['close'].min():.2f} ~ {self.data['close'].max():.2f}\n")
            return True
            
        except Exception as e:
            print(f"✗ 加載失敗: {e}\n")
            return False
    
    def create_sample_data(self, n_samples=2000):
        """生成示例數據"""
        np.random.seed(42)
        close_prices = np.cumsum(np.random.randn(n_samples) * 0.0005) + 42000
        high_prices = close_prices + np.abs(np.random.randn(n_samples) * 0.002)
        low_prices = close_prices - np.abs(np.random.randn(n_samples) * 0.002)
        
        self.data = pd.DataFrame({
            'open': close_prices + np.random.randn(n_samples) * 0.001,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.uniform(100, 10000, n_samples)
        })
        print(f"✓ 生成 {n_samples} 條示例数整数\n")
    
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
    
    def create_training_labels(self, lookforward=5):
        """創建訓練標籤 - 預測未來低點和高點"""
        print("⬼ 建立訓練標籤...")
        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values
        
        buy_levels = []
        sell_levels = []
        has_profit = []
        
        # 為每個 K線計算未來 lookforward 根中的最低和最高點
        for i in range(len(close) - lookforward):
            future_low = low[i+1:i+1+lookforward].min()
            future_high = high[i+1:i+1+lookforward].max()
            buy_levels.append(future_low)
            sell_levels.append(future_high)
            has_profit.append(1 if (future_high - future_low) / future_low > 0.005 else 0)
        
        # 的残余的行填为 NaN
        for _ in range(lookforward):
            buy_levels.append(np.nan)
            sell_levels.append(np.nan)
            has_profit.append(np.nan)
        
        self.data['buy_target'] = buy_levels
        self.data['sell_target'] = sell_levels
        self.data['has_profit'] = has_profit
        
        initial_len = len(self.data)
        self.data = self.data.dropna()
        print(f"✓ 訓練標籤建立完成，保留 {len(self.data)} 樣本\n")
    
    def prepare_ml_data(self):
        """準備 ML 数整數據"""
        feature_cols = ['rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 
                       'macd_hist', 'bb_position', 'bb_width', 'atr']
        
        X = self.data[feature_cols].copy()
        y_buy = self.data['buy_target'].copy()
        y_sell = self.data['sell_target'].copy()
        y_profit = self.data['has_profit'].copy()
        
        X_scaled = self.scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        print(f"✓ 準備完成：{len(X_scaled)} 樣本，{len(feature_cols)} 特徵\n")
        return X_scaled, y_buy, y_sell, y_profit, feature_cols

print("✓ 數據處理類已定義\n")

# Model Trainer
class ModelTrainer:
    def __init__(self):
        self.models = {}
    
    def train_models(self, X_train, y_buy_train, y_sell_train, y_profit_train,
                    X_test, y_buy_test, y_sell_test, y_profit_test):
        print("="*60)
        print("訓練模型")
        print("="*60 + "\n")
        
        print("⬼ 訓練買入點位迴歸器...")
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        model.fit(X_train, y_buy_train)
        buy_r2 = model.score(X_test, y_buy_test)
        self.models['buy'] = model
        print(f"✓ 買入點位 R²: {buy_r2:.4f}\n")
        
        print("⬼ 訓練賣出點位迴歸器...")
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        model.fit(X_train, y_sell_train)
        sell_r2 = model.score(X_test, y_sell_test)
        self.models['sell'] = model
        print(f"✓ 賣出點位 R²: {sell_r2:.4f}\n")
        
        print("⬼ 訓練盈利判定分類器...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_train, y_profit_train)
        profit_acc = model.score(X_test, y_profit_test)
        self.models['profit'] = model
        print(f"✓ 盈利判定準確率: {profit_acc:.4f}\n")

print("✓ 模型訓練類已定義\n")

# Prediction Engine - FIXED
class PredictionEngine:
    def __init__(self, trainer, feature_cols, data_original, data_processed):
        """
        Args:
            trainer: 訓練模式
            feature_cols: 特徵了欄
            data_original: 原始未被 dropna 的數據
            data_processed: 經過理理的数整数整何
        """
        self.trainer = trainer
        self.feature_cols = feature_cols
        self.data_original = data_original.reset_index(drop=True)
        self.data_processed = data_processed.reset_index(drop=True)
    
    def predict_and_backtest(self, X_test, train_test_indices):
        """預測並回測 - 使用原始數據的未來值進行驗證"""
        results = []
        lookforward = 5
        
        # 獲取訓練集地索引
        train_indices, test_indices = train_test_indices
        
        for test_idx_in_processed, original_idx in enumerate(test_indices):
            # 原始数整中的索引
            if original_idx + lookforward >= len(self.data_original):
                continue
            
            try:
                # 使用处理後的數據中的特徵進行預測
                features = X_test.iloc[test_idx_in_processed].values.reshape(1, -1)
                
                buy_pred = float(self.trainer.models['buy'].predict(features)[0])
                sell_pred = float(self.trainer.models['sell'].predict(features)[0])
                
                if hasattr(self.trainer.models['profit'], 'predict_proba'):
                    profit_prob = float(self.trainer.models['profit'].predict_proba(features)[0][1])
                else:
                    profit_prob = float(self.trainer.models['profit'].predict(features)[0])
                
                if buy_pred >= sell_pred:
                    continue
                
                # 獲取原始数整中的當前價格和未來数整
                current_price = self.data_original.iloc[original_idx]['close']
                
                # 取未來 lookforward 根中的最低和最高點
                future_data = self.data_original.iloc[original_idx+1:original_idx+1+lookforward]
                if len(future_data) == 0:
                    continue
                
                future_low = future_data['low'].min()
                future_high = future_data['high'].max()
                
                # 判斷預測是否被觸發
                buy_hit = future_low <= buy_pred
                sell_hit = future_high >= sell_pred
                
                # 計算收益
                if buy_hit and sell_hit:
                    profit = (sell_pred - buy_pred) / buy_pred * 100
                    status = 'SUCCESS'
                elif buy_hit:
                    profit = max(0, (future_high - buy_pred) / buy_pred * 100)
                    status = 'PARTIAL'
                else:
                    profit = (future_high - current_price) / current_price * 100
                    status = 'FAILED'
                
                results.append({
                    'idx': original_idx,
                    'current_price': float(current_price),
                    'pred_buy': buy_pred,
                    'pred_sell': sell_pred,
                    'actual_low': float(future_low),
                    'actual_high': float(future_high),
                    'profit_pct': profit,
                    'status': status,
                    'profit_prob': profit_prob
                })
            
            except Exception as e:
                continue
        
        return pd.DataFrame(results) if results else pd.DataFrame()

print("✓ 預測器類已定義\n")

# ===== MAIN PIPELINE =====
print("="*60)
print("開始訓練流程")
print("="*60 + "\n")

print("步驟 1: 數據準備")
print("-" * 60)

handler = MLDataHandler()

# Try loading CSV
csv_files = [
    '/content/drive/MyDrive/BTCUSDT_15m.csv',
    'BTCUSDT_15m.csv',
    '/tmp/BTCUSDT_15m.csv',
]

data_loaded = False
for csv_path in csv_files:
    if handler.load_csv_data(csv_path):
        data_loaded = True
        break

if not data_loaded:
    print("未找到 CSV 文件，使用示例数整数")
    handler.create_sample_data(n_samples=2000)
    print()

# Keep reference to original data BEFORE indicators
data_before_indicators = handler.data.copy()

handler.calculate_indicators()
handler.create_training_labels(lookforward=5)

print("步驟 2: 準備 ML 数整數據")
print("-" * 60)
X, y_buy, y_sell, y_profit, feature_cols = handler.prepare_ml_data()

# Split with index tracking
X_train, X_test, y_buy_train, y_buy_test, train_idx, test_idx = train_test_split(
    X, y_buy, np.arange(len(X)), test_size=0.2, random_state=42
)
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
print("-" * 60 + "\n")

engine = PredictionEngine(trainer, feature_cols, data_before_indicators, handler.data)
backtest_df = engine.predict_and_backtest(X_test, (train_idx, test_idx))

if len(backtest_df) > 0:
    print(f"✓ 完成 {len(backtest_df)} 個預測\n")
    
    # Results
    print("="*60)
    print("回測統計")
    print("="*60 + "\n")
    
    success = (backtest_df['status'] == 'SUCCESS').sum()
    partial = (backtest_df['status'] == 'PARTIAL').sum()
    failed = (backtest_df['status'] == 'FAILED').sum()
    total = len(backtest_df)
    
    print(f"總交易次数: {total}")
    print(f"成功交易: {success} ({success/total*100:.2f}%)")
    print(f"部分成功: {partial} ({partial/total*100:.2f}%)")
    print(f"失敗交易: {failed} ({failed/total*100:.2f}%)\n")
    
    print(f"平均收益率: {backtest_df['profit_pct'].mean():+.4f}%")
    print(f"最大收益率: {backtest_df['profit_pct'].max():+.4f}%")
    print(f"最小收益率: {backtest_df['profit_pct'].min():+.4f}%")
    print(f"總累積收益: {backtest_df['profit_pct'].sum():+.2f}%\n")
    
    # Show sample predictions
    print("前 5 個預測結果:")
    print("-" * 80)
    for i in range(min(5, len(backtest_df))):
        row = backtest_df.iloc[i]
        print(f"預測 {i+1}: 現價 {row['current_price']:.2f} | 買 {row['pred_buy']:.2f} | 賣 {row['pred_sell']:.2f} | 收益 {row['profit_pct']:+.2f}% | {row['status']}")
    print()
    
    # Export
    output_file = 'btc_predictions_backtest.csv'
    backtest_df.to_csv(output_file, index=False)
    print(f"✓ 結果已導出到: {output_file}\n")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BTC ML Trader - 回測結果分析', fontsize=16, fontweight='bold')
    
    # Chart 1
    ax1 = axes[0, 0]
    sizes = [success, partial, failed]
    labels = [f'成功 ({success})', f'部分 ({partial})', f'失敗 ({failed})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('交易結果分布', fontweight='bold')
    
    # Chart 2
    ax2 = axes[0, 1]
    ax2.hist(backtest_df['profit_pct'], bins=30, color='#3498db', edgecolor='black')
    ax2.axvline(backtest_df['profit_pct'].mean(), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('收益率 (%)')
    ax2.set_ylabel('交易次数')
    ax2.set_title('收益率分布', fontweight='bold')
    
    # Chart 3
    ax3 = axes[1, 0]
    cumulative = backtest_df['profit_pct'].cumsum()
    ax3.plot(cumulative.values, linewidth=2, color='#9b59b6')
    ax3.fill_between(range(len(cumulative)), cumulative.values, alpha=0.3, color='#9b59b6')
    ax3.set_xlabel('交易序列')
    ax3.set_ylabel('累積收益 (%)')
    ax3.set_title('累積收益曲線', fontweight='bold')
    
    # Chart 4
    ax4 = axes[1, 1]
    metrics = ['成功率', '平均收益', '最大收益']
    values = [success/total*100, backtest_df['profit_pct'].mean(), backtest_df['profit_pct'].max()]
    colors_bar = ['#2ecc71', '#3498db', '#e74c3c']
    ax4.bar(metrics, values, color=colors_bar)
    ax4.set_ylabel('值')
    ax4.set_title('正向指標', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('btc_backtest_results.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表已生成: btc_backtest_results.png\n")
    
    print("="*60)
    print("✓ 訓練流程完成！")
    print("="*60)
else:
    print("⚠️ 無法生成足夠的預測結果。")
