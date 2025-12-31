"""
ML Trading Data Trainer - BTC 15min K-line Model
用途：從 HuggingFace 導入 Parquet 數據，訓練機器學習模型預測掛單點位有效性
版本：2.0 - 支持 HuggingFace 數據源
要求：Python 3.8+, pandas, scikit-learn, matplotlib, pyarrow, huggingface_hub
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 機器學習相關
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 數據源相關
try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("提示：安裝 huggingface_hub 以支持 HuggingFace 數據源")
    print("  pip install huggingface_hub")

# ============================================================================
# SECTION 1: 技術指標計算引擎
# ============================================================================

class TechnicalIndicators:
    """計算技術指標"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """計算 RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """計算隨機指標 (Stochastic Oscillator)"""
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent.fillna(50), d_percent.fillna(50)
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """計算 MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, num_std=2):
        """計算布林帶 (Bollinger Bands)"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_width = upper_band - lower_band
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return upper_band, lower_band, bb_width, bb_position
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """計算 ATR (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean())
    
    @staticmethod
    def calculate_momentum(prices, period=10):
        """計算動量 (Momentum)"""
        momentum = prices.diff(period)
        return momentum.fillna(0)
    
    @staticmethod
    def calculate_roc(prices, period=12):
        """計算變化率 (Rate of Change)"""
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc.fillna(0)

# ============================================================================
# SECTION 2: 自創指標 (Custom Signals)
# ============================================================================

class CustomSignals:
    """計算自創交易信號"""
    
    @staticmethod
    def momentum_convergence_signal(rsi, stoch_k, momentum, roc):
        """
        指標 1: 動量聚合信號
        多個動量指標同時指向同一方向的強度
        """
        # 標準化各指標到 0-100 範圍
        rsi_norm = (rsi - 30) / 40  # RSI 30-70 映射到 0-1
        stoch_norm = stoch_k / 100
        momentum_norm = (momentum + 100) / 200  # -100 to 100 映射到 0-1
        roc_norm = (roc + 20) / 40  # -20% to 20% 映射到 0-1
        
        # 計算聚合強度
        convergence = (rsi_norm + stoch_norm + momentum_norm + roc_norm) / 4 * 100
        convergence = convergence.clip(0, 100)
        
        return convergence
    
    @staticmethod
    def volatility_adaptive_signal(atr, bb_width, close):
        """
        指標 2: 波動率自適應信號
        基於當前波動性調整交易敏感度
        """
        # 計算波動率百分比
        volatility_pct = (atr / close) * 100
        
        # 標準化 BB 寬度
        bb_width_sma = bb_width.rolling(window=20).mean()
        bb_expansion = (bb_width / bb_width_sma).fillna(1)
        
        # 組合信號：波動率越高，信號越強
        volatility_signal = (volatility_pct / volatility_pct.rolling(20).mean().fillna(1)) * bb_expansion * 50
        volatility_signal = volatility_signal.clip(0, 100)
        
        return volatility_signal
    
    @staticmethod
    def trend_confirmation_signal(macd_line, bb_position, close, volume=None):
        """
        指標 3: 趨勢確認信號
        MACD + 布林帶位置 + 價格動量確認
        """
        # MACD 方向 (0-100)
        macd_norm = (macd_line / abs(macd_line).rolling(20).mean().fillna(1)) * 50 + 50
        macd_norm = macd_norm.clip(0, 100)
        
        # BB 位置 (0-100)
        bb_position_norm = bb_position * 100
        
        # 價格動量
        price_change = close.pct_change() * 100
        price_momentum = (price_change / price_change.rolling(20).std().fillna(1)) * 25 + 50
        price_momentum = price_momentum.clip(0, 100)
        
        # 組合確認信號
        trend_signal = (macd_norm * 0.4 + bb_position_norm * 0.3 + price_momentum * 0.3)
        
        return trend_signal

# ============================================================================
# SECTION 3: 數據處理
# ============================================================================

class MLDataHandler:
    """處理交易數據的加載、清理和特徵工程"""
    
    def __init__(self):
        self.data = None
        self.scaler_features = StandardScaler()
        self.scaler_price = MinMaxScaler()
        self.indicators = TechnicalIndicators()
        self.signals = CustomSignals()
    
    def load_from_huggingface(self, repo_id="zongowo111/v2-crypto-ohlcv-data", 
                             file_path="klines/BTCUSDT/BTC_15m.parquet"):
        """從 HuggingFace 加載 Parquet 數據"""
        if not HAS_HF:
            print("✗ 錯誤：需要安裝 huggingface_hub")
            print("  執行: pip install huggingface_hub")
            return False
        
        try:
            print(f"⏳ 從 HuggingFace 下載數據...")
            print(f"  Repository: {repo_id}")
            print(f"  File: {file_path}")
            
            # 下載文件
            parquet_file = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir="./cache"
            )
            
            # 讀取 Parquet 文件
            self.data = pd.read_parquet(parquet_file)
            
            # 標準化列名
            self.data.columns = self.data.columns.str.lower()
            
            # 確保索引是時間格式
            if not isinstance(self.data.index, pd.DatetimeIndex):
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data = self.data.set_index('timestamp')
                elif 'time' in self.data.columns:
                    self.data['time'] = pd.to_datetime(self.data['time'])
                    self.data = self.data.set_index('time')
            
            self.data.sort_index(inplace=True)
            
            print(f"✓ 成功加載 {len(self.data)} 根 K線數據")
            print(f"  列名: {list(self.data.columns)}")
            print(f"  時間範圍: {self.data.index[0]} ~ {self.data.index[-1]}")
            print(f"  數據大小: {self.data.memory_usage().sum() / 1024**2:.2f} MB")
            
            return True
        
        except Exception as e:
            print(f"✗ 加載失敗: {e}")
            return False
    
    def load_from_csv(self, csv_path):
        """從 CSV 文件加載數據"""
        try:
            self.data = pd.read_csv(csv_path)
            self.data.columns = self.data.columns.str.lower()
            print(f"✓ 成功加載 {len(self.data)} 根 K線數據")
            return True
        except Exception as e:
            print(f"✗ 加載失敗: {e}")
            return False
    
    def preprocess_data(self):
        """數據清理和預處理"""
        # 移除 NaN 值
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        print(f"  移除 NaN 值: {initial_rows - len(self.data)} 行")
        
        # 移除重複行
        self.data = self.data.drop_duplicates()
        
        # 移除異常值（基於 volume 和 price）
        for col in ['close', 'volume']:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (self.data[col] < Q1 - 3 * IQR) | (self.data[col] > Q3 + 3 * IQR)
                self.data = self.data[~outliers]
        
        print(f"✓ 數據清理完成，保留 {len(self.data)} 行數據")
        return self.data
    
    def calculate_technical_indicators(self):
        """計算所有技術指標"""
        print("⏳ 計算技術指標...")
        
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data.get('volume', pd.Series(np.ones(len(self.data))))
        
        # 基本指標
        self.data['rsi'] = self.indicators.calculate_rsi(close, period=14)
        self.data['stoch_k'], self.data['stoch_d'] = self.indicators.calculate_stochastic(high, low, close)
        
        macd_line, signal_line, histogram = self.indicators.calculate_macd(close)
        self.data['macd'] = macd_line
        self.data['macd_signal'] = signal_line
        self.data['macd_hist'] = histogram
        
        upper_bb, lower_bb, bb_width, bb_position = self.indicators.calculate_bollinger_bands(close)
        self.data['bb_upper'] = upper_bb
        self.data['bb_lower'] = lower_bb
        self.data['bb_width'] = bb_width
        self.data['bb_position'] = bb_position
        
        self.data['atr'] = self.indicators.calculate_atr(high, low, close)
        self.data['momentum'] = self.indicators.calculate_momentum(close)
        self.data['roc'] = self.indicators.calculate_roc(close)
        
        print("✓ 基本指標計算完成")
        return self.data
    
    def calculate_custom_signals(self):
        """計算自創信號"""
        print("⏳ 計算自創信號...")
        
        # 指標 1: 動量聚合
        self.data['momentum_convergence'] = self.signals.momentum_convergence_signal(
            self.data['rsi'],
            self.data['stoch_k'],
            self.data['momentum'],
            self.data['roc']
        )
        
        # 指標 2: 波動率自適應
        self.data['volatility_adaptive'] = self.signals.volatility_adaptive_signal(
            self.data['atr'],
            self.data['bb_width'],
            self.data['close']
        )
        
        # 指標 3: 趨勢確認
        self.data['trend_confirmation'] = self.signals.trend_confirmation_signal(
            self.data['macd'],
            self.data['bb_position'],
            self.data['close']
        )
        
        print("✓ 自創信號計算完成")
        return self.data
    
    def create_training_labels(self, lookforward=3, profit_threshold=0.0005):
        """
        創建訓練標籤
        
        Args:
            lookforward: 向前看 N 根 K線
            profit_threshold: 盈利閾值 (0.0005 = 0.05%)
        """
        print("⏳ 創建訓練標籤...")
        
        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values
        
        # 預測買入點位 (未來向下觸及的最低點)
        buy_levels = []
        # 預測賣出點位 (未來向上觸及的最高點)
        sell_levels = []
        # 是否有盈利機會
        has_profit = []
        
        for i in range(len(close) - lookforward):
            future_low = low[i+1:i+1+lookforward].min()
            future_high = high[i+1:i+1+lookforward].max()
            current_price = close[i]
            
            buy_levels.append(future_low)
            sell_levels.append(future_high)
            
            # 判斷是否有盈利機會
            profit_opp = (future_high - future_low) / current_price > profit_threshold * 2
            has_profit.append(1 if profit_opp else 0)
        
        # 填充最後幾行
        for _ in range(lookforward):
            buy_levels.append(np.nan)
            sell_levels.append(np.nan)
            has_profit.append(np.nan)
        
        self.data['buy_level_target'] = buy_levels
        self.data['sell_level_target'] = sell_levels
        self.data['has_profit'] = has_profit
        
        # 移除包含 NaN 的行
        self.data = self.data.dropna()
        
        print(f"✓ 訓練標籤創建完成，保留 {len(self.data)} 個樣本")
        return self.data
    
    def prepare_ml_data(self):
        """準備 ML 訓練所需的特徵和標籤"""
        feature_cols = [
            'rsi', 'stoch_k', 'stoch_d', 
            'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'bb_width',
            'atr', 'momentum', 'roc',
            'momentum_convergence', 'volatility_adaptive', 'trend_confirmation'
        ]
        
        X = self.data[feature_cols].copy()
        y_buy = self.data['buy_level_target'].copy()
        y_sell = self.data['sell_level_target'].copy()
        y_profit = self.data['has_profit'].copy()
        
        # 標準化特徵
        X_scaled = self.scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        print(f"✓ 準備完成：{X_scaled.shape[0]} 樣本，{X_scaled.shape[1]} 特徵")
        
        return X_scaled, y_buy, y_sell, y_profit, feature_cols

# ============================================================================
# SECTION 4: 模型訓練
# ============================================================================

class ModelTrainer:
    """機器學習模型訓練器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_buy_level_predictor(self, X_train, y_train, X_test, y_test):
        """訓練買入點位預測模型"""
        print("\n" + "="*60)
        print("訓練模型 1: 買入點位預測 (迴歸)")
        print("="*60)
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=self.random_state
            )
        }
        
        best_model = None
        best_r2 = -np.inf
        
        for name, model in models.items():
            print(f"\n  訓練 {name}...")
            model.fit(X_train, y_train)
            
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
            test_mae = mean_absolute_error(y_test, model.predict(X_test))
            test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            
            print(f"    訓練 R²: {train_r2:.4f}")
            print(f"    測試 R²: {test_r2:.4f}")
            print(f"    MAE: {test_mae:.8f}")
            print(f"    RMSE: {test_rmse:.8f}")
            
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
                best_name = name
        
        print(f"\n  ✓ 最佳模型: {best_name} (R²: {best_r2:.4f})")
        
        self.models['buy_level'] = best_model
        self.results['buy_level'] = {
            'model': best_name,
            'r2': best_r2,
            'feature_importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
        }
        
        return best_model
    
    def train_sell_level_predictor(self, X_train, y_train, X_test, y_test):
        """訓練賣出點位預測模型"""
        print("\n" + "="*60)
        print("訓練模型 2: 賣出點位預測 (迴歸)")
        print("="*60)
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=self.random_state
            )
        }
        
        best_model = None
        best_r2 = -np.inf
        
        for name, model in models.items():
            print(f"\n  訓練 {name}...")
            model.fit(X_train, y_train)
            
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
            test_mae = mean_absolute_error(y_test, model.predict(X_test))
            test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            
            print(f"    訓練 R²: {train_r2:.4f}")
            print(f"    測試 R²: {test_r2:.4f}")
            print(f"    MAE: {test_mae:.8f}")
            print(f"    RMSE: {test_rmse:.8f}")
            
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
                best_name = name
        
        print(f"\n  ✓ 最佳模型: {best_name} (R²: {best_r2:.4f})")
        
        self.models['sell_level'] = best_model
        self.results['sell_level'] = {
            'model': best_name,
            'r2': best_r2,
            'feature_importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
        }
        
        return best_model
    
    def train_profit_classifier(self, X_train, y_train, X_test, y_test):
        """訓練盈利機會分類模型"""
        print("\n" + "="*60)
        print("訓練模型 3: 盈利機會判定 (分類)")
        print("="*60)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
            )
        }
        
        best_model = None
        best_acc = 0
        
        for name, model in models.items():
            print(f"\n  訓練 {name}...")
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            print(f"    訓練準確率: {train_acc:.4f}")
            print(f"    測試準確率: {test_acc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model
                best_name = name
        
        print(f"\n  ✓ 最佳模型: {best_name} (準確率: {best_acc:.4f})")
        
        self.models['profit_classifier'] = best_model
        self.results['profit_classifier'] = {
            'model': best_name,
            'accuracy': best_acc
        }
        
        return best_model
    
    def save_models(self, output_dir='./models'):
        """保存訓練的模型"""
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ 模型已保存: {model_path}")
        
        results_path = f"{output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ 結果已保存: {results_path}")

# ============================================================================
# SECTION 5: 主執行程序
# ============================================================================

def main():
    """主訓練流程"""
    
    print("\n" + "="*60)
    print("BTC K線數據 ML 訓練系統 v2.0")
    print("="*60 + "\n")
    
    # 步驟 1: 加載數據
    print("步驟 1: 加載數據")
    print("-" * 60)
    
    handler = MLDataHandler()
    
    # 嘗試從 HuggingFace 加載
    success = handler.load_from_huggingface(
        repo_id="zongowo111/v2-crypto-ohlcv-data",
        file_path="klines/BTCUSDT/BTC_15m.parquet"
    )
    
    if not success:
        print("\n⚠️  無法從 HuggingFace 加載，請檢查網路連接或文件路徑")
        return
    
    # 步驟 2: 數據預處理
    print("\n步驟 2: 數據預處理")
    print("-" * 60)
    
    handler.preprocess_data()
    
    # 步驟 3: 計算指標
    print("\n步驟 3: 計算技術指標")
    print("-" * 60)
    
    handler.calculate_technical_indicators()
    handler.calculate_custom_signals()
    
    # 步驟 4: 創建標籤
    print("\n步驟 4: 創建訓練標籤")
    print("-" * 60)
    
    handler.create_training_labels(lookforward=3, profit_threshold=0.0005)
    
    # 步驟 5: 準備 ML 數據
    print("\n步驟 5: 準備 ML 數據")
    print("-" * 60)
    
    X, y_buy, y_sell, y_profit, feature_cols = handler.prepare_ml_data()
    
    X_train, X_test, y_buy_train, y_buy_test = train_test_split(
        X, y_buy, test_size=0.2, random_state=42
    )
    _, _, y_sell_train, y_sell_test = train_test_split(
        X, y_sell, test_size=0.2, random_state=42
    )
    _, _, y_profit_train, y_profit_test = train_test_split(
        X, y_profit, test_size=0.2, random_state=42
    )
    
    print(f"訓練集: {len(X_train)} 樣本")
    print(f"測試集: {len(X_test)} 樣本")
    
    # 步驟 6: 訓練模型
    print("\n步驟 6: 訓練模型")
    print("-" * 60)
    
    trainer = ModelTrainer()
    
    trainer.train_buy_level_predictor(X_train, y_buy_train, X_test, y_buy_test)
    trainer.train_sell_level_predictor(X_train, y_sell_train, X_test, y_sell_test)
    trainer.train_profit_classifier(X_train, y_profit_train, X_test, y_profit_test)
    
    # 步驟 7: 保存模型
    print("\n步驟 7: 保存模型")
    print("-" * 60)
    
    trainer.save_models()
    
    print("\n" + "="*60)
    print("✓ 訓練完成！")
    print("="*60)
    
    return handler, trainer

if __name__ == "__main__":
    main()
