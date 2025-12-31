"""
ML Trading Data Exporter & Model Trainer - DEBUG VERSION
用途：從 TradingView 導出指標數據，訓練機器學習模型預測掛單點位有效性
增強版本：完整的調試信息、進度追蹤、性能監控
版本：2.0
要求：Python 3.8+, pandas, scikit-learn, matplotlib, tqdm
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import sys
import time
from collections import defaultdict

# 進度條和計時
from tqdm import tqdm

# 機器學習相關
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DEBUG UTILITIES
# ============================================================================

class DebugLogger:
    """調試日誌系統"""
    def __init__(self):
        self.logs = defaultdict(list)
        self.start_time = datetime.now()
        self.section_times = {}
    
    def log(self, level, message, section=None):
        """記錄日誌"""
        timestamp = (datetime.now() - self.start_time).total_seconds()
        formatted = f"[{timestamp:7.2f}s] {level:8s} {message}"
        print(formatted)
        if section:
            self.logs[section].append(formatted)
    
    def start_section(self, name):
        """開始計時新章節"""
        self.section_times[name] = {'start': time.time(), 'logs': []}
        self.log('INFO', f"┌─ 開始: {name} {'─'*50}")
    
    def end_section(self, name):
        """結束章節計時"""
        if name in self.section_times:
            elapsed = time.time() - self.section_times[name]['start']
            self.log('INFO', f"└─ 完成: {name} ({elapsed:.2f}s) {'─'*50}")
    
    def metric(self, name, value, format_str=".4f"):
        """記錄指標"""
        if isinstance(value, (int, float)):
            self.log('METRIC', f"{name:30s} = {value:{format_str}}")
        else:
            self.log('METRIC', f"{name:30s} = {value}")

debug = DebugLogger()

# ============================================================================
# SECTION 1: 數據準備和加載
# ============================================================================

class MLDataHandler:
    """處理交易數據的加載、清理和特徵工程"""
    
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.data = None
        self.scaler_features = StandardScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_stats = {}
    
    def load_data(self, csv_path=None):
        """加載 CSV 數據"""
        if csv_path:
            self.csv_path = csv_path
        
        try:
            debug.log('INFO', f"正在加載: {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)
            
            debug.metric("加載的行數", len(self.data))
            debug.metric("加載的列数", len(self.data.columns))
            debug.log('INFO', f"列名: {list(self.data.columns)}")
            
            # 顯示數據類型
            debug.log('INFO', "數據類型分佈:")
            for dtype, count in self.data.dtypes.value_counts().items():
                debug.log('INFO', f"  {dtype}: {count} 列")
            
            return True
        except Exception as e:
            debug.log('ERROR', f"數據加載失敗: {e}")
            return False
    
    def create_sample_data(self, n_samples=1000):
        """生成示例數據用於測試"""
        debug.start_section("生成示例數據")
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='15T')
        
        # 基礎指標
        rsi = np.random.uniform(20, 80, n_samples)
        stoch = np.random.uniform(10, 90, n_samples)
        macd = np.random.uniform(-1, 1, n_samples)
        bb_width = np.random.uniform(0.5, 3.0, n_samples)
        
        # 自創指標
        momentum = np.random.uniform(-100, 100, n_samples)
        volatility = np.random.uniform(0, 100, n_samples)
        rsi_convergence = np.random.uniform(0, 100, n_samples)
        composite = np.random.uniform(-100, 100, n_samples)
        
        # 價格和掛單
        close_price = np.random.uniform(1.0500, 1.1000, n_samples)
        buy_pending = close_price - np.random.uniform(0.001, 0.010, n_samples)
        sell_pending = close_price + np.random.uniform(0.001, 0.010, n_samples)
        
        # 標籤
        order_filled = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        order_profitable = np.where(
            (order_filled == 1) & (momentum > 0), 1, 
            np.where((order_filled == 1) & (momentum < 0), np.random.choice([0, 1]), 0)
        )
        
        self.data = pd.DataFrame({
            'datetime': dates,
            'rsi': rsi,
            'stoch': stoch,
            'macd': macd,
            'bb_width': bb_width,
            'momentum_score': momentum,
            'volatility_index': volatility,
            'rsi_convergence': rsi_convergence,
            'composite_signal': composite,
            'close_price': close_price,
            'buy_pending_level': buy_pending,
            'sell_pending_level': sell_pending,
            'order_filled': order_filled,
            'order_profitable': order_profitable
        })
        
        debug.metric("生成樣本數", n_samples)
        debug.metric("數據時間跨度", f"{(dates[-1] - dates[0]).days} 天")
        debug.end_section("生成示例數據")
        return self.data
    
    def preprocess_data(self):
        """數據清理和預處理"""
        debug.start_section("數據預處理")
        
        initial_rows = len(self.data)
        debug.metric("初始行數", initial_rows)
        
        # 移除 NaN 值
        nan_count_before = self.data.isna().sum().sum()
        self.data = self.data.dropna()
        nan_count_after = self.data.isna().sum().sum()
        
        debug.metric("移除的 NaN 值", initial_rows - len(self.data))
        debug.metric("剩餘行數", len(self.data))
        
        # 移除異常值
        outlier_count = 0
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < Q1 - 1.5 * IQR) | 
                       (self.data[col] > Q3 + 1.5 * IQR))
            outlier_count += outliers.sum()
            self.data = self.data[~outliers]
        
        debug.metric("移除的異常值行", outlier_count)
        debug.metric("最終行數", len(self.data))
        debug.end_section("數據預處理")
        return self.data
    
    def feature_engineering(self):
        """特徵工程：創建新特徵"""
        debug.start_section("特徵工程")
        
        initial_cols = len(self.data.columns)
        
        # 動量變化率
        self.data['momentum_change'] = self.data['momentum_score'].diff().fillna(0)
        debug.log('INFO', "✓ 添加: momentum_change")
        
        # RSI 斜率
        self.data['rsi_slope'] = self.data['rsi'].diff().fillna(0)
        debug.log('INFO', "✓ 添加: rsi_slope")
        
        # Volatility Expansion Ratio
        self.data['volatility_ratio'] = (
            self.data['bb_width'] / self.data['bb_width'].rolling(20).mean()
        ).fillna(1)
        debug.log('INFO', "✓ 添加: volatility_ratio")
        
        # Price distance from pending levels
        self.data['price_to_buy_distance'] = (
            (self.data['buy_pending_level'] - self.data['close_price']) / 
            self.data['close_price']
        )
        debug.log('INFO', "✓ 添加: price_to_buy_distance")
        
        self.data['price_to_sell_distance'] = (
            (self.data['sell_pending_level'] - self.data['close_price']) / 
            self.data['close_price']
        )
        debug.log('INFO', "✓ 添加: price_to_sell_distance")
        
        # 掛單成功率
        self.data['order_fill_rate'] = (
            self.data['order_filled'].rolling(50).mean()
        ).fillna(0)
        debug.log('INFO', "✓ 添加: order_fill_rate")
        
        # 掛單盈利率
        self.data['order_profit_rate'] = (
            self.data['order_profitable'].rolling(50).mean()
        ).fillna(0)
        debug.log('INFO', "✓ 添加: order_profit_rate")
        
        new_cols = len(self.data.columns) - initial_cols
        debug.metric("新增特徵數", new_cols)
        debug.metric("總特徵數", len(self.data.columns))
        debug.end_section("特徵工程")
        return self.data
    
    def prepare_ml_data(self):
        """準備 ML 訓練所需的特徵和標籤"""
        debug.start_section("準備 ML 數據")
        
        feature_cols = [
            'rsi', 'stoch', 'macd', 'bb_width', 
            'momentum_score', 'volatility_index', 'rsi_convergence', 
            'composite_signal', 'momentum_change', 'rsi_slope',
            'volatility_ratio', 'price_to_buy_distance', 
            'price_to_sell_distance', 'order_fill_rate', 'order_profit_rate'
        ]
        
        # 包括两个分类標签和两个回归求解目標
        target_cols = ['order_filled', 'order_profitable', 'buy_pending_level', 'sell_pending_level']
        
        debug.metric("特徵数", len(feature_cols))
        debug.metric("目標数", len(target_cols))
        
        X = self.data[feature_cols].copy()
        y = self.data[target_cols].copy()
        
        debug.log('INFO', "特徵統計:")
        for col in feature_cols[:5]:  # 顯示前 5 個
            stats = X[col].describe()
            debug.log('INFO', f"  {col:30s} μ={stats['mean']:8.4f} σ={stats['std']:8.4f} [{stats['min']:8.4f}, {stats['max']:8.4f}]")
        
        # 標準化
        debug.log('INFO', "正在標準化特徵...")
        X_scaled = self.scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        debug.metric("準備的樣本数", X_scaled.shape[0])
        debug.metric("最終特徵数", X_scaled.shape[1])
        debug.end_section("準備 ML 數據")
        
        return X_scaled, y, feature_cols

# ============================================================================
# SECTION 2: 模型訓練
# ============================================================================

class MLModelTrainer:
    """機器學習模型訓練器 - 增強版本"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.training_history = defaultdict(list)
    
    def train_order_filled_classifier(self, X_train, y_train, X_test, y_test):
        """訓練掛單填充概率分類模型"""
        debug.start_section("訓練: 掛單是否被觸發")
        debug.metric("訓練樣本", len(X_train))
        debug.metric("測試樣本", len(X_test))
        
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=self.random_state)
        }
        
        best_model = None
        best_score = 0
        results = []
        
        for name, clf in tqdm(classifiers.items(), desc="訓練分類器", leave=False):
            debug.log('INFO', f"\n  訓練: {name}")
            
            start = time.time()
            clf.fit(X_train, y_train['order_filled'])
            train_time = time.time() - start
            
            train_score = clf.score(X_train, y_train['order_filled'])
            test_score = clf.score(X_test, y_test['order_filled'])
            
            debug.metric("  訓練時間", f"{train_time:.2f}s")
            debug.metric("  訓練準確率", train_score, ".4f")
            debug.metric("  測試準確率", test_score, ".4f")
            
            results.append({'model': name, 'train': train_score, 'test': test_score})
            
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        debug.log('INFO', f"\n最佳模型: {best_name} (準確率: {best_score:.4f})")
        
        self.models['order_filled'] = best_model
        self.results['order_filled'] = {
            'model': best_name,
            'test_accuracy': best_score,
            'all_results': results
        }
        
        debug.end_section("訓練: 掛單是否被觸發")
        return best_model
    
    def train_order_profitable_classifier(self, X_train, y_train, X_test, y_test):
        """訓練掛單盈利概率分類模型"""
        debug.start_section("訓練: 掛單是否盈利")
        debug.metric("訓練樣本", len(X_train))
        debug.metric("測試樣本", len(X_test))
        
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=self.random_state)
        }
        
        best_model = None
        best_score = 0
        results = []
        
        for name, clf in tqdm(classifiers.items(), desc="訓練分類器", leave=False):
            debug.log('INFO', f"\n  訓練: {name}")
            
            start = time.time()
            clf.fit(X_train, y_train['order_profitable'])
            train_time = time.time() - start
            
            train_score = clf.score(X_train, y_train['order_profitable'])
            test_score = clf.score(X_test, y_test['order_profitable'])
            
            debug.metric("  訓練時間", f"{train_time:.2f}s")
            debug.metric("  訓練準確率", train_score, ".4f")
            debug.metric("  測試準確率", test_score, ".4f")
            
            results.append({'model': name, 'train': train_score, 'test': test_score})
            
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        debug.log('INFO', f"\n最佳模型: {best_name} (準確率: {best_score:.4f})")
        
        self.models['order_profitable'] = best_model
        self.results['order_profitable'] = {
            'model': best_name,
            'test_accuracy': best_score,
            'all_results': results
        }
        
        debug.end_section("訓練: 掛單是否盈利")
        return best_model
    
    def train_pending_level_regressor(self, X_train, y_train, X_test, y_test):
        """訓練掛單點位預測模型"""
        debug.start_section("訓練: 掛單點位預測 (迴歸)")
        debug.metric("訓練樣本", len(X_train))
        debug.metric("測試樣本", len(X_test))
        
        regressors = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        }
        
        # 訓練買入點位
        debug.log('INFO', "\n  訓練買入掛單點位...")
        best_buy_model = None
        best_buy_r2 = -np.inf
        buy_results = []
        
        for name, reg in tqdm(regressors.items(), desc="買入迴歸", leave=False):
            start = time.time()
            reg.fit(X_train, y_train['buy_pending_level'])
            train_time = time.time() - start
            
            train_r2 = reg.score(X_train, y_train['buy_pending_level'])
            test_r2 = reg.score(X_test, y_test['buy_pending_level'])
            test_mae = mean_absolute_error(y_test['buy_pending_level'], reg.predict(X_test))
            
            debug.log('INFO', f"\n    {name}:")
            debug.metric("      訓練時間", f"{train_time:.2f}s")
            debug.metric("      訓練 R²", train_r2, ".4f")
            debug.metric("      測試 R²", test_r2, ".4f")
            debug.metric("      MAE", test_mae, ".6f")
            
            buy_results.append({'model': name, 'train_r2': train_r2, 'test_r2': test_r2, 'mae': test_mae})
            
            if test_r2 > best_buy_r2:
                best_buy_r2 = test_r2
                best_buy_model = reg
                best_buy_name = name
        
        # 訓練賣出點位
        debug.log('INFO', "\n  訓練賣出掛單點位...")
        best_sell_model = None
        best_sell_r2 = -np.inf
        sell_results = []
        
        for name, reg in tqdm(regressors.items(), desc="賣出迴歸", leave=False):
            start = time.time()
            reg.fit(X_train, y_train['sell_pending_level'])
            train_time = time.time() - start
            
            train_r2 = reg.score(X_train, y_train['sell_pending_level'])
            test_r2 = reg.score(X_test, y_test['sell_pending_level'])
            test_mae = mean_absolute_error(y_test['sell_pending_level'], reg.predict(X_test))
            
            debug.log('INFO', f"\n    {name}:")
            debug.metric("      訓練時間", f"{train_time:.2f}s")
            debug.metric("      訓練 R²", train_r2, ".4f")
            debug.metric("      測試 R²", test_r2, ".4f")
            debug.metric("      MAE", test_mae, ".6f")
            
            sell_results.append({'model': name, 'train_r2': train_r2, 'test_r2': test_r2, 'mae': test_mae})
            
            if test_r2 > best_sell_r2:
                best_sell_r2 = test_r2
                best_sell_model = reg
                best_sell_name = name
        
        debug.log('INFO', f"\n最佳買入模型: {best_buy_name} (R²: {best_buy_r2:.4f})")
        debug.log('INFO', f"最佳賣出模型: {best_sell_name} (R²: {best_sell_r2:.4f})")
        
        self.models['buy_pending_level'] = best_buy_model
        self.models['sell_pending_level'] = best_sell_model
        
        self.results['pending_levels'] = {
            'buy': {'model': best_buy_name, 'r2': best_buy_r2, 'results': buy_results},
            'sell': {'model': best_sell_name, 'r2': best_sell_r2, 'results': sell_results}
        }
        
        debug.end_section("訓練: 掛單點位預測")
        return best_buy_model, best_sell_model
    
    def visualize_results(self, X_test, y_test, feature_cols):
        """可視化訓練結果和特徵重要性"""
        debug.start_section("可視化結果")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Trading Model - 訓練結果分析', fontsize=16, fontweight='bold')
        
        # 1. 模型準確率對比
        ax1 = axes[0, 0]
        models_info = []
        accuracy_values = []
        
        if 'order_filled' in self.results:
            models_info.append('Order Filled')
            accuracy_values.append(self.results['order_filled']['test_accuracy'])
        
        if 'order_profitable' in self.results:
            models_info.append('Order Profitable')
            accuracy_values.append(self.results['order_profitable']['test_accuracy'])
        
        ax1.bar(models_info, accuracy_values, color=['#3498db', '#2ecc71'])
        ax1.set_ylabel('準確率')
        ax1.set_title('分類模型準確率')
        ax1.set_ylim([0, 1])
        for i, v in enumerate(accuracy_values):
            ax1.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        # 2. 迴歸模型 R² 對比
        ax2 = axes[0, 1]
        if 'pending_levels' in self.results:
            pending = self.results['pending_levels']
            r2_models = ['Buy', 'Sell']
            r2_values = [pending['buy']['r2'], pending['sell']['r2']]
            
            ax2.bar(r2_models, r2_values, color=['#e74c3c', '#f39c12'])
            ax2.set_ylabel('R² 分数')
            ax2.set_title('迴歸模型 R² 分数')
            ax2.set_ylim([0, 1])
            for i, v in enumerate(r2_values):
                ax2.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        # 3. 特徵重要性
        ax3 = axes[1, 0]
        if 'order_filled' in self.models and hasattr(self.models['order_filled'], 'feature_importances_'):
            importances = self.models['order_filled'].feature_importances_
            indices = np.argsort(importances)[-10:]  # 前 10 個
            top_features = [feature_cols[i] for i in indices]
            top_importances = importances[indices]
            
            ax3.barh(top_features, top_importances, color='#9b59b6')
            ax3.set_xlabel('重要性')
            ax3.set_title('Top 10 特徵重要性 (掛單觸發)')
        
        # 4. 訓練統計
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = "訓練統計\n" + "="*40 + "\n"
        stats_text += f"訓練樣本: {len(X_test)}\n"
        stats_text += f"測試樣本: {len(X_test)}\n"
        stats_text += f"特徵数: {len(feature_cols)}\n\n"
        stats_text += "分類模型準確率:\n"
        
        if 'order_filled' in self.results:
            stats_text += f"  - 掛單觸發: {self.results['order_filled']['test_accuracy']:.4f}\n"
        if 'order_profitable' in self.results:
            stats_text += f"  - 掛單盈利: {self.results['order_profitable']['test_accuracy']:.4f}\n"
        
        stats_text += "\n迴歸模型 R²:\n"
        if 'pending_levels' in self.results:
            stats_text += f"  - 買入點位: {self.results['pending_levels']['buy']['r2']:.4f}\n"
            stats_text += f"  - 賣出點位: {self.results['pending_levels']['sell']['r2']:.4f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        output_file = 'ml_training_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        debug.log('INFO', f"圖表已保存: {output_file}")
        debug.end_section("可視化結果")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """主訓練流程"""
    
    print("\n" + "="*70)
    print("ML Trading Data Exporter & Model Trainer - DEBUG 版本")
    print("="*70 + "\n")
    
    # 步驟 1: 數據加載
    debug.start_section("步驟 1: 數據加載和準備")
    
    handler = MLDataHandler()
    handler.create_sample_data(n_samples=1500)
    handler.preprocess_data()
    handler.feature_engineering()
    
    debug.end_section("步驟 1: 數據加載和準備")
    
    # 步驟 2: 準備 ML 數據
    debug.start_section("步驟 2: 準備 ML 數據")
    
    X, y, feature_cols = handler.prepare_ml_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    debug.metric("訓練集大小", len(X_train))
    debug.metric("測試集大小", len(X_test))
    debug.metric("訓練/測試比", f"{len(X_train)/len(X_test):.2f}")
    debug.end_section("步驟 2: 準備 ML 數據")
    
    # 步驟 3: 訓練模型
    debug.start_section("步驟 3: 模型訓練")
    
    trainer = MLModelTrainer()
    trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
    trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
    trainer.train_pending_level_regressor(X_train, y_train, X_test, y_test)
    
    debug.end_section("步驟 3: 模型訓練")
    
    # 步驟 4: 可視化
    debug.start_section("步驟 4: 結果可視化")
    trainer.visualize_results(X_test, y_test, feature_cols)
    debug.end_section("步驟 4: 結果可視化")
    
    # 完成
    print("\n" + "="*70)
    print("✓ 訓練流程完成")
    print("="*70 + "\n")
    
    return trainer, handler

if __name__ == "__main__":
    trainer, handler = main()
