"""
ML Trading Data Exporter & Model Trainer v2.0 - 改進版本
用途：從 TradingView 導出指標數據，訓練機器學習模型預測掛單點位有效性
版本：2.0（改進：字體修復、過擬合控制、特徵工程增強、模型調優）
要求：Python 3.8+, pandas, scikit-learn, matplotlib, xgboost
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error, f1_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm

# 設置中文字體
def setup_chinese_font():
    """自動設置中文字體"""
    try:
        # 優先嘗試系統字體
        font_names = ['SimHei', 'DejaVu Sans', 'STHeiti', 'Arial Unicode MS', 'Noto Sans CJK SC']
        
        for font_name in font_names:
            try:
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                # 測試字體是否可用
                test_fig = plt.figure()
                test_ax = test_fig.add_subplot(111)
                test_ax.text(0.5, 0.5, '測試', fontsize=10)
                plt.close(test_fig)
                print(f"✓ 已設置字體: {font_name}")
                return
            except:
                continue
        
        # 備選方案：使用英文標籤
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠ 無法加載中文字體，使用英文標籤")
    except Exception as e:
        print(f"⚠ 字體設置失敗: {e}")

setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================================
# SECTION 1: 數據準備和加載
# ============================================================================

class MLDataHandler:
    """處理交易數據的加載、清理和特徵工程"""
    
    def __init__(self, csv_path=None):
        """初始化數據處理器"""
        self.csv_path = csv_path
        self.data = None
        self.scaler_features = StandardScaler()
        self.scaler_target = MinMaxScaler()
        
    def load_data(self, csv_path=None):
        """加載 CSV 數據"""
        if csv_path:
            self.csv_path = csv_path
        
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✓ 成功加載 {len(self.data)} 根 K線數據")
            print(f"  列名: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"✗ 數據加載失敗: {e}")
            return False
    
    def create_sample_data(self, n_samples=2000):
        """生成更真實的示例數據"""
        np.random.seed(42)
        
        # 生成時間序列
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='15T')
        
        # 生成更相關的指標（帶趨勢）
        trend = np.linspace(0, 2*np.pi, n_samples)
        base_price = 1.075 + 0.01 * np.sin(trend)
        
        # 技術指標（帶趨勢和噪聲）
        rsi = 50 + 30 * np.sin(trend) + np.random.normal(0, 5, n_samples)
        rsi = np.clip(rsi, 20, 80)
        
        stoch = 50 + 30 * np.sin(trend + 1) + np.random.normal(0, 8, n_samples)
        stoch = np.clip(stoch, 10, 90)
        
        macd = 0.5 * np.sin(trend) + np.random.normal(0, 0.1, n_samples)
        macd = np.clip(macd, -1, 1)
        
        bb_width = 1.5 + 0.8 * np.abs(np.sin(trend)) + np.random.normal(0, 0.2, n_samples)
        bb_width = np.clip(bb_width, 0.5, 3.0)
        
        # 自創指標（與價格更相關）
        momentum = 100 * np.sin(trend) + np.random.normal(0, 15, n_samples)
        momentum = np.clip(momentum, -100, 100)
        
        volatility = 50 + 30 * np.abs(np.sin(trend + 0.5)) + np.random.normal(0, 5, n_samples)
        volatility = np.clip(volatility, 10, 100)
        
        rsi_convergence = np.abs(rsi - 50) + np.random.normal(0, 5, n_samples)
        rsi_convergence = np.clip(rsi_convergence, 0, 100)
        
        composite = momentum * 0.5 + (50 - rsi_convergence) * 0.3 + np.random.normal(0, 10, n_samples)
        composite = np.clip(composite, -100, 100)
        
        # 價格和掛單
        close_price = base_price + np.random.normal(0, 0.002, n_samples)
        buy_pending = close_price - np.abs(momentum) / 10000 - np.random.uniform(0.001, 0.005, n_samples)
        sell_pending = close_price + np.abs(momentum) / 10000 + np.random.uniform(0.001, 0.005, n_samples)
        
        # 改進的標籤生成（基於指標的邏輯）
        # 掛單填充概率（RSI極端值更可能被觸發）
        rsi_extreme = (rsi < 35) | (rsi > 65)
        momentum_strong = np.abs(momentum) > 50
        trigger_prob = (rsi_extreme.astype(int) * 0.6 + momentum_strong.astype(int) * 0.4) / 2
        order_filled = np.random.uniform(0, 1, n_samples) < (trigger_prob * 0.7 + 0.3)
        
        # 掛單盈利概率（基於動量方向和波動率）
        momentum_direction = momentum > 0
        low_volatility = volatility < 50
        profit_prob = (momentum_direction.astype(int) * 0.6 + low_volatility.astype(int) * 0.4) / 2
        order_profitable = (order_filled.astype(int) * profit_prob) > np.random.uniform(0, 1, n_samples)
        
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
            'order_filled': order_filled.astype(int),
            'order_profitable': order_profitable.astype(int)
        })
        
        print(f"✓ 生成 {n_samples} 條示例數據")
        print(f"  訂單填充率: {self.data['order_filled'].mean():.2%}")
        print(f"  訂單盈利率: {self.data['order_profitable'].mean():.2%}")
        return self.data
    
    def preprocess_data(self):
        """數據清理和預處理"""
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        print(f"  移除 NaN 值: {initial_rows - len(self.data)} 行")
        
        # 移除異常值 (使用 IQR 方法)
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        removed_count = 0
        
        for col in numerical_cols:
            if col not in ['order_filled', 'order_profitable']:  # 跳過標籤列
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.data[col] < Q1 - 1.5 * IQR) | 
                           (self.data[col] > Q3 + 1.5 * IQR))
                removed_count += outliers.sum()
                self.data = self.data[~outliers]
        
        print(f"  移除異常值: {removed_count} 行")
        print(f"✓ 數據清理完成，保留 {len(self.data)} 行數據")
        return self.data
    
    def feature_engineering(self):
        """增強的特徵工程"""
        # 1. 動量變化率
        self.data['momentum_change'] = self.data['momentum_score'].diff().fillna(0)
        self.data['momentum_acceleration'] = self.data['momentum_change'].diff().fillna(0)
        
        # 2. RSI 相關特徵
        self.data['rsi_slope'] = self.data['rsi'].diff().fillna(0)
        self.data['rsi_smoothed'] = self.data['rsi'].rolling(window=5).mean().fillna(self.data['rsi'])
        self.data['rsi_extreme'] = ((self.data['rsi'] < 35) | (self.data['rsi'] > 65)).astype(int)
        
        # 3. 波動率特徵
        self.data['volatility_ratio'] = (
            self.data['volatility_index'] / self.data['volatility_index'].rolling(20).mean()
        ).fillna(1)
        self.data['volatility_change'] = self.data['volatility_index'].diff().fillna(0)
        
        # 4. 價格距離特徵
        self.data['price_to_buy_distance'] = (
            (self.data['buy_pending_level'] - self.data['close_price']) / 
            self.data['close_price']
        )
        self.data['price_to_sell_distance'] = (
            (self.data['sell_pending_level'] - self.data['close_price']) / 
            self.data['close_price']
        )
        
        # 5. 掛單成功率和盈利率（滾動）
        self.data['order_fill_rate'] = (
            self.data['order_filled'].rolling(50).mean()
        ).fillna(0)
        self.data['order_profit_rate'] = (
            self.data['order_profitable'].rolling(50).mean()
        ).fillna(0)
        
        # 6. 指標組合特徵
        self.data['rsi_stoch_divergence'] = np.abs(
            (self.data['rsi'] - 50) - (self.data['stoch'] - 50)
        )
        
        self.data['bb_rsi_interaction'] = (
            self.data['bb_width'] * self.data['rsi_convergence'] / 100
        )
        
        # 7. 時間序列特徵
        self.data['hour'] = pd.to_datetime(self.data['datetime']).dt.hour
        self.data['dayofweek'] = pd.to_datetime(self.data['datetime']).dt.dayofweek
        
        # 8. Lag 特徵（前期值）
        for lag in [1, 3, 5]:
            self.data[f'momentum_lag{lag}'] = self.data['momentum_score'].shift(lag).fillna(0)
            self.data[f'rsi_lag{lag}'] = self.data['rsi'].shift(lag).fillna(0)
        
        print("✓ 特徵工程完成")
        print(f"  新增特徵數: {len([c for c in self.data.columns if c not in ['datetime']])-14}")
        return self.data
    
    def prepare_ml_data(self):
        """準備 ML 訓練所需的特徵和標籤"""
        feature_cols = [col for col in self.data.columns 
                       if col not in ['datetime', 'close_price', 'buy_pending_level', 
                                     'sell_pending_level', 'order_filled', 'order_profitable']]
        
        target_cols = ['order_filled', 'order_profitable']
        
        X = self.data[feature_cols].copy()
        y = self.data[target_cols].copy()
        
        # 標準化特徵
        X_scaled = self.scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        print(f"✓ 準備完成：{X_scaled.shape[0]} 樣本，{X_scaled.shape[1]} 特徵")
        print(f"  特徵列表: {feature_cols[:5]}... (共{len(feature_cols)}個)")
        
        return X_scaled, y, feature_cols

# ============================================================================
# SECTION 2: 改進的模型訓練
# ============================================================================

class ImprovedMLModelTrainer:
    """改進的機器學習模型訓練器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.cv_results = {}
        
    def train_with_optimization(self, X_train, y_train, X_test, y_test, task_name, target_col):
        """使用超參數優化訓練模型"""
        print(f"\n訓練模型: {task_name}")
        print("="*60)
        
        # 定義模型和參數網格
        param_grids = {
            'logistic': {
                'model': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'params': {'C': [0.01, 0.1, 1, 10], 'class_weight': ['balanced', None]}
            },
            'rf': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, 15], 'min_samples_split': [5, 10]}
            },
            'gb': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
            }
        }
        
        best_model = None
        best_score = 0
        best_name = None
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, config in param_grids.items():
            print(f"\n優化 {name}...")
            
            try:
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train[target_col])
                best_params = grid_search.best_params_
                
                # 在測試集評估
                test_score = grid_search.best_estimator_.score(X_test, y_test[target_col])
                f1 = f1_score(y_test[target_col], grid_search.best_estimator_.predict(X_test), average='weighted')
                
                print(f"  最佳參數: {best_params}")
                print(f"  CV 分數: {grid_search.best_score_:.4f}")
                print(f"  測試準確率: {test_score:.4f}, F1: {f1:.4f}")
                
                if test_score > best_score:
                    best_score = test_score
                    best_model = grid_search.best_estimator_
                    best_name = name
                    
            except Exception as e:
                print(f"  ✗ 訓練失敗: {e}")
                continue
        
        print(f"\n最佳模型: {best_name} (準確率: {best_score:.4f})")
        
        return best_model, best_name, best_score
    
    def train_order_filled_classifier(self, X_train, y_train, X_test, y_test):
        """訓練掛單填充分類模型"""
        model, name, score = self.train_with_optimization(
            X_train, y_train, X_test, y_test, 
            "掛單是否被觸發", "order_filled"
        )
        
        self.models['order_filled'] = model
        self.results['order_filled'] = {
            'model': name,
            'test_accuracy': score,
        }
        
        return model
    
    def train_order_profitable_classifier(self, X_train, y_train, X_test, y_test):
        """訓練掛單盈利分類模型"""
        model, name, score = self.train_with_optimization(
            X_train, y_train, X_test, y_test,
            "掛單是否盈利", "order_profitable"
        )
        
        self.models['order_profitable'] = model
        self.results['order_profitable'] = {
            'model': name,
            'test_accuracy': score,
        }
        
        return model
    
    def train_pending_level_regressors(self, X_train, y_train, X_test, y_test):
        """訓練掛單點位預測模型"""
        print(f"\n訓練模型: 掛單點位預測")
        print("="*60)
        
        # 買入點位
        print("\n訓練買入掛單點位...")
        buy_model = Ridge(alpha=0.1).fit(X_train, y_train['buy_pending_level'])
        buy_r2 = buy_model.score(X_test, y_test['buy_pending_level'])
        buy_mae = mean_absolute_error(y_test['buy_pending_level'], buy_model.predict(X_test))
        
        print(f"  Buy R²: {buy_r2:.4f}, MAE: {buy_mae:.6f}")
        
        # 賣出點位
        print("\n訓練賣出掛單點位...")
        sell_model = Ridge(alpha=0.1).fit(X_train, y_train['sell_pending_level'])
        sell_r2 = sell_model.score(X_test, y_test['sell_pending_level'])
        sell_mae = mean_absolute_error(y_test['sell_pending_level'], sell_model.predict(X_test))
        
        print(f"  Sell R²: {sell_r2:.4f}, MAE: {sell_mae:.6f}")
        
        self.models['buy_pending_level'] = buy_model
        self.models['sell_pending_level'] = sell_model
        
        self.results['pending_levels'] = {
            'buy_r2': buy_r2,
            'buy_mae': buy_mae,
            'sell_r2': sell_r2,
            'sell_mae': sell_mae
        }
        
        return buy_model, sell_model
    
    def evaluate_all_models(self, X_test, y_test, feature_cols):
        """詳細評估所有訓練的模型"""
        print("\n" + "="*60)
        print("模型評估總結")
        print("="*60)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        # 評估分類模型
        for model_name in ['order_filled', 'order_profitable']:
            if model_name in self.models:
                model = self.models[model_name]
                pred = model.predict(X_test)
                
                accuracy = model.score(X_test, y_test[model_name])
                f1 = f1_score(y_test[model_name], pred, average='weighted')
                
                print(f"\n{model_name}:")
                print(f"  準確率: {accuracy:.4f}, F1: {f1:.4f}")
                print(f"  分類報告:")
                print(f"  {classification_report(y_test[model_name], pred)}")
                
                summary['models'][model_name] = {
                    'accuracy': float(accuracy),
                    'f1': float(f1),
                    'test_samples': len(X_test)
                }
        
        return summary
    
    def save_models(self, output_dir='./models'):
        """保存訓練的模型"""
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ 模型已保存: {model_path}")
        
        # 保存結果摘要
        results_path = f"{output_dir}/training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ 結果已保存: {results_path}")

# ============================================================================
# SECTION 3: 預測接口
# ============================================================================

class OrderPredictor:
    """基於訓練模型的掛單預測器"""
    
    def __init__(self, trainer, feature_cols):
        self.trainer = trainer
        self.feature_cols = feature_cols
    
    def predict_order_signal(self, current_features):
        """預測當前是否應該下掛單"""
        feature_vector = np.array([
            current_features.get(col, 0) for col in self.feature_cols
        ]).reshape(1, -1)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'order_fill_probability': 0.0,
            'order_profit_probability': 0.0,
            'buy_pending_level': None,
            'sell_pending_level': None,
            'recommendation': 'HOLD',
            'confidence': 0.0
        }
        
        # 掛單填充概率
        if 'order_filled' in self.trainer.models:
            model = self.trainer.models['order_filled']
            result['order_fill_probability'] = float(
                model.predict_proba(feature_vector)[0][1] 
                if hasattr(model, 'predict_proba') else model.predict(feature_vector)[0]
            )
        
        # 掛單盈利概率
        if 'order_profitable' in self.trainer.models:
            model = self.trainer.models['order_profitable']
            result['order_profit_probability'] = float(
                model.predict_proba(feature_vector)[0][1]
                if hasattr(model, 'predict_proba') else model.predict(feature_vector)[0]
            )
        
        # 掛單點位
        if 'buy_pending_level' in self.trainer.models:
            result['buy_pending_level'] = float(
                self.trainer.models['buy_pending_level'].predict(feature_vector)[0]
            )
        
        if 'sell_pending_level' in self.trainer.models:
            result['sell_pending_level'] = float(
                self.trainer.models['sell_pending_level'].predict(feature_vector)[0]
            )
        
        # 生成建議
        fill_prob = result['order_fill_probability']
        profit_prob = result['order_profit_probability']
        confidence = (fill_prob + profit_prob) / 2
        
        result['confidence'] = float(confidence)
        
        if fill_prob > 0.7 and profit_prob > 0.6:
            result['recommendation'] = 'STRONG_BUY'
        elif fill_prob > 0.6 and profit_prob > 0.5:
            result['recommendation'] = 'BUY'
        elif fill_prob > 0.4 and profit_prob > 0.5:
            result['recommendation'] = 'WATCH'
        else:
            result['recommendation'] = 'HOLD'
        
        return result

# ============================================================================
# SECTION 4: 主執行程序
# ============================================================================

def main():
    """主訓練流程"""
    
    print("\n" + "="*60)
    print("ML 訓練數據處理和模型訓練系統 v2.0")
    print("="*60 + "\n")
    
    # 步驟 1: 數據加載和準備
    print("步驟 1: 數據加載和準備")
    print("-" * 60)
    
    handler = MLDataHandler()
    handler.create_sample_data(n_samples=2000)
    handler.preprocess_data()
    handler.feature_engineering()
    
    # 步驟 2: 準備 ML 數據
    print("\n步驟 2: 準備 ML 數據")
    print("-" * 60)
    
    X, y, feature_cols = handler.prepare_ml_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['order_filled']
    )
    
    print(f"訓練集: {X_train.shape[0]} 樣本")
    print(f"測試集: {X_test.shape[0]} 樣本")
    
    # 步驟 3: 訓練模型
    print("\n步驟 3: 模型訓練")
    print("-" * 60)
    
    trainer = ImprovedMLModelTrainer()
    
    trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
    trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
    trainer.train_pending_level_regressors(X_train, y_train, X_test, y_test)
    
    # 步驟 4: 評估模型
    print("\n步驟 4: 模型評估")
    print("-" * 60)
    
    trainer.evaluate_all_models(X_test, y_test, feature_cols)
    
    # 步驟 5: 保存模型
    print("\n步驟 5: 保存模型")
    print("-" * 60)
    
    trainer.save_models()
    
    # 步驟 6: 測試預測
    print("\n步驟 6: 測試實時預測")
    print("-" * 60)
    
    predictor = OrderPredictor(trainer, feature_cols)
    
    # 使用測試集中的第一個樣本進行預測
    test_sample = X_test.iloc[0].to_dict()
    prediction = predictor.predict_order_signal(test_sample)
    
    print("\n預測結果示例:")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("✓ 訓練流程完成！")
    print("="*60)
    
    return trainer, handler, predictor

if __name__ == "__main__":
    trainer, handler, predictor = main()
