import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """训练LightGBM模型"""
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'random_state': self.random_state
            }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(params, train_data, valid_sets=[val_data], 
                            num_boost_round=1000, early_stopping_rounds=100, verbose_eval=False)
        else:
            model = lgb.train(params, train_data, num_boost_round=1000)
        
        self.models['lightgbm'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """训练XGBoost模型"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state
            }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(params, dtrain, evals=[(dval, 'val')], 
                            num_boost_round=1000, early_stopping_rounds=100, verbose_eval=False)
        else:
            model = xgb.train(params, dtrain, num_boost_round=1000)
        
        self.models['xgboost'] = model
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """训练CatBoost模型"""
        if params is None:
            params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'random_state': self.random_state,
                'verbose': False
            }
        
        model = CatBoostRegressor(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
        else:
            model.fit(X_train, y_train)
        
        self.models['catboost'] = model
        return model
    
    def cross_validate(self, X, y, model_type='lightgbm', cv=5, params=None):
        """交叉验证"""
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            if model_type == 'lightgbm':
                model = self.train_lightgbm(X_train_fold, y_train_fold, params=params)
                pred = model.predict(X_val_fold)
            elif model_type == 'xgboost':
                model = self.train_xgboost(X_train_fold, y_train_fold, params=params)
                pred = model.predict(xgb.DMatrix(X_val_fold))
            elif model_type == 'catboost':
                model = self.train_catboost(X_train_fold, y_train_fold, params=params)
                pred = model.predict(X_val_fold)
            
            score = np.sqrt(np.mean((y_val_fold - pred) ** 2))
            scores.append(score)
            print(f"Fold {fold + 1}: RMSE = {score:.4f}")
        
        print(f"平均RMSE: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        return scores
