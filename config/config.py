# 常用配置参数
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 模型参数
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': RANDOM_SEED
}

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED
}

# 文件路径
DATA_PATH = '../data/'
MODEL_PATH = '../models/'
SUBMISSION_PATH = '../submissions/'
