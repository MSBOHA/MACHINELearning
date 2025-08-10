# 房价预测项目配置
import numpy as np

# 基本设置
RANDOM_SEED = 42
TARGET_COLUMN = 'SalePrice'
ID_COLUMN = 'Id'

# 数据路径
DATA_PATH = '../data/home-data-for-ml-course/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SUBMISSION_FILE = 'sample_submission.csv'

# 输出路径
SUBMISSION_PATH = '../submissions/'
MODEL_PATH = '../models/'

# 缺失值处理配置
# 这些特征的NA实际上是有意义的值（表示"无"）
NA_AS_NONE_FEATURES = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
    'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
]

# 有序类别特征及其排序
ORDINAL_FEATURES = {
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
    'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PoolQC': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
    'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
}

# 重要特征（从数据分析中得出）
IMPORTANT_NUMERIC_FEATURES = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
    '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd'
]

IMPORTANT_CATEGORICAL_FEATURES = [
    'Neighborhood', 'ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC'
]

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
    'random_state': RANDOM_SEED,
    'verbosity': -1,
    'n_estimators': 1000
}

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'verbosity': 0,
    'n_estimators': 1000
}

CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'random_state': RANDOM_SEED,
    'verbose': False
}

# 交叉验证设置
CV_FOLDS = 5
TEST_SIZE = 0.2

# 特征工程配置
APPLY_LOG_TRANSFORM = True  # 对目标变量应用对数变换
OUTLIER_THRESHOLD = 3  # 异常值检测的标准差倍数
