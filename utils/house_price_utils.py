import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

def load_house_price_data(data_path):
    """加载房价预测数据集"""
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")
    sample_submission = pd.read_csv(f"{data_path}/sample_submission.csv")
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"样本提交文件形状: {sample_submission.shape}")
    
    return train_df, test_df, sample_submission

def analyze_target_variable(df, target_col='SalePrice'):
    """分析目标变量的分布"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 原始分布
    axes[0].hist(df[target_col], bins=50, alpha=0.7, color='skyblue')
    axes[0].set_title('房价分布')
    axes[0].set_xlabel('房价 ($)')
    axes[0].set_ylabel('频次')
    
    # 对数变换后的分布
    log_prices = np.log1p(df[target_col])
    axes[1].hist(log_prices, bins=50, alpha=0.7, color='lightgreen')
    axes[1].set_title('房价对数分布')
    axes[1].set_xlabel('log(房价)')
    axes[1].set_ylabel('频次')
    
    # 箱线图
    axes[2].boxplot(df[target_col])
    axes[2].set_title('房价箱线图')
    axes[2].set_ylabel('房价 ($)')
    
    plt.tight_layout()
    plt.show()
    
    # 统计信息
    print(f"房价统计:")
    print(df[target_col].describe())
    print(f"\\n偏度: {skew(df[target_col]):.4f}")
    print(f"峰度: {kurtosis(df[target_col]):.4f}")
    print(f"对数变换后偏度: {skew(log_prices):.4f}")

def missing_values_analysis(df, title=""):
    """缺失值分析"""
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    
    missing_table = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_percent
    })
    
    missing_table = missing_table[missing_table['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_table) > 0:
        print(f"{title} 缺失值统计:")
        print(missing_table)
        
        # 可视化
        if len(missing_table) <= 20:
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(missing_table)), missing_table['Missing_Count'])
            plt.xticks(range(len(missing_table)), missing_table.index, rotation=45, ha='right')
            plt.title(f'{title} 缺失值统计')
            plt.ylabel('缺失值数量')
            plt.tight_layout()
            plt.show()
    else:
        print(f"{title}: 无缺失值")
    
    return missing_table

def feature_correlation_analysis(df, target_col='SalePrice', top_n=15):
    """特征相关性分析"""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除ID和目标变量
    if 'Id' in numeric_features:
        numeric_features.remove('Id')
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    # 计算相关性
    correlations = df[numeric_features + [target_col]].corr()[target_col].sort_values(
        key=abs, ascending=False
    )
    
    print(f"与{target_col}相关性最高的数值特征 (Top {top_n}):")
    print(correlations.head(top_n + 1))  # +1 因为包含目标变量本身
    
    # 可视化
    plt.figure(figsize=(10, 8))
    top_corr = correlations.head(top_n + 1)
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f'与{target_col}相关性最高的特征')
    plt.xlabel('相关系数')
    plt.tight_layout()
    plt.show()
    
    return correlations

def categorical_feature_analysis(df, cat_features, target_col='SalePrice'):
    """类别特征分析"""
    n_features = len(cat_features)
    if n_features == 0:
        print("没有类别特征需要分析")
        return
    
    # 计算子图布局
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(cat_features[:min(len(axes), n_features)]):
        if feature in df.columns:
            # 计算每个类别的平均房价
            avg_price = df.groupby(feature)[target_col].mean().sort_values(ascending=False)
            
            if len(avg_price) <= 15:
                avg_price.plot(kind='bar', ax=axes[i], color='skyblue')
            else:
                avg_price.head(15).plot(kind='bar', ax=axes[i], color='skyblue')
                
            axes[i].set_title(f'{feature} 平均房价')
            axes[i].set_ylabel('平均房价')
            axes[i].tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for i in range(len(cat_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def house_price_feature_engineering(df):
    """房价预测特征工程"""
    df = df.copy()
    
    # 1. 创建新特征
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Bathrooms'] = (df['FullBath'] + df['HalfBath'] * 0.5 + 
                            df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5)
    df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                           df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    
    # 二值特征
    df['haspool'] = (df['PoolArea'] > 0).astype(int)
    df['has2ndfloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['hasgarage'] = (df['GarageArea'] > 0).astype(int)
    df['hasbsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['hasfireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    # 房龄特征
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['Years_Since_Remod'] = df['YrSold'] - df['YearRemodAdd']
    
    # 2. 处理缺失值
    # NA实际上是有意义的值
    na_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
               'PoolQC', 'Fence', 'MiscFeature']
    
    for col in na_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 特殊处理
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 数值特征缺失值处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if col == 'GarageYrBlt':
                df[col] = df[col].fillna(df['YearBuilt'])
            else:
                df[col] = df[col].fillna(0)
    
    # 类别特征缺失值处理
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    return df

def encode_categorical_features(train_df, test_df, ordinal_features=None):
    """编码类别特征"""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # 有序特征编码
    if ordinal_features:
        for feature, order in ordinal_features.items():
            if feature in train_encoded.columns:
                encoding_map = {val: i for i, val in enumerate(order)}
                train_encoded[feature] = train_encoded[feature].map(encoding_map)
                test_encoded[feature] = test_encoded[feature].map(encoding_map)
    
    # 标签编码其余类别特征
    label_encoders = {}
    categorical_features = train_encoded.select_dtypes(include=['object']).columns.tolist()
    
    for feature in categorical_features:
        le = LabelEncoder()
        # 合并训练集和测试集的值来训练编码器
        combined_values = pd.concat([train_encoded[feature], test_encoded[feature]]).unique()
        le.fit(combined_values)
        
        train_encoded[feature] = le.transform(train_encoded[feature])
        test_encoded[feature] = le.transform(test_encoded[feature])
        label_encoders[feature] = le
    
    return train_encoded, test_encoded, label_encoders

def rmse_score(y_true, y_pred):
    """计算RMSE得分"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """评估预测结果"""
    rmse = rmse_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    print(f"{model_name} 评估结果:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # 预测vs实际值散点图
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} - 预测vs实际')
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title(f'{model_name} - 残差图')
    
    plt.tight_layout()
    plt.show()
    
    return rmse, mae

def create_submission_file(test_ids, predictions, filename, target_col='SalePrice', 
                          submission_path='../submissions/'):
    """创建Kaggle提交文件"""
    submission = pd.DataFrame({
        'Id': test_ids,
        target_col: predictions
    })
    
    filepath = f'{submission_path}/{filename}'
    submission.to_csv(filepath, index=False)
    print(f"提交文件已保存: {filepath}")
    
    # 显示预测统计
    print(f"\\n预测统计:")
    print(submission[target_col].describe())
    
    return submission
