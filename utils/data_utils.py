import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """加载数据"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError("不支持的文件格式")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def basic_info(df):
    """显示数据基本信息"""
    print("数据形状:", df.shape)
    print("\n数据类型:")
    print(df.dtypes)
    print("\n缺失值:")
    print(df.isnull().sum())
    print("\n基础统计信息:")
    print(df.describe())

def plot_distributions(df, columns, figsize=(15, 10)):
    """绘制特征分布图"""
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(columns):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'{col} 分布')
    
    # 隐藏多余的子图
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df, figsize=(12, 8)):
    """绘制相关性热力图"""
    plt.figure(figsize=figsize)
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.show()

def evaluate_model(y_true, y_pred):
    """模型评估"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """交叉验证"""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    rmse_scores = np.sqrt(-scores)
    
    print(f"交叉验证RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    return rmse_scores

def create_submission(test_ids, predictions, filename='submission.csv', target_col='target'):
    """创建提交文件"""
    submission = pd.DataFrame({
        'id': test_ids,
        target_col: predictions
    })
    submission.to_csv(f'../submissions/{filename}', index=False)
    print(f"提交文件已保存: {filename}")
    return submission
