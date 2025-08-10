# Kaggle房价预测竞赛环境

这是一个专门为Kaggle房价预测竞赛设计的完整开发环境，包含所有必要的工具、模板和分析代码。

## 🏠 房价预测竞赛

当前项目已针对 **"House Prices - Advanced Regression Techniques"** 竞赛进行了优化配置。

### 数据集信息
- **竞赛**: House Prices - Advanced Regression Techniques
- **目标**: 预测房屋销售价格
- **数据位置**: `data/home-data-for-ml-course/`
- **文件**:
  - `train.csv`: 训练数据 (1460行, 81列)
  - `test.csv`: 测试数据 (1459行, 80列)
  - `sample_submission.csv`: 提交样本
  - `data_description.txt`: 数据说明文档

## 🚀 快速开始

### 方法1: 使用启动脚本 (推荐)
```bash
start_house_price_analysis.bat
```

### 方法2: 手动激活
```bash
conda activate kaggle
cd c:\Git_Code\kaggle_code
python quick_validation.py  # 验证环境
```

### 方法3: 直接运行分析
1. 激活环境: `conda activate kaggle`
2. 在VS Code中打开 `notebooks/house_price_analysis.ipynb`
3. 运行所有单元格

## 📁 项目结构

```
kaggle_code/
├── data/
│   └── home-data-for-ml-course/    # 房价竞赛数据
├── notebooks/
│   ├── kaggle_starter.ipynb        # 通用模板
│   └── house_price_analysis.ipynb  # 房价预测完整分析 ⭐
├── src/                            # 源代码
├── models/                         # 训练好的模型
├── submissions/                    # 提交文件
├── config/
│   ├── config.py                   # 通用配置
│   └── house_price_config.py       # 房价预测专用配置 ⭐
├── utils/
│   ├── data_utils.py              # 通用数据工具
│   ├── model_utils.py             # 模型工具
│   └── house_price_utils.py       # 房价预测专用工具 ⭐
├── quick_validation.py            # 环境验证脚本 ⭐
└── start_house_price_analysis.bat # 快速启动脚本 ⭐
```

## 🎯 主要功能

### 房价预测专用功能
- 🏡 **房价特征工程**：房屋总面积、浴室数量、房龄等
- 📊 **房价EDA**：价格分布、地理位置、房屋特征分析
- 🔧 **缺失值处理**：针对房价数据的智能填充策略
- 📈 **价格预测模型**：LightGBM、XGBoost、CatBoost调优
- 🎯 **模型集成**：多模型融合提升预测精度

### 通用功能
- 📊 **完整的EDA模板**：数据探索和可视化
- 🤖 **多种机器学习模型**：LightGBM, XGBoost, CatBoost等
- 🔧 **特征工程工具**：自动特征生成和选择
- 📈 **模型集成**：投票和堆叠集成方法
- 📋 **自动化评估**：交叉验证和性能指标
- 🎯 **提交文件生成**：自动格式化竞赛提交

## 📦 环境配置

### Conda环境: kaggle (Python 3.9)

**机器学习核心包**:
- pandas 2.3.1 - 数据处理
- numpy 2.0.2 - 数值计算  
- scikit-learn 1.6.1 - 机器学习
- xgboost 2.1.4 - 梯度提升
- lightgbm 4.6.0 - 轻量梯度提升
- catboost 1.2.8 - 类别梯度提升

**可视化和分析**:
- matplotlib 3.10.0 - 基础绘图
- seaborn 0.13.2 - 统计可视化
- plotly 6.2.0 - 交互式图表

**开发和优化**:
- jupyter 1.1.1 - 笔记本环境
- optuna 4.4.0 - 超参数优化
- shap 0.48.0 - 模型解释
- kaggle 1.6.17 - API接口

## 🔍 验证和调试

### 环境验证
```bash
python quick_validation.py
```

验证内容：
- ✅ Python版本和包安装状态
- ✅ 数据文件完整性检查
- ✅ 数据基本统计信息
- ✅ 缺失值分析

### 常见问题
1. **包导入错误**: 确保已激活kaggle环境
2. **数据路径错误**: 数据应在 `data/home-data-for-ml-course/`
3. **内存不足**: 考虑使用数据采样或特征选择

## 🏆 竞赛提交

1. **训练模型**: 运行 `house_price_analysis.ipynb`
2. **生成预测**: 使用最佳模型预测测试集
3. **创建提交**: 使用 `create_submission_file()` 函数
4. **上传结果**: 提交到Kaggle平台

## 🔗 相关资源

- [Kaggle API设置](KAGGLE_API_SETUP.md)
- [房价竞赛页面](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [数据说明文档](data/home-data-for-ml-course/data_description.txt)

## 📖 使用指南

### 对于初学者
1. 首先运行 `quick_validation.py` 确保环境正常
2. 阅读 `data/home-data-for-ml-course/data_description.txt` 了解数据
3. 逐步运行 `house_price_analysis.ipynb` 学习分析流程

### 对于有经验的用户
1. 直接修改 `config/house_price_config.py` 调整参数
2. 在 `utils/house_price_utils.py` 中添加自定义函数
3. 实验不同的特征工程和模型组合

---

**祝您在Kaggle房价预测竞赛中取得好成绩！** 🏆
