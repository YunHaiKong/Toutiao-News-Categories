# 新闻分类项目

基于深度学习的中文新闻文本分类系统，支持朴素贝叶斯、TextCNN和RNN三种模型。

## 项目结构

```
新闻分类项目/
├── config.py              # 配置文件
├── data_process.py         # 数据预处理模块
├── models.py              # 模型定义模块
├── train.py               # 模型训练模块
├── train_bert.py          # BERT模型专用训练脚本（新增）
├── evaluate.py            # 模型评估模块
├── predict.py             # 模型预测模块
├── main.py                # 主程序入口
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明文档
```

## 功能特性

### 支持的模型
- **朴素贝叶斯**: 基于TF-IDF特征的传统机器学习模型
- **TextCNN**: 基于卷积神经网络的文本分类模型
- **RNN**: 基于双向LSTM的循环神经网络模型
- **BERT**: 基于预训练BERT模型的微调分类器

### 主要功能
- 数据预处理（文本清洗、分词、序列化）
- 模型训练和保存
- 模型评估和性能对比
- 交互式预测界面（基于Gradio）
- 可视化结果展示

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将数据文件 `toutiao_news_6.csv` 放置在项目根目录下。数据格式要求：
- CSV格式文件
- 包含 `text` 列（新闻标题）
- 包含 `category` 列（新闻类别）

## 使用方法

### 1. 完整流程（推荐）

```bash
python main.py --mode all
```

这将依次执行：
1. 数据预处理
2. 训练所有模型
3. 评估模型性能
4. 启动预测界面

### 2. 单独训练模型

```bash
python main.py --mode train
```

### 3. 单独评估模型

```bash
python main.py --mode evaluate
```

### 4. 启动预测界面

```bash
python main.py --mode predict
```

### 5. 交互式预测

```bash
python main.py --mode interactive
```

### 6. BERT模型训练（新增）

#### 安装BERT依赖
```bash
pip install transformers torch
```

#### 训练所有模型（包括BERT）
```bash
python train.py
```

#### 单独训练BERT模型
```bash
python train_bert.py --mode train
```

#### 测试BERT模型
```bash
python train_bert.py --mode test
```

#### BERT模型特点
- 使用预训练的 `bert-base-chinese` 模型
- 支持中文文本的深度理解
- 通常能获得更高的分类准确率
- 训练时间较长，建议使用GPU
- 模型文件较大（约400MB）

## 配置说明

主要配置项在 `config.py` 中：

- `DATA_PATH`: 数据文件路径
- `MAX_WORDS`: 词汇表最大大小
- `MAX_LEN`: 文本序列最大长度
- `BATCH_SIZE`: 训练批次大小
- `EPOCHS`: 训练轮数
- `TEST_SPLIT`: 测试集比例
- `VALIDATION_SPLIT`: 验证集比例

## 模型性能

训练完成后，系统会自动生成：
- 各模型的准确率、精确率、召回率、F1分数
- 混淆矩阵可视化
- 模型性能对比图
- 详细的分类报告

## 预测界面

基于Gradio构建的Web界面，支持：
- 实时文本输入
- 三种模型同时预测
- 置信度显示
- 友好的用户界面

## 文件说明

### config.py
项目配置文件，包含所有可调参数。

### data_process.py
数据预处理模块，包含：
- 文本清洗和分词
- 标签映射创建
- TF-IDF向量化
- 文本序列化和填充
- 预处理工具的保存和加载

### models.py
模型定义模块，包含四个模型类：
- `NaiveBayesClassifier`: 朴素贝叶斯分类器
- `TextCNNClassifier`: TextCNN分类器
- `RNNClassifier`: RNN分类器
- `BertClassifier`: BERT预训练模型微调分类器（新增）

### train.py
模型训练模块，包含：
- 数据准备
- 模型训练
- 训练曲线绘制
- 性能总结

### evaluate.py
模型评估模块，包含：
- 模型加载
- 性能评估
- 结果可视化
- 分类报告生成

### predict.py
模型预测模块，包含：
- 单文本预测
- 批量预测
- Gradio界面

### main.py
主程序入口，提供命令行接口。

## 注意事项

1. 确保数据文件格式正确
2. 首次运行需要较长时间进行数据预处理和模型训练
3. 建议在GPU环境下训练深度学习模型
4. 模型文件会保存在 `models/` 目录下
5. 预处理工具会保存在 `preprocessors/` 目录下

## 扩展功能

- 支持更多模型类型（如GPT、XLNet等）
- 支持更多评估指标
- 支持模型集成和投票
- 支持在线学习和模型更新
- 支持多语言文本分类

## 问题排查

1. **导入错误**: 检查依赖包是否正确安装
2. **数据加载失败**: 检查数据文件路径和格式
3. **内存不足**: 减少批次大小或序列长度
4. **训练速度慢**: 考虑使用GPU或减少数据量

## 联系方式

如有问题或建议，请联系项目维护者。