# -*- coding: utf-8 -*-
"""
新闻分类项目配置文件
"""

class Config:
    # 数据路径
    DATA_PATH = "../toutiao_news_6.csv"
    TRAIN_DATA_PATH = "data/train.csv"
    TEST_DATA_PATH = "data/test.csv"
    
    # 文本预处理
    MAX_WORDS = 10000  # 最大词汇量
    MAX_LEN = 100      # 最大序列长度
    STOP_WORDS = False # 是否使用停用词
    NUM_CLASSES = 6    # 分类数目
    
    # 模型参数
    EMBEDDING_DIM = 128  # 词向量维度
    
    # 训练参数
    BATCH_SIZE = 128
    EPOCHS = 1
    TEST_SPLIT = 0.2     # 数据集按80%:20%比例划分训练集和测试集
    VALIDATION_SPLIT = 0.1 # 训练集中，选10%的样本用于验证
    PLOT_HISTORY = True    # 是否绘制训练曲线
    
    # 保存路径
    MODEL_SAVE_PATH = "models/"
    TOKENIZER_SAVE_PATH = "models/tokenizer.pkl"
    LABELS_SAVE_PATH = "models/labels.pkl"
    VECTORIZER_SAVE_PATH = "models/vectorizer.pkl"
    NB_MODEL_SAVE_PATH = "models/naive_bayes.pkl"
    
    # TextCNN模型保存路径
    TEXTCNN_MODEL_PATH = "models/textcnn_model.h5"
    
    # RNN模型保存路径
    RNN_MODEL_PATH = "models/rnn_model.h5"
    
    # BERT模型保存路径
    BERT_MODEL_PATH = "models/bert_model.h5"
    BERT_TOKENIZER_PATH = "models/bert_tokenizer/"
    
    # 预训练词向量
    USE_PRETRAINED_EMBEDDING = False
    EMBEDDING_FILE_PATH = ""  # 本地词向量文件路径
    TRAINABLE = True # 是否微调
    
    # BERT配置
    BERT_MODEL_NAME = "models/bert-base-chinese"  # 改为本地路径
    BERT_MAX_LEN = 128  # BERT最大序列长度
    BERT_LEARNING_RATE = 2e-5  # BERT微调学习率
    BERT_EPOCHS = 1 # BERT训练轮数
    BERT_BATCH_SIZE = 16  # BERT批次大小