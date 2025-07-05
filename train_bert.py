#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT模型单独训练脚本
"""

import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from config import Config
from data_process import DataPreprocessor
from models import BertClassifier

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def train_bert_model():
    """训练BERT模型"""
    print("=" * 60)
    print("BERT预训练模型微调 - 新闻分类任务")
    print("=" * 60)
    
    # 初始化数据预处理器
    preprocessor = DataPreprocessor()
    
    # 准备数据
    print("\n1. 数据准备...")
    data = preprocessor.load_data(Config.DATA_PATH)
    if data is None:
        raise ValueError("数据加载失败")
    
    # 预处理数据
    texts, labels = preprocessor.preprocess_data(data)
    
    # 划分数据集
    X_train_text, X_test_text, y_train, y_test = preprocessor.split_data(texts, labels)
    
    print(f"训练集大小: {len(X_train_text)}")
    print(f"测试集大小: {len(X_test_text)}")
    print(f"类别数量: {Config.NUM_CLASSES}")
    
    # 创建BERT分类器
    print("\n2. 创建BERT模型...")
    bert_classifier = BertClassifier()
    
    # 显示模型配置
    print(f"BERT模型: {Config.BERT_MODEL_NAME}")
    print(f"最大序列长度: {Config.BERT_MAX_LEN}")
    print(f"学习率: {Config.BERT_LEARNING_RATE}")
    print(f"批次大小: {Config.BERT_BATCH_SIZE}")
    print(f"训练轮数: {Config.BERT_EPOCHS}")
    
    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            Config.BERT_MODEL_PATH, 
            monitor='val_accuracy', 
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print("\n3. 开始训练BERT模型...")
    start_time = time.time()
    
    try:
        history = bert_classifier.train(
            X_train_text, y_train,
            callbacks=callbacks
        )
        
        training_time = time.time() - start_time
        print(f"\n训练完成！耗时: {training_time:.2f}秒")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查是否安装了transformers库: pip install transformers")
        print("2. 检查是否有足够的GPU内存")
        print("3. 尝试减小批次大小 (BERT_BATCH_SIZE)")
        print("4. 检查网络连接，确保能下载预训练模型")
        return None
    
    # 评估模型
    print("\n4. 评估模型性能...")
    test_loss, test_accuracy = bert_classifier.evaluate(X_test_text, y_test)
    y_pred = bert_classifier.predict(X_test_text)
    
    # 计算详细指标
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 打印结果
    print("\n" + "=" * 50)
    print("BERT模型性能评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {test_accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print(f"训练时间: {training_time:.2f}秒")
    print("=" * 50)
    
    # 保存模型和tokenizer
    print("\n5. 保存模型...")
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    bert_classifier.save_model(Config.BERT_MODEL_PATH, Config.BERT_TOKENIZER_PATH)
    
    # 保存预处理工具
    preprocessor.save_preprocessors()
    
    # 绘制训练曲线
    if Config.PLOT_HISTORY and history:
        print("\n6. 绘制训练曲线...")
        plot_training_history(history)
    
    print("\nBERT模型训练完成！")
    return bert_classifier, test_accuracy, history

def plot_training_history(history):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率', marker='o')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='验证准确率', marker='s')
    plt.title('BERT模型准确率曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失', marker='o')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='验证损失', marker='s')
    plt.title('BERT模型损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bert_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练曲线已保存为 bert_training_history.png")

def test_bert_model():
    """测试已训练的BERT模型"""
    print("\n测试BERT模型...")
    
    # 加载模型
    bert_classifier = BertClassifier.load_model(
        Config.BERT_MODEL_PATH, 
        Config.BERT_TOKENIZER_PATH
    )
    
    if bert_classifier is None:
        print("模型加载失败！")
        return
    
    # 测试样本
    test_texts = pd.Series([
        "中国足球队在世界杯上的表现令人失望",
        "苹果公司发布了新款iPhone手机",
        "股市今日大涨，投资者信心增强",
        "科学家发现了新的治疗癌症的方法",
        "明星张三与李四的恋情曝光",
        "新的教育政策将于下月实施"
    ])
    
    # 预测
    predictions = bert_classifier.predict(test_texts)
    probabilities = bert_classifier.predict_proba(test_texts)
    
    # 显示结果
    print("\n预测结果:")
    print("-" * 60)
    for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probabilities)):
        max_prob = np.max(prob)
        print(f"{i+1}. 文本: {text}")
        print(f"   预测类别: {pred} (置信度: {max_prob:.4f})")
        print("-" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BERT模型训练和测试')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='运行模式: train(训练) 或 test(测试)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        result = train_bert_model()
        if result:
            print("\n训练成功！可以使用以下命令测试模型:")
            print("python train_bert.py --mode test")
    else:
        # 测试模式
        test_bert_model()