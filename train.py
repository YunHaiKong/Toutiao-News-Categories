# -*- coding: utf-8 -*-
"""
模型训练模块
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
from models import NaiveBayesClassifier, TextCNNClassifier, RNNClassifier, BertClassifier

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model_performance = {}
        
    def prepare_data(self):
        """准备数据"""
        print("=" * 50)
        print("开始数据准备...")
        
        # 加载数据
        data = self.preprocessor.load_data(Config.DATA_PATH)
        if data is None:
            raise ValueError("数据加载失败")
        
        # 预处理数据
        texts, labels = self.preprocessor.preprocess_data(data)
        
        # 划分数据集
        X_train_text, X_test_text, y_train, y_test = self.preprocessor.split_data(texts, labels)
        
        print("数据准备完成")
        return X_train_text, X_test_text, y_train, y_test
    
    def train_naive_bayes(self, X_train_text, X_test_text, y_train, y_test):
        """训练朴素贝叶斯模型"""
        print("\n" + "=" * 50)
        print("训练朴素贝叶斯模型...")
        start_time = time.time()
        
        # 创建TF-IDF向量化器
        vectorizer = self.preprocessor.create_vectorizer(X_train_text)
        
        # 转换为TF-IDF向量
        X_train_tfidf = vectorizer.transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)
        
        # 训练模型
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.train(X_train_tfidf, y_train)
        
        # 预测
        y_pred = nb_classifier.predict(X_test_tfidf)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = time.time() - start_time
        
        print(f"朴素贝叶斯准确率: {accuracy:.4f}")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 记录结果
        self.model_performance['朴素贝叶斯'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': training_time
        }
        
        # 保存模型
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        nb_classifier.save_model(Config.NB_MODEL_SAVE_PATH)
        
        return nb_classifier, accuracy
    
    def train_textcnn(self, X_train_text, X_test_text, y_train, y_test):
        """训练TextCNN模型"""
        print("\n" + "=" * 50)
        print("训练TextCNN模型...")
        start_time = time.time()
        
        # 创建tokenizer
        tokenizer = self.preprocessor.create_tokenizer(X_train_text)
        
        # 文本转序列
        X_train_seq = self.preprocessor.texts_to_sequences(X_train_text)
        X_test_seq = self.preprocessor.texts_to_sequences(X_test_text)
        
        # 创建和训练模型
        cnn_classifier = TextCNNClassifier()
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(Config.TEXTCNN_MODEL_PATH, monitor='val_accuracy', save_best_only=True)
        ]
        
        # 训练模型
        history = cnn_classifier.train(
            X_train_seq, y_train,
            callbacks=callbacks
        )
        
        # 评估
        test_loss, test_accuracy = cnn_classifier.evaluate(X_test_seq, y_test)
        y_pred = cnn_classifier.predict(X_test_seq)
        
        # 计算指标
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = time.time() - start_time
        
        print(f"TextCNN准确率: {test_accuracy:.4f}")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 记录结果
        self.model_performance['TextCNN'] = {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': training_time
        }
        
        # 绘制训练曲线
        if Config.PLOT_HISTORY:
            self.plot_training_history(history, 'TextCNN')
        
        return cnn_classifier, test_accuracy, history
    
    def train_rnn(self, X_train_text, X_test_text, y_train, y_test):
        """训练RNN模型"""
        print("\n" + "=" * 50)
        print("训练RNN模型...")
        start_time = time.time()
        
        # 使用已有的tokenizer或创建新的
        if not self.preprocessor.tokenizer:
            tokenizer = self.preprocessor.create_tokenizer(X_train_text)
        
        # 文本转序列
        X_train_seq = self.preprocessor.texts_to_sequences(X_train_text)
        X_test_seq = self.preprocessor.texts_to_sequences(X_test_text)
        
        # 创建和训练模型
        rnn_classifier = RNNClassifier()
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(Config.RNN_MODEL_PATH, monitor='val_accuracy', save_best_only=True)
        ]
        
        # 训练模型
        history = rnn_classifier.train(
            X_train_seq, y_train,
            callbacks=callbacks
        )
        
        # 评估
        test_loss, test_accuracy = rnn_classifier.evaluate(X_test_seq, y_test)
        y_pred = rnn_classifier.predict(X_test_seq)
        
        # 计算指标
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = time.time() - start_time
        
        print(f"RNN准确率: {test_accuracy:.4f}")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 记录结果
        self.model_performance['RNN'] = {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': training_time
        }
        
        # 绘制训练曲线
        if Config.PLOT_HISTORY:
            self.plot_training_history(history, 'RNN')
        
        return rnn_classifier, test_accuracy, history
    
    def train_bert(self, X_train_text, X_test_text, y_train, y_test):
        """训练BERT模型"""
        print("\n" + "=" * 50)
        print("训练BERT模型...")
        start_time = time.time()
        
        # 创建BERT分类器
        bert_classifier = BertClassifier()
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(Config.BERT_MODEL_PATH, monitor='val_accuracy', save_best_only=True)
        ]
        
        # 训练模型
        history = bert_classifier.train(
            X_train_text, y_train,
            callbacks=callbacks
        )
        
        # 评估
        test_loss, test_accuracy = bert_classifier.evaluate(X_test_text, y_test)
        y_pred = bert_classifier.predict(X_test_text)
        
        # 计算指标
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = time.time() - start_time
        
        print(f"BERT准确率: {test_accuracy:.4f}")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 记录结果
        self.model_performance['BERT'] = {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': training_time
        }
        
        # 保存模型和tokenizer
        bert_classifier.save_model(Config.BERT_MODEL_PATH, Config.BERT_TOKENIZER_PATH)
        
        # 绘制训练曲线
        if Config.PLOT_HISTORY:
            self.plot_training_history(history, 'BERT')
        
        return bert_classifier, test_accuracy, history
    
    def plot_training_history(self, history, model_name):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练准确率')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title(f'{model_name}准确率曲线')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练损失')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='验证损失')
        plt.title(f'{model_name}损失曲线')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self):
        """打印性能总结"""
        print("\n" + "=" * 50)
        print("模型性能总结:")
        print("-" * 50)
        
        for model_name, metrics in self.model_performance.items():
            print(f"{model_name}:")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  训练时间: {metrics['train_time']:.2f} 秒")
            print("-" * 30)
    
    def train_all_models(self, include_bert=True):
        """训练所有模型"""
        # 准备数据
        X_train_text, X_test_text, y_train, y_test = self.prepare_data()
        
        models = {}
        
        # 训练朴素贝叶斯
        nb_model, nb_acc = self.train_naive_bayes(X_train_text, X_test_text, y_train, y_test)
        models['naive_bayes'] = nb_model
        
        # 训练TextCNN
        cnn_model, cnn_acc, cnn_history = self.train_textcnn(X_train_text, X_test_text, y_train, y_test)
        models['textcnn'] = cnn_model
        
        # 训练RNN
        rnn_model, rnn_acc, rnn_history = self.train_rnn(X_train_text, X_test_text, y_train, y_test)
        models['rnn'] = rnn_model
        
        # 训练BERT（可选）
        if include_bert:
            try:
                bert_model, bert_acc, bert_history = self.train_bert(X_train_text, X_test_text, y_train, y_test)
                models['bert'] = bert_model
            except Exception as e:
                print(f"BERT模型训练失败: {e}")
                print("请确保已安装transformers库: pip install transformers")
        
        # 保存预处理工具
        self.preprocessor.save_preprocessors()
        
        # 打印性能总结
        self.print_performance_summary()
        
        return models

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # 可以选择是否训练BERT模型
    # models = trainer.train_all_models(include_bert=False)  # 不训练BERT
    models = trainer.train_all_models(include_bert=True)   # 训练BERT
    
    print("\n所有模型训练完成！")
    print(f"训练完成的模型: {list(models.keys())}")
    
    # 显示最佳模型
    if trainer.model_performance:
        best_model = max(trainer.model_performance.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.4f})")
    
    # 单独训练BERT模型的示例
    # print("\n单独训练BERT模型...")
    # X_train_text, X_test_text, y_train, y_test = trainer.prepare_data()
    # bert_model, bert_acc, bert_history = trainer.train_bert(X_train_text, X_test_text, y_train, y_test)