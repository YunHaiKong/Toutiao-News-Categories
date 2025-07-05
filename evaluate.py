# -*- coding: utf-8 -*-
"""
模型评估模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

from config import Config
from data_process import DataPreprocessor
from models import NaiveBayesClassifier, TextCNNClassifier, RNNClassifier

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.evaluation_results = {}
        
    def load_models_and_data(self):
        """加载模型和数据"""
        print("正在加载模型和数据...")
        
        # 加载预处理工具
        tokenizer, vectorizer, label_mapping = DataPreprocessor.load_preprocessors()
        self.preprocessor.tokenizer = tokenizer
        self.preprocessor.vectorizer = vectorizer
        
        if label_mapping:
            self.preprocessor.label2id = label_mapping['label2id']
            self.preprocessor.id2label = label_mapping['id2label']
        
        # 加载模型
        try:
            self.models['naive_bayes'] = NaiveBayesClassifier.load_model(Config.NB_MODEL_SAVE_PATH)
        except Exception as e:
            print(f"朴素贝叶斯模型加载失败: {e}")
        
        try:
            self.models['textcnn'] = TextCNNClassifier.load_model(Config.TEXTCNN_MODEL_PATH)
        except Exception as e:
            print(f"TextCNN模型加载失败: {e}")
        
        try:
            self.models['rnn'] = RNNClassifier.load_model(Config.RNN_MODEL_PATH)
        except Exception as e:
            print(f"RNN模型加载失败: {e}")
        
        # 加载测试数据
        data = self.preprocessor.load_data(Config.DATA_PATH)
        if data is not None:
            texts, labels = self.preprocessor.preprocess_data(data)
            # 这里简单地使用相同的数据划分方式获取测试集
            from sklearn.model_selection import train_test_split
            _, self.X_test_text, _, self.y_test = train_test_split(
                texts, labels, test_size=Config.TEST_SPLIT, random_state=1234, stratify=labels
            )
        
        print("模型和数据加载完成")
    
    def evaluate_naive_bayes(self):
        """评估朴素贝叶斯模型"""
        if 'naive_bayes' not in self.models or self.models['naive_bayes'] is None:
            print("朴素贝叶斯模型未加载")
            return None
        
        print("\n评估朴素贝叶斯模型...")
        
        # 转换测试数据
        X_test_tfidf = self.preprocessor.texts_to_tfidf(self.X_test_text)
        
        # 预测
        y_pred = self.models['naive_bayes'].predict(X_test_tfidf)
        y_pred_proba = self.models['naive_bayes'].predict_proba(X_test_tfidf)
        
        # 计算指标
        metrics = self.calculate_metrics(self.y_test, y_pred)
        
        self.evaluation_results['朴素贝叶斯'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        return metrics
    
    def evaluate_textcnn(self):
        """评估TextCNN模型"""
        if 'textcnn' not in self.models or self.models['textcnn'] is None:
            print("TextCNN模型未加载")
            return None
        
        print("\n评估TextCNN模型...")
        
        # 转换测试数据
        X_test_seq = self.preprocessor.texts_to_sequences(self.X_test_text)
        
        # 预测
        y_pred = self.models['textcnn'].predict(X_test_seq)
        y_pred_proba = self.models['textcnn'].predict_proba(X_test_seq)
        
        # 计算指标
        metrics = self.calculate_metrics(self.y_test, y_pred)
        
        self.evaluation_results['TextCNN'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        return metrics
    
    def evaluate_rnn(self):
        """评估RNN模型"""
        if 'rnn' not in self.models or self.models['rnn'] is None:
            print("RNN模型未加载")
            return None
        
        print("\n评估RNN模型...")
        
        # 转换测试数据
        X_test_seq = self.preprocessor.texts_to_sequences(self.X_test_text)
        
        # 预测
        y_pred = self.models['rnn'].predict(X_test_seq)
        y_pred_proba = self.models['rnn'].predict_proba(X_test_seq)
        
        # 计算指标
        metrics = self.calculate_metrics(self.y_test, y_pred)
        
        self.evaluation_results['RNN'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def print_evaluation_results(self):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            print(f"\n{model_name}:")
            print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
            print(f"  精确率 (Precision): {metrics['precision']:.4f}")
            print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
            print(f"  F1分数 (F1-Score):  {metrics['f1']:.4f}")
            print("-" * 40)
    
    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        n_models = len(self.evaluation_results)
        if n_models == 0:
            print("没有评估结果可绘制")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        # 获取标签名称
        label_names = list(self.preprocessor.id2label.values())
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            y_pred = results['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            # 绘制混淆矩阵
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[idx]
            )
            axes[idx].set_title(f'{model_name} 混淆矩阵')
            axes[idx].set_xlabel('预测标签')
            axes[idx].set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self):
        """绘制性能对比图"""
        if not self.evaluation_results:
            print("没有评估结果可绘制")
            return
        
        # 准备数据
        models = list(self.evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数']
        
        # 创建数据矩阵
        data = []
        for model in models:
            row = [self.evaluation_results[model]['metrics'][metric] for metric in metrics]
            data.append(row)
        
        # 绘制柱状图
        x = np.arange(len(metric_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model in enumerate(models):
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[i], width, label=model)
            
            # 在柱子上添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('评估指标')
        ax.set_ylabel('分数')
        ax.set_title('模型性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
    
    def generate_classification_report(self):
        """生成分类报告"""
        label_names = list(self.preprocessor.id2label.values())
        
        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name} 详细分类报告:")
            print("=" * 50)
            y_pred = results['predictions']
            report = classification_report(
                self.y_test, y_pred,
                target_names=label_names,
                digits=4
            )
            print(report)
    
    def evaluate_all_models(self):
        """评估所有模型"""
        # 加载模型和数据
        self.load_models_and_data()
        
        # 评估各个模型
        self.evaluate_naive_bayes()
        self.evaluate_textcnn()
        self.evaluate_rnn()
        
        # 打印结果
        self.print_evaluation_results()
        
        # 生成详细报告
        self.generate_classification_report()
        
        # 绘制图表
        self.plot_confusion_matrices()
        self.plot_performance_comparison()
        
        return self.evaluation_results

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models()
    print("\n模型评估完成！")