#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数调优训练器
使用不同参数训练不同模型并记录表格
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from data_process import DataPreprocessor
from models import NaiveBayesClassifier, TextCNNClassifier, RNNClassifier, BertClassifier

class ParameterTuningTrainer:
    """
    参数调优训练器
    系统地测试不同参数组合并记录结果
    """
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_and_prepare_data(self):
        """
        加载和准备数据
        """
        print("正在加载和预处理数据...")
        
        # 加载数据
        data = self.preprocessor.load_data(Config.DATA_PATH)
        if data is None:
            raise ValueError("数据加载失败")
        
        # 预处理数据
        texts, labels = self.preprocessor.preprocess_data(data)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(texts, labels)
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"类别数: {len(set(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def train_naive_bayes_variants(self, X_train, X_test, y_train, y_test):
        """
        训练朴素贝叶斯的不同参数变体
        """
        print("\n=== 训练朴素贝叶斯模型变体 ===")
        
        # 不同的TF-IDF参数组合
        tfidf_configs = [
            {'max_features': 1000, 'name': 'NB_1K'},
            {'max_features': 3000, 'name': 'NB_3K'},
            {'max_features': 5000, 'name': 'NB_5K'},
            {'max_features': 10000, 'name': 'NB_10K'},
        ]
        
        for config in tfidf_configs:
            print(f"\n训练 {config['name']} (max_features={config['max_features']})...")
            
            start_time = time.time()
            
            # 创建新的向量化器
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=config['max_features'])
            
            # 转换数据
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            # 训练模型
            model = NaiveBayesClassifier()
            model.train(X_train_tfidf, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_test_tfidf)
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            training_time = time.time() - start_time
            
            # 记录结果
            result = {
                'experiment_id': self.experiment_id,
                'model_name': config['name'],
                'model_type': '朴素贝叶斯',
                'max_features': config['max_features'],
                'max_len': 'N/A',
                'embedding_dim': 'N/A',
                'batch_size': 'N/A',
                'epochs': 'N/A',
                'learning_rate': 'N/A',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'model_size': 'Small',
                'parameters': f"TF-IDF max_features={config['max_features']}",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.results.append(result)
            
            print(f"准确率: {accuracy:.4f}, F1: {f1:.4f}, 训练时间: {training_time:.2f}s")
    
    def train_textcnn_variants(self, X_train, X_test, y_train, y_test):
        """
        训练TextCNN的不同参数变体
        """
        print("\n=== 训练TextCNN模型变体 ===")
        
        # 不同的参数组合
        cnn_configs = [
            {
                'name': 'TextCNN_Small',
                'max_words': 5000,
                'max_len': 50,
                'embedding_dim': 64,
                'batch_size': 64,
                'epochs': 1
            },
            {
                'name': 'TextCNN_Medium',
                'max_words': 10000,
                'max_len': 100,
                'embedding_dim': 128,
                'batch_size': 128,
                'epochs': 1
            },
            {
                'name': 'TextCNN_Large',
                'max_words': 15000,
                'max_len': 150,
                'embedding_dim': 256,
                'batch_size': 64,
                'epochs': 2
            }
        ]
        
        for config in cnn_configs:
            print(f"\n训练 {config['name']}...")
            print(f"参数: max_words={config['max_words']}, max_len={config['max_len']}, embedding_dim={config['embedding_dim']}")
            
            start_time = time.time()
            
            try:
                # 创建tokenizer
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                
                tokenizer = Tokenizer(num_words=config['max_words'], oov_token='<UNK>')
                tokenizer.fit_on_texts(X_train)
                
                # 转换数据
                X_train_seq = tokenizer.texts_to_sequences(X_train)
                X_test_seq = tokenizer.texts_to_sequences(X_test)
                
                X_train_pad = pad_sequences(X_train_seq, maxlen=config['max_len'], padding='post')
                X_test_pad = pad_sequences(X_test_seq, maxlen=config['max_len'], padding='post')
                
                # 创建和训练模型
                model = TextCNNClassifier(
                    vocab_size=config['max_words'],
                    max_len=config['max_len'],
                    embedding_dim=config['embedding_dim'],
                    num_classes=Config.NUM_CLASSES
                )
                
                model.build_model()
                
                # 训练模型
                history = model.train(
                    X_train_pad, y_train,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    verbose=0
                )
                
                # 预测和评估
                y_pred_proba = model.predict_proba(X_test_pad)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                training_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'experiment_id': self.experiment_id,
                    'model_name': config['name'],
                    'model_type': 'TextCNN',
                    'max_features': config['max_words'],
                    'max_len': config['max_len'],
                    'embedding_dim': config['embedding_dim'],
                    'batch_size': config['batch_size'],
                    'epochs': config['epochs'],
                    'learning_rate': 'default',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time,
                    'model_size': 'Medium',
                    'parameters': f"max_words={config['max_words']}, max_len={config['max_len']}, emb_dim={config['embedding_dim']}",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.results.append(result)
                
                print(f"准确率: {accuracy:.4f}, F1: {f1:.4f}, 训练时间: {training_time:.2f}s")
                
            except Exception as e:
                print(f"训练 {config['name']} 时出错: {e}")
                continue
    
    def train_bert_variants(self, X_train, X_test, y_train, y_test):
        """
        训练BERT的不同参数变体
        """
        print("\n=== 训练BERT模型变体 ===")
        
        # 不同的参数组合
        bert_configs = [
            {
                'name': 'BERT_Small',
                'max_len': 64,
                'batch_size': 16,
                'epochs': 1,
                'learning_rate': 2e-5
            },
            {
                'name': 'BERT_Medium',
                'max_len': 128,
                'batch_size': 8,
                'epochs': 2,
                'learning_rate': 2e-5
            },
            {
                'name': 'BERT_Large',
                'max_len': 256,
                'batch_size': 4,
                'epochs': 3,
                'learning_rate': 1e-5
            }
        ]
        
        for config in bert_configs:
            print(f"\n训练 {config['name']}...")
            print(f"参数: max_len={config['max_len']}, batch_size={config['batch_size']}, epochs={config['epochs']}")
            
            start_time = time.time()
            
            try:
                # 创建和训练模型
                model = BertClassifier(
                    max_len=config['max_len'],
                    num_classes=Config.NUM_CLASSES
                )
                
                # 修改模型的批次大小和学习率
                model.build_model()
                
                # 重新编译模型以使用新的学习率
                from tensorflow.keras.optimizers import Adam
                model.model.compile(
                    optimizer=Adam(learning_rate=config['learning_rate']),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # 训练模型（使用较少的数据以加快训练）
                sample_size = min(1000, len(X_train))  # 使用1000个样本进行快速测试
                X_train_sample = X_train[:sample_size]
                y_train_sample = y_train[:sample_size]
                
                # 训练模型
                history = model.train(
                    X_train_sample, y_train_sample,
                    callbacks=None
                )
                
                # 预测和评估（使用较少的测试数据）
                test_sample_size = min(200, len(X_test))
                X_test_sample = X_test[:test_sample_size]
                y_test_sample = y_test[:test_sample_size]
                
                y_pred_proba = model.predict_proba(X_test_sample)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # 计算指标
                accuracy = accuracy_score(y_test_sample, y_pred)
                precision = precision_score(y_test_sample, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_sample, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_sample, y_pred, average='weighted', zero_division=0)
                
                training_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'experiment_id': self.experiment_id,
                    'model_name': config['name'],
                    'model_type': 'BERT',
                    'max_features': 'N/A',
                    'max_len': config['max_len'],
                    'embedding_dim': 768,  # BERT固定维度
                    'batch_size': config['batch_size'],
                    'epochs': config['epochs'],
                    'learning_rate': config['learning_rate'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time,
                    'model_size': 'Large',
                    'parameters': f"max_len={config['max_len']}, lr={config['learning_rate']}",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.results.append(result)
                
                print(f"准确率: {accuracy:.4f}, F1: {f1:.4f}, 训练时间: {training_time:.2f}s")
                
            except Exception as e:
                print(f"训练 {config['name']} 时出错: {e}")
                continue
    
    def train_rnn_variants(self, X_train, X_test, y_train, y_test):
        """
        训练RNN的不同参数变体
        """
        print("\n=== 训练RNN模型变体 ===")
        
        # 不同的参数组合
        rnn_configs = [
            {
                'name': 'RNN_Small',
                'max_words': 5000,
                'max_len': 50,
                'embedding_dim': 64,
                'lstm_units': 32,
                'batch_size': 64,
                'epochs': 1
            },
            {
                'name': 'RNN_Medium',
                'max_words': 10000,
                'max_len': 100,
                'embedding_dim': 128,
                'lstm_units': 64,
                'batch_size': 128,
                'epochs': 1
            },
            {
                'name': 'RNN_Large',
                'max_words': 15000,
                'max_len': 150,
                'embedding_dim': 256,
                'lstm_units': 128,
                'batch_size': 64,
                'epochs': 2
            }
        ]
        
        for config in rnn_configs:
            print(f"\n训练 {config['name']}...")
            print(f"参数: max_words={config['max_words']}, lstm_units={config['lstm_units']}")
            
            start_time = time.time()
            
            try:
                # 创建tokenizer
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                
                tokenizer = Tokenizer(num_words=config['max_words'], oov_token='<UNK>')
                tokenizer.fit_on_texts(X_train)
                
                # 转换数据
                X_train_seq = tokenizer.texts_to_sequences(X_train)
                X_test_seq = tokenizer.texts_to_sequences(X_test)
                
                X_train_pad = pad_sequences(X_train_seq, maxlen=config['max_len'], padding='post')
                X_test_pad = pad_sequences(X_test_seq, maxlen=config['max_len'], padding='post')
                
                # 创建和训练模型
                model = RNNClassifier(
                    vocab_size=config['max_words'],
                    max_len=config['max_len'],
                    embedding_dim=config['embedding_dim'],
                    lstm_units=config['lstm_units'],
                    num_classes=Config.NUM_CLASSES
                )
                
                model.build_model()
                
                # 训练模型
                history = model.train(
                    X_train_pad, y_train,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    verbose=0
                )
                
                # 预测和评估
                y_pred_proba = model.predict_proba(X_test_pad)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                training_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'experiment_id': self.experiment_id,
                    'model_name': config['name'],
                    'model_type': 'RNN',
                    'max_features': config['max_words'],
                    'max_len': config['max_len'],
                    'embedding_dim': config['embedding_dim'],
                    'batch_size': config['batch_size'],
                    'epochs': config['epochs'],
                    'learning_rate': 'default',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time,
                    'model_size': 'Medium',
                    'parameters': f"max_words={config['max_words']}, lstm_units={config['lstm_units']}",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.results.append(result)
                
                print(f"准确率: {accuracy:.4f}, F1: {f1:.4f}, 训练时间: {training_time:.2f}s")
                
            except Exception as e:
                print(f"训练 {config['name']} 时出错: {e}")
                continue
    
    def save_results_to_csv(self):
        """
        保存结果到CSV文件
        """
        if not self.results:
            print("没有结果可保存")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 按F1分数排序
        df = df.sort_values('f1_score', ascending=False)
        
        # 保存到CSV
        filename = f'parameter_tuning_results_{self.experiment_id}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存到: {filename}")
        
        # 显示最佳结果
        print("\n=== 最佳模型性能 ===")
        best_model = df.iloc[0]
        print(f"最佳模型: {best_model['model_name']}")
        print(f"准确率: {best_model['accuracy']:.4f}")
        print(f"F1分数: {best_model['f1_score']:.4f}")
        print(f"训练时间: {best_model['training_time']:.2f}s")
        print(f"参数: {best_model['parameters']}")
        
        # 显示各模型类型的最佳结果
        print("\n=== 各模型类型最佳结果 ===")
        for model_type in df['model_type'].unique():
            best_of_type = df[df['model_type'] == model_type].iloc[0]
            print(f"{model_type}: {best_of_type['model_name']} (F1: {best_of_type['f1_score']:.4f})")
        
        return filename
    
    def create_summary_report(self):
        """
        创建汇总报告
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # 创建汇总统计
        summary_stats = {
            '模型类型': [],
            '实验数量': [],
            '最佳准确率': [],
            '最佳F1分数': [],
            '平均训练时间': [],
            '最佳模型': []
        }
        
        for model_type in df['model_type'].unique():
            type_df = df[df['model_type'] == model_type]
            
            summary_stats['模型类型'].append(model_type)
            summary_stats['实验数量'].append(len(type_df))
            summary_stats['最佳准确率'].append(type_df['accuracy'].max())
            summary_stats['最佳F1分数'].append(type_df['f1_score'].max())
            summary_stats['平均训练时间'].append(type_df['training_time'].mean())
            
            best_model = type_df.loc[type_df['f1_score'].idxmax(), 'model_name']
            summary_stats['最佳模型'].append(best_model)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # 保存汇总报告
        summary_filename = f'parameter_tuning_summary_{self.experiment_id}.csv'
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n汇总报告已保存到: {summary_filename}")
        
        return summary_filename
    
    def run_parameter_tuning(self):
        """
        运行完整的参数调优实验
        """
        print(f"开始参数调优实验 (ID: {self.experiment_id})")
        print("=" * 50)
        
        try:
            # 加载数据
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # 训练不同模型的变体
            self.train_naive_bayes_variants(X_train, X_test, y_train, y_test)
            self.train_textcnn_variants(X_train, X_test, y_train, y_test)
            self.train_rnn_variants(X_train, X_test, y_train, y_test)
            self.train_bert_variants(X_train, X_test, y_train, y_test)
            
            # 保存结果
            results_file = self.save_results_to_csv()
            summary_file = self.create_summary_report()
            
            print("\n=== 实验完成 ===")
            print(f"详细结果: {results_file}")
            print(f"汇总报告: {summary_file}")
            
            return results_file, summary_file
            
        except Exception as e:
            print(f"实验过程中出错: {e}")
            return None, None

def main():
    """
    主函数
    """
    trainer = ParameterTuningTrainer()
    trainer.run_parameter_tuning()

if __name__ == "__main__":
    main()