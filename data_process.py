# -*- coding: utf-8 -*-
"""
数据预处理模块
"""

import re
import pandas as pd
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
from config import Config

class DataPreprocessor:
    def __init__(self, tokenizer=None, vectorizer=None):
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.label2id = {}
        self.id2label = {}
        
    def clean_text(self, text):
        """文本清洗和分词"""
        # 清洗文本，仅保留中文
        text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
        # 分词
        tokens = jieba.lcut(text)
        return ' '.join(tokens)
    
    def load_data(self, file_path):
        """加载数据"""
        try:
            data = pd.read_csv(file_path)
            print(f"数据加载成功，共{len(data)}条记录")
            return data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def create_label_mapping(self, labels):
        """创建标签映射"""
        unique_labels = labels.unique()
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for idx, label in enumerate(unique_labels)}
        Config.NUM_CLASSES = len(unique_labels)
        print(f"标签映射创建完成，共{Config.NUM_CLASSES}个类别")
        return self.label2id, self.id2label
    
    def preprocess_data(self, data, text_column='题目', label_column='类别'):
        """预处理数据"""
        print("开始数据预处理...")
        
        # 创建标签映射
        self.create_label_mapping(data[label_column])
        
        # 文本预处理
        texts = data[text_column].apply(self.clean_text)
        
        # 标签转换
        labels = data[label_column].map(self.label2id)
        
        print("数据预处理完成")
        return texts, labels
    
    def split_data(self, texts, labels):
        """划分数据集"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=Config.TEST_SPLIT,
            random_state=1234,
            stratify=labels
        )
        print(f"数据集划分完成: 训练集{len(X_train)}条，测试集{len(X_test)}条")
        return X_train, X_test, y_train, y_test
    
    def create_tokenizer(self, texts):
        """创建并训练tokenizer"""
        if not self.tokenizer:
            self.tokenizer = Tokenizer(
                num_words=Config.MAX_WORDS,
                oov_token='<UNK>'
            )
        
        self.tokenizer.fit_on_texts(texts)
        print(f"Tokenizer创建完成，词汇表大小: {len(self.tokenizer.word_index)}")
        return self.tokenizer
    
    def texts_to_sequences(self, texts):
        """文本转序列并填充"""
        if not self.tokenizer:
            raise ValueError("Tokenizer未初始化")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=Config.MAX_LEN,
            padding='post',
            truncating='post'
        )
        return padded_sequences
    
    def create_vectorizer(self, texts):
        """创建TF-IDF向量化器"""
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(max_features=5000)
        
        self.vectorizer.fit(texts)
        print("TF-IDF向量化器创建完成")
        return self.vectorizer
    
    def texts_to_tfidf(self, texts):
        """文本转TF-IDF向量"""
        if not self.vectorizer:
            raise ValueError("Vectorizer未初始化")
        
        return self.vectorizer.transform(texts)
    
    def save_preprocessors(self):
        """保存预处理工具"""
        os.makedirs(os.path.dirname(Config.TOKENIZER_SAVE_PATH), exist_ok=True)
        
        if self.tokenizer:
            joblib.dump(self.tokenizer, Config.TOKENIZER_SAVE_PATH)
            print(f"Tokenizer已保存到: {Config.TOKENIZER_SAVE_PATH}")
        
        if self.vectorizer:
            joblib.dump(self.vectorizer, Config.VECTORIZER_SAVE_PATH)
            print(f"Vectorizer已保存到: {Config.VECTORIZER_SAVE_PATH}")
        
        # 保存标签映射
        label_mapping = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        joblib.dump(label_mapping, Config.LABELS_SAVE_PATH)
        print(f"标签映射已保存到: {Config.LABELS_SAVE_PATH}")
    
    @staticmethod
    def load_preprocessors():
        """加载预处理工具"""
        tokenizer = None
        vectorizer = None
        label_mapping = None
        
        try:
            tokenizer = joblib.load(Config.TOKENIZER_SAVE_PATH)
            print("Tokenizer加载成功")
        except:
            print("Tokenizer加载失败")
        
        try:
            vectorizer = joblib.load(Config.VECTORIZER_SAVE_PATH)
            print("Vectorizer加载成功")
        except:
            print("Vectorizer加载失败")
        
        try:
            label_mapping = joblib.load(Config.LABELS_SAVE_PATH)
            print("标签映射加载成功")
        except:
            print("标签映射加载失败")
        
        return tokenizer, vectorizer, label_mapping