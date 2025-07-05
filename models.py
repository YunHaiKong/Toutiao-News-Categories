# -*- coding: utf-8 -*-
"""
模型定义模块
"""

import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, LSTM, Bidirectional, Input, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from config import Config

class NaiveBayesClassifier:
    """朴素贝叶斯分类器"""
    
    def __init__(self):
        self.model = MultinomialNB()
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("朴素贝叶斯模型训练完成")
    
    def predict(self, X):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)
    
    def save_model(self, path):
        """保存模型"""
        joblib.dump(self.model, path)
        print(f"朴素贝叶斯模型已保存到: {path}")
    
    @staticmethod
    def load_model(path):
        """加载模型"""
        try:
            model = joblib.load(path)
            classifier = NaiveBayesClassifier()
            classifier.model = model
            classifier.is_trained = True
            print(f"朴素贝叶斯模型加载成功: {path}")
            return classifier
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None

class BertClassifier:
    """BERT分类器"""
    
    def __init__(self, model_name=None, max_len=None, num_classes=None):
        self.model_name = model_name or Config.BERT_MODEL_NAME
        self.max_len = max_len or Config.BERT_MAX_LEN
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.tokenizer = None
        self.model = None
        self.is_trained = False
    
    # 在BertClassifier类的load_tokenizer方法中添加镜像配置
    def load_tokenizer(self):
        """加载BERT tokenizer"""
        try:
            # 使用国内镜像
            import os
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_name,
                mirror='https://hf-mirror.com'
            )
            print(f"BERT tokenizer加载成功: {self.model_name}")
        except Exception as e:
            print(f"BERT tokenizer加载失败: {e}")
            raise e
        return self.tokenizer
    
    def build_model(self):
        """构建BERT微调模型"""
        # 加载预训练BERT模型
        bert_model = TFBertModel.from_pretrained(self.model_name)
        
        # 构建分类模型
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name='attention_mask')
        
        # BERT编码
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        
        # 分类层
        x = Dropout(0.3)(pooled_output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # 创建模型
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=Config.BERT_LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("BERT模型构建完成")
        return model
    
    def encode_texts(self, texts):
        """编码文本为BERT输入格式"""
        if self.tokenizer is None:
            self.load_tokenizer()
        
        encoded = self.tokenizer(
            texts.tolist(),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def train(self, X_train_text, y_train, X_val_text=None, y_val=None, callbacks=None):
        """训练BERT模型"""
        if self.model is None:
            self.build_model()
        
        # 编码训练数据
        train_encoded = self.encode_texts(X_train_text)
        
        # 准备验证数据
        validation_data = None
        if X_val_text is not None and y_val is not None:
            val_encoded = self.encode_texts(X_val_text)
            validation_data = ([val_encoded['input_ids'], val_encoded['attention_mask']], y_val)
        
        # 训练模型
        history = self.model.fit(
            [train_encoded['input_ids'], train_encoded['attention_mask']],
            y_train,
            batch_size=Config.BERT_BATCH_SIZE,
            epochs=Config.BERT_EPOCHS,
            validation_data=validation_data,
            validation_split=Config.VALIDATION_SPLIT if validation_data is None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("BERT模型训练完成")
        return history
    
    def predict(self, X_text):
        """预测"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        
        encoded = self.encode_texts(X_text)
        predictions = self.model.predict(
            [encoded['input_ids'], encoded['attention_mask']], 
            verbose=0
        )
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X_text):
        """预测概率"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        
        encoded = self.encode_texts(X_text)
        return self.model.predict(
            [encoded['input_ids'], encoded['attention_mask']], 
            verbose=0
        )
    
    def evaluate(self, X_test_text, y_test):
        """评估模型"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        
        encoded = self.encode_texts(X_test_text)
        return self.model.evaluate(
            [encoded['input_ids'], encoded['attention_mask']], 
            y_test, 
            verbose=0
        )
    
    def save_model(self, model_path, tokenizer_path=None):
        """保存模型和tokenizer"""
        if self.model is None:
            raise ValueError("模型未构建")
        
        # 保存模型
        self.model.save(model_path)
        print(f"BERT模型已保存到: {model_path}")
        
        # 保存tokenizer
        if tokenizer_path and self.tokenizer:
            self.tokenizer.save_pretrained(tokenizer_path)
            print(f"BERT tokenizer已保存到: {tokenizer_path}")
    
    @staticmethod
    def load_model(model_path, tokenizer_path=None, model_name=None):
        """加载模型和tokenizer"""
        try:
            # 加载模型
            model = load_model(model_path, custom_objects={'TFBertModel': TFBertModel})
            
            # 创建分类器实例
            classifier = BertClassifier(model_name=model_name)
            classifier.model = model
            classifier.is_trained = True
            
            # 加载tokenizer
            if tokenizer_path:
                classifier.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            else:
                classifier.load_tokenizer()
            
            print(f"BERT模型加载成功: {model_path}")
            return classifier
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            return None

class TextCNNClassifier:
    """TextCNN分类器"""
    
    def __init__(self, vocab_size=None, max_len=None, embedding_dim=None, num_classes=None):
        self.vocab_size = vocab_size or Config.MAX_WORDS
        self.max_len = max_len or Config.MAX_LEN
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.model = None
        self.is_trained = False
    
    def build_model(self):
        """构建TextCNN模型"""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len
            ),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            GlobalMaxPool1D(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        self.model = model
        print("TextCNN模型构建完成")
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, callbacks=None, epochs=None, batch_size=None, verbose=1):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size or Config.BATCH_SIZE,
            epochs=epochs or Config.EPOCHS,
            validation_data=validation_data,
            validation_split=Config.VALIDATION_SPLIT if validation_data is None else None,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        print("TextCNN模型训练完成")
        return history
    
    def predict(self, X):
        """预测"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未构建")
        self.model.save(path)
        print(f"TextCNN模型已保存到: {path}")
    
    @staticmethod
    def load_model(path):
        """加载模型"""
        try:
            model = load_model(path)
            classifier = TextCNNClassifier()
            classifier.model = model
            classifier.is_trained = True
            print(f"TextCNN模型加载成功: {path}")
            return classifier
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None

class RNNClassifier:
    """RNN分类器"""
    
    def __init__(self, vocab_size=None, max_len=None, embedding_dim=None, num_classes=None, lstm_units=64, rnn_type='lstm'):
        self.vocab_size = vocab_size or Config.MAX_WORDS
        self.max_len = max_len or Config.MAX_LEN
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.lstm_units = lstm_units
        self.rnn_type = rnn_type.lower()
        self.model = None
        self.is_trained = False
    
    def build_model(self):
        """构建RNN模型"""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len
            ),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Bidirectional(LSTM(self.lstm_units // 2)),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        self.model = model
        print("RNN模型构建完成")
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, callbacks=None, epochs=None, batch_size=None, verbose=1):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size or Config.BATCH_SIZE,
            epochs=epochs or Config.EPOCHS,
            validation_data=validation_data,
            validation_split=Config.VALIDATION_SPLIT if validation_data is None else None,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        print("RNN模型训练完成")
        return history
    
    def predict(self, X):
        """预测"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练")
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未构建")
        self.model.save(path)
        print(f"RNN模型已保存到: {path}")
    
    @staticmethod
    def load_model(path):
        """加载模型"""
        try:
            model = load_model(path)
            classifier = RNNClassifier()
            classifier.model = model
            classifier.is_trained = True
            print(f"RNN模型加载成功: {path}")
            return classifier
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None