# -*- coding: utf-8 -*-
"""
模型预测模块
"""

import numpy as np
import gradio as gr
from config import Config
from data_process import DataPreprocessor
from models import NaiveBayesClassifier, TextCNNClassifier, RNNClassifier

class ModelPredictor:
    """模型预测器"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.nb_model = None
        self.cnn_model = None
        self.rnn_model = None
        self.label_mapping = None
        self.load_models()
    
    def load_models(self):
        """加载所有模型和预处理工具"""
        print("正在加载模型和预处理工具...")
        
        # 加载预处理工具
        tokenizer, vectorizer, label_mapping = DataPreprocessor.load_preprocessors()
        self.preprocessor.tokenizer = tokenizer
        self.preprocessor.vectorizer = vectorizer
        
        if label_mapping:
            self.preprocessor.label2id = label_mapping['label2id']
            self.preprocessor.id2label = label_mapping['id2label']
            self.label_mapping = label_mapping
        
        # 加载朴素贝叶斯模型
        try:
            self.nb_model = NaiveBayesClassifier.load_model(Config.NB_MODEL_SAVE_PATH)
        except Exception as e:
            print(f"朴素贝叶斯模型加载失败: {e}")
        
        # 加载TextCNN模型
        try:
            self.cnn_model = TextCNNClassifier.load_model(Config.TEXTCNN_MODEL_PATH)
        except Exception as e:
            print(f"TextCNN模型加载失败: {e}")
        
        # 加载RNN模型
        try:
            self.rnn_model = RNNClassifier.load_model(Config.RNN_MODEL_PATH)
        except Exception as e:
            print(f"RNN模型加载失败: {e}")
        
        print("模型加载完成")
    
    def preprocess_single_text(self, text):
        """预处理单个文本"""
        return self.preprocessor.clean_text(text)
    
    def predict_naive_bayes(self, text):
        """朴素贝叶斯预测"""
        if self.nb_model is None or self.preprocessor.vectorizer is None:
            return "模型未加载", 0.0
        
        try:
            # 预处理文本
            processed_text = self.preprocess_single_text(text)
            
            # 转换为TF-IDF向量
            text_vector = self.preprocessor.vectorizer.transform([processed_text])
            
            # 预测
            pred_id = self.nb_model.predict(text_vector)[0]
            pred_proba = self.nb_model.predict_proba(text_vector)[0]
            
            # 获取标签和置信度
            pred_label = self.preprocessor.id2label[pred_id]
            confidence = pred_proba[pred_id]
            
            return pred_label, confidence
        except Exception as e:
            return f"预测失败: {e}", 0.0
    
    def predict_textcnn(self, text):
        """TextCNN预测"""
        if self.cnn_model is None or self.preprocessor.tokenizer is None:
            return "模型未加载", 0.0
        
        try:
            # 预处理文本
            processed_text = self.preprocess_single_text(text)
            
            # 转换为序列
            text_seq = self.preprocessor.tokenizer.texts_to_sequences([processed_text])
            text_padded = self.preprocessor.texts_to_sequences([processed_text])
            
            # 预测
            pred_proba = self.cnn_model.predict_proba(text_padded)
            pred_id = np.argmax(pred_proba[0])
            
            # 获取标签和置信度
            pred_label = self.preprocessor.id2label[pred_id]
            confidence = pred_proba[0][pred_id]
            
            return pred_label, confidence
        except Exception as e:
            return f"预测失败: {e}", 0.0
    
    def predict_rnn(self, text):
        """RNN预测"""
        if self.rnn_model is None or self.preprocessor.tokenizer is None:
            return "模型未加载", 0.0
        
        try:
            # 预处理文本
            processed_text = self.preprocess_single_text(text)
            
            # 转换为序列
            text_padded = self.preprocessor.texts_to_sequences([processed_text])
            
            # 预测
            pred_proba = self.rnn_model.predict_proba(text_padded)
            pred_id = np.argmax(pred_proba[0])
            
            # 获取标签和置信度
            pred_label = self.preprocessor.id2label[pred_id]
            confidence = pred_proba[0][pred_id]
            
            return pred_label, confidence
        except Exception as e:
            return f"预测失败: {e}", 0.0
    
    def predict_all_models(self, text):
        """使用所有模型进行预测"""
        if not text.strip():
            return "请输入新闻标题"
        
        # 朴素贝叶斯预测
        nb_label, nb_conf = self.predict_naive_bayes(text)
        
        # TextCNN预测
        cnn_label, cnn_conf = self.predict_textcnn(text)
        
        # RNN预测
        rnn_label, rnn_conf = self.predict_rnn(text)
        
        # 格式化结果
        results = {
            '朴素贝叶斯分类结果': f"{nb_label} (置信度: {nb_conf:.4f})",
            'TextCNN分类结果': f"{cnn_label} (置信度: {cnn_conf:.4f})",
            'RNN分类结果': f"{rnn_label} (置信度: {rnn_conf:.4f})"
        }
        
        return results

def create_gradio_interface():
    """创建Gradio界面"""
    predictor = ModelPredictor()
    
    # 创建界面
    iface = gr.Interface(
        fn=predictor.predict_all_models,
        inputs=gr.Textbox(
            label="输入新闻标题",
            placeholder="请输入要分类的新闻标题...",
            lines=2
        ),
        outputs=gr.JSON(label="各模型分类结果"),
        title="新闻分类模型预测系统",
        description="输入新闻标题，查看朴素贝叶斯、TextCNN和RNN三种模型的分类结果",
        examples=[
            ["苹果发布新款iPhone手机"],
            ["中国足球队获得世界杯冠军"],
            ["股市今日大涨创历史新高"],
            ["科学家发现新的治疗癌症方法"],
            ["明星结婚引发网友热议"],
            ["新能源汽车销量持续增长"]
        ],
        theme=gr.themes.Soft()
    )
    
    return iface

if __name__ == "__main__":
    # 创建并启动Gradio界面
    interface = create_gradio_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )