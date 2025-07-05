# -*- coding: utf-8 -*-
"""
新闻分类项目主程序
"""

import os
import argparse
from train import ModelTrainer
from evaluate import ModelEvaluator
from predict import ModelPredictor
from config import Config

def create_directories():
    """创建必要的目录"""
    directories = [
        Config.MODEL_SAVE_DIR,
        Config.PREPROCESSOR_SAVE_DIR,
        Config.PLOT_SAVE_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")

def train_models():
    """训练所有模型"""
    print("开始训练模型...")
    
    # 创建必要目录
    create_directories()
    
    # 初始化训练器
    trainer = ModelTrainer()
    
    # 准备数据
    trainer.prepare_data()
    
    # 训练所有模型
    trainer.train_all_models()
    
    print("所有模型训练完成！")

def evaluate_models():
    """评估所有模型"""
    print("开始评估模型...")
    
    # 初始化评估器
    evaluator = ModelEvaluator()
    
    # 评估所有模型
    results = evaluator.evaluate_all_models()
    
    print("模型评估完成！")
    return results

def predict_interface():
    """启动预测界面"""
    print("启动预测界面...")
    
    # 初始化预测器
    predictor = ModelPredictor()
    
    # 加载模型
    predictor.load_models()
    
    # 创建并启动Gradio界面
    interface = predictor.create_gradio_interface()
    interface.launch(share=True)

def interactive_predict():
    """交互式预测"""
    print("进入交互式预测模式...")
    print("输入 'quit' 退出")
    
    # 初始化预测器
    predictor = ModelPredictor()
    predictor.load_models()
    
    while True:
        text = input("\n请输入新闻标题: ").strip()
        
        if text.lower() == 'quit':
            print("退出预测模式")
            break
        
        if not text:
            print("请输入有效的文本")
            continue
        
        try:
            # 获取所有模型的预测结果
            results = predictor.predict_all_models(text)
            
            print("\n预测结果:")
            print("-" * 40)
            for model_name, result in results.items():
                print(f"{model_name}: {result['category']} (置信度: {result['confidence']:.4f})")
            
        except Exception as e:
            print(f"预测出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='新闻分类项目')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'interactive', 'all'],
                       default='all', help='运行模式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("新闻分类项目")
    print("=" * 60)
    
    if args.mode == 'train':
        train_models()
    
    elif args.mode == 'evaluate':
        evaluate_models()
    
    elif args.mode == 'predict':
        predict_interface()
    
    elif args.mode == 'interactive':
        interactive_predict()
    
    elif args.mode == 'all':
        # 完整流程：训练 -> 评估 -> 预测界面
        print("执行完整流程: 训练 -> 评估 -> 预测界面")
        
        # 1. 训练模型
        train_models()
        
        # 2. 评估模型
        evaluate_models()
        
        # 3. 询问是否启动预测界面
        choice = input("\n是否启动预测界面? (y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            predict_interface()
        else:
            print("程序结束")
    
    else:
        print(f"未知模式: {args.mode}")
        parser.print_help()

if __name__ == "__main__":
    main()