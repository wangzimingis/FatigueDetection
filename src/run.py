#!/usr/bin/env python3
"""
驾驶疲劳评估系统运行脚本
"""

import subprocess
import sys
import os


def run_command(command):
    """运行命令并打印输出"""
    print(f"执行命令: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.stdout:
        print("输出:")
        print(result.stdout)

    if result.stderr:
        print("错误:")
        print(result.stderr)

    return result.returncode


def main():
    """主函数"""
    print("=" * 60)
    print("驾驶疲劳评估系统 - SEED-VIG数据集")
    print("=" * 60)

    # 安装依赖
    print("\n1. 安装依赖...")
    run_command("pip install -r requirements.txt")

    # 创建目录结构
    print("\n2. 创建目录结构...")
    os.makedirs("./data/SEED-VIG", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)

    print("\n目录结构创建完成:")
    print("  ./data/SEED-VIG/ - 存放SEED-VIG数据集")
    print("  ./checkpoints/   - 存放训练好的模型")
    print("  ./logs/          - 存放训练日志")
    print("  ./results/       - 存放实验结果")
    print("  ./figures/       - 存放图表")

    print("\n3. 使用说明:")
    print("\n  基本命令:")
    print("    python main.py --analyze_features --subject_id 1")
    print("    python main.py --train --subject_id 1")
    print("    python main.py --train --cross_validation --subject_id 1")
    print("    python main.py --evaluate --model_path ./checkpoints/model_xxx.pth")

    print("\n  参数说明:")
    print("    --subject_id: 被试者ID (1-23)")
    print("    --feature_type: 特征类型 (2Hz 或 5Bands)")
    print("    --use_eog: 使用EOG特征")
    print("    --use_multimodal: 使用多模态融合")
    print("    --cross_validation: 使用5折交叉验证")
    print("    --epochs: 训练轮数")
    print("    --batch_size: 批大小")
    print("    --learning_rate: 学习率")

    print("\n4. 示例运行:")
    print("\n  示例1: 分析被试者1的特征")
    print("    python main.py --analyze_features --subject_id 1")

    print("\n  示例2: 训练单模态模型")
    print("    python main.py --train --subject_id 1 --epochs 50 --batch_size 32")

    print("\n  示例3: 训练多模态模型")
    print("    python main.py --train --subject_id 1 --use_eog --use_multimodal")

    print("\n  示例4: 使用5折交叉验证训练")
    print("    python main.py --train --cross_validation --subject_id 1")

    print("\n5. 注意事项:")
    print("  - 请确保SEED-VIG数据集已放置在 ./data/SEED-VIG/ 目录下")
    print("  - 首次运行前请安装依赖: pip install -r requirements.txt")
    print("  - 如果出现文件找不到错误，请检查文件命名和路径")

    print("\n" + "=" * 60)
    print("脚本执行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()