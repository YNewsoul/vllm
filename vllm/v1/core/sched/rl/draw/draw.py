import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import argparse

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='绘制vLLM RL训练日志图表')
    parser.add_argument('--log-file', type=str, \
                      default='/home/paperspace/cys/projects/rl/vllm-workspace/vllm/vllm/v1/core/sched/rl/training_logs/2025-11-04/2025-11-04 02:27:23/training_log_2025-11-04 02:27:23.jsonl', \
                      help='训练日志文件路径')
    parser.add_argument('--output_dir', type=str, default=None, \
                      help='图表保存目录，默认为日志文件所在目录')
    parser.add_argument('--step_min', type=int, default=None, \
                      help='起始训练步数，不指定则使用日志中最小步数')
    parser.add_argument('--step_max', type=int, default=None, \
                      help='结束训练步数，不指定则使用日志中最大步数')
    return parser.parse_args()

def plot_training_logs():
    """绘制训练日志图表"""
    # 解析命令行参数
    args = parse_arguments()
    log_file_path = args.log_file
    
    # 设置matplotlib中文字体支持
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 解析日志文件
    train_steps = []
    loss_values = []
    avg_target_q_values = []
    avg_rewards_values = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        train_step = data['train_step']
                        # 根据指定的train_step范围过滤数据
                        if (args.step_min is None or train_step >= args.step_min) and \
                           (args.step_max is None or train_step <= args.step_max):
                            train_steps.append(train_step)
                            loss_values.append(float(data['loss']))
                            avg_target_q_values.append(float(data['avg_target_q']))
                            avg_rewards_values.append(float(data['avg_rewards']))
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        print(f"解析行时出错: {line}, 错误: {e}")
                        continue
        
        if not train_steps:
            print(f"错误: 在指定的train_step范围[{args.step_min}, {args.step_max}]内未找到有效数据")
            return
        
        # 转换为numpy数组便于处理
        train_steps = np.array(train_steps)
        loss_values = np.array(loss_values)
        avg_target_q_values = np.array(avg_target_q_values)
        avg_rewards_values = np.array(avg_rewards_values)
        
        # 确定数据范围
        min_step = min(train_steps)
        max_step = max(train_steps)
        step_range = max_step - min_step
        
        # 确定输出目录
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.dirname(log_file_path)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制Loss图表
        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, loss_values, 'b-', linewidth=1)
        plt.title('Loss varies with the number of training steps')
        plt.xlabel('Training steps ')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴刻度为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 设置合理的x轴范围
        plt.xlim(min_step - step_range * 0.05, max_step + step_range * 0.05)
        
        # 如果loss值跨度很大，可以考虑使用对数坐标
        if max(loss_values) / min(loss_values) > 100:
            plt.yscale('log')
            plt.ylabel('Loss')
        
        # 保存图表
        loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        
        # 2. 绘制Avg Target Q图表
        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, avg_target_q_values, 'g-', linewidth=1)
        plt.title('The variation of avg_target_q with the number of training steps')
        plt.xlabel('Training steps ')
        plt.ylabel('avg_target_q')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴刻度为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 设置合理的x轴范围
        plt.xlim(min_step - step_range * 0.05, max_step + step_range * 0.05)
        
        # 保存图表
        q_plot_path = os.path.join(output_dir, 'avg_target_q_plot.png')
        plt.tight_layout()
        plt.savefig(q_plot_path, dpi=300)
        plt.close()
        
        # 3. 绘制Avg Rewards图表
        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, avg_rewards_values, 'r-', linewidth=1)
        plt.title('avg_rewards changes with the number of training steps')
        plt.xlabel('Training steps ')
        plt.ylabel('avg_rewards')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴刻度为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 设置合理的x轴范围
        plt.xlim(min_step - step_range * 0.05, max_step + step_range * 0.05)
        
        # 添加水平线表示零值
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # 保存图表
        rewards_plot_path = os.path.join(output_dir, 'avg_rewards_plot.png')
        plt.tight_layout()
        plt.savefig(rewards_plot_path, dpi=300)
        plt.close()
        
        print(f"图表已成功生成并保存至:")
        print(f"- Loss图表: {loss_plot_path}")
        print(f"- Avg Target Q图表: {q_plot_path}")
        print(f"- Avg Rewards图表: {rewards_plot_path}")
        
    except FileNotFoundError:
            print(f"错误: 找不到日志文件 '{log_file_path}'")
    except Exception as e:
            print(f"处理日志文件时发生错误: {e}")

if __name__ == "__main__":
    plot_training_logs()