# trainer/train_simple.py
"""
简化的训练脚本，避免多进程问题
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import json
from transformers import AutoTokenizer, CLIPProcessor
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
import argparse
import time


class SimpleTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.setup_model()

        # 初始化数据
        self.setup_data()

        # 初始化优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        print(f"训练设备: {self.device}")
        print(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.3f} 百万")

    def setup_model(self):
        """初始化模型"""
        config = VLMConfig(
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_hidden_layers,
            max_seq_len=self.args.max_seq_len,
            use_moe=bool(self.args.use_moe)
        )

        # 加载预训练权重
        self.model = MiniMindVLM(config)

        if self.args.from_weight != 'none':
            try:
                weight_path = f"../checkpoints/{self.args.from_weight}.pth"
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location='cpu')
                    # 过滤掉不匹配的键
                    model_dict = self.model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"加载预训练权重: {weight_path}")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")

        self.model = self.model.to(self.device)
        self.model.train()

    def setup_data(self):
        """初始化数据集"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # 添加特殊token
        special_tokens = {"additional_special_tokens": ["@" * 196]}
        tokenizer.add_special_tokens(special_tokens)

        # 加载数据集
        self.dataset = VLMDataset(
            self.args.data_path,
            tokenizer,
            max_length=self.args.max_seq_len
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,  # Windows必须为0
            pin_memory=True
        )

        print(f"数据集大小: {len(self.dataset)}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        total_loss = 0
        start_time = time.time()

        for batch_idx, (X, Y, loss_mask, pixel_values) in enumerate(self.dataloader):
            # 移动到设备
            X = X.to(self.device)
            Y = Y.to(self.device)
            loss_mask = loss_mask.to(self.device)
            pixel_values = pixel_values.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X, pixel_values=pixel_values)

            # 计算损失
            loss = self.loss_fn(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

            # 打印日志
            if batch_idx % self.args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.args.batch_size / elapsed

                print(f"Epoch {epoch + 1} [{batch_idx + 1}/{len(self.dataloader)}] "
                      f"Loss: {loss.item():.4f} Avg: {avg_loss:.4f} "
                      f"Speed: {samples_per_sec:.1f} samples/sec")

            # 保存检查点
            if (batch_idx + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, batch_idx)

        return total_loss / len(self.dataloader)

    def save_checkpoint(self, epoch, batch_idx):
        """保存检查点"""
        os.makedirs(self.args.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'batch': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
        }

        checkpoint_path = os.path.join(
            self.args.save_dir,
            f"{self.args.save_weight}_epoch{epoch + 1}_batch{batch_idx + 1}.pt"
        )

        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def train(self):
        """主训练循环"""
        print("开始训练...")

        for epoch in range(self.args.epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print(f"{'=' * 50}")

            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

            # 保存每个epoch的模型
            self.save_checkpoint(epoch, len(self.dataloader) - 1)

        print("\n训练完成！")

        # 保存最终模型
        final_path = os.path.join(self.args.save_dir, f"{self.args.save_weight}_final.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"最终模型已保存: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="简单训练脚本")

    # 训练配置
    parser.add_argument("--save_dir", type=str, default="./output", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="minimind_v", help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批大小")  # Windows上减小batch_size
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")

    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层大小")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="层数")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度")  # 减小序列长度
    parser.add_argument("--use_moe", type=int, default=0, help="是否使用MoE")

    # 数据配置
    parser.add_argument("--data_path", type=str, default="../dataset/sft_data.parquet", help="数据路径")

    # 训练参数
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--from_weight", type=str, default="none", help="预训练权重")

    args = parser.parse_args()

    # 创建训练器
    trainer = SimpleTrainer(args)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()