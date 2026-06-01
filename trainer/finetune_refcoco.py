# trainer/finetune_refcoco.py
"""
使用sft_vlm_512.pth作为基础，微调refcoco数据集的专用脚本
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.refcoco_dataset import RefCOCODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, \
    vlm_checkpoint, SkipBatchSampler

warnings.filterwarnings('ignore')


class RefCOCOFinetuner:

    def __init__(self, args):
        self.args = args
        self.experiment_dir = None
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.metrics = {
            'loss_history': [],
            'best_loss': float('inf'),
            'best_epoch': 0
        }

    def setup(self):
        local_rank = init_distributed_mode()
        if dist.is_initialized():
            self.args.device = f"cuda:{local_rank}"

        setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

        self.create_experiment_dir()

        self.setup_model()

        self.setup_data()

        self.setup_optimizer()

        Logger(f"\n{'=' * 60}")
        Logger(f"RefCOCO微调配置")
        Logger(f"基础模型: {self.args.from_weight}")
        Logger(f"数据集: {self.args.dataset_type}")
        Logger(f"学习率: {self.args.learning_rate}")
        Logger(f"Batch size: {self.args.batch_size}")
        Logger(f"Epochs: {self.args.epochs}")
        Logger(f"实验目录: {self.experiment_dir}")
        Logger(f"{'=' * 60}\n")

    def create_experiment_dir(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        folder_name = f"refcoco_finetune_{self.args.dataset_type}_{timestamp}"
        self.experiment_dir = os.path.join(self.args.save_dir, folder_name)

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)

        import json
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)

    def setup_model(self):
        vlm_config = VLMConfig(
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_hidden_layers,
            max_seq_len=self.args.max_seq_len,
            use_moe=bool(self.args.use_moe)
        )

        Logger(f"正在加载基础模型: {self.args.from_weight}")
        self.model, tokenizer, preprocess = init_vlm_model(
            vlm_config,
            from_weight=self.args.from_weight,
            device=self.args.device
        )

        if self.args.freeze_vision:
            Logger("冻结视觉编码器参数")
            for name, param in self.model.named_parameters():
                if 'vision_encoder' in name or 'vision_proj' in name:
                    param.requires_grad = False

        if self.args.train_lm_only:
            Logger("仅训练语言模型部分")
            for name, param in self.model.named_parameters():
                if 'vision' in name:
                    param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        Logger(f"总参数: {total_params / 1e6:.2f}M, 可训练参数: {trainable_params / 1e6:.2f}M")

        self.tokenizer = tokenizer
        self.preprocess = preprocess

        return tokenizer, preprocess

    def setup_data(self):
        Logger(f"加载 {self.args.dataset_type} 数据集")

        self.train_ds = RefCOCODataset(
            image_dir=self.args.image_dir,
            ann_file=self.args.ann_file,
            ref_file=self.args.ref_file,
            dataset_type=self.args.dataset_type,
            tokenizer=self.tokenizer,
            processor=self.preprocess,
            max_length=self.args.max_seq_len,
            image_special_token='@' * 196
        )

        Logger(f"数据集大小: {len(self.train_ds)}")

        train_sampler = None
        if dist.is_initialized():
            train_sampler = DistributedSampler(self.train_ds)

        self.train_sampler = train_sampler

        num_workers = 0
        if sys.platform != "win32":
            num_workers = self.args.num_workers

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

    def setup_optimizer(self):
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype == 'float16'))

    def train_epoch(self, epoch):
        self.model.train()
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        epoch_loss = 0
        total_steps = len(self.train_loader)

        Logger(f"\nEpoch {epoch + 1}/{self.args.epochs}")

        for step, batch in enumerate(self.train_loader):
            if len(batch) == 5:
                X, Y, loss_mask, pixel_values, bbox_labels = batch
            else:
                X, Y, loss_mask, pixel_values = batch
                bbox_labels = None

            if isinstance(X, str):
                Logger(f"错误: X 是字符串，不是张量")
                continue

            X = X.to(self.args.device)
            Y = Y.to(self.args.device)
            loss_mask = loss_mask.to(self.args.device)
            pixel_values = pixel_values.to(self.args.device)

            lr = get_lr(epoch * total_steps + step, self.args.epochs * total_steps, self.args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with torch.cuda.amp.autocast(enabled=(self.args.dtype != 'float32')):
                if bbox_labels is not None and hasattr(self.model, 'forward_with_bbox'):
                    bbox_labels = bbox_labels.to(self.args.device)
                    res = self.model.forward_with_bbox(
                        X,
                        pixel_values=pixel_values,
                        bbox_labels=bbox_labels
                    )
                else:
                    res = self.model(X, pixel_values=pixel_values)

                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())

                loss = (loss * loss_mask).sum() / loss_mask.sum()
                if hasattr(res, 'aux_loss'):
                    loss += res.aux_loss

                loss = loss / self.args.accumulation_steps

            self.scaler.scale(loss).backward()

            grad_norm = None
            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item() * self.args.accumulation_steps
            epoch_loss += current_loss

            if step % self.args.log_interval == 0 or step == total_steps - 1:
                avg_loss = epoch_loss / (step + 1)
                Logger(f"  Step {step + 1}/{total_steps}, Loss: {current_loss:.4f}, Avg: {avg_loss:.4f}, LR: {lr:.2e}")

            del X, Y, loss_mask, pixel_values
            if 'res' in locals():
                del res
            del loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / total_steps
        self.metrics['loss_history'].append(avg_epoch_loss)

        return avg_epoch_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint_name = f"checkpoint_epoch{epoch + 1}_loss{loss:.4f}"

        if is_best:
            checkpoint_name = f"best_model"

        checkpoint_path = os.path.join(self.experiment_dir, 'checkpoints', f'{checkpoint_name}.pth')

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        clean_state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith('vision_encoder.')
        }

        torch.save(clean_state_dict, checkpoint_path)

        full_checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'metrics': self.metrics
        }

        full_path = os.path.join(self.experiment_dir, 'checkpoints', f'{checkpoint_name}_full.pt')
        torch.save(full_checkpoint, full_path)

        Logger(f"检查点已保存: {checkpoint_path}")

        return checkpoint_path

    def finetune(self):
        Logger("开始RefCOCO微调...")

        if dist.is_initialized():
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(self.model, device_ids=[int(os.environ['LOCAL_RANK'])])

        for epoch in range(self.args.epochs):
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            avg_loss = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            if avg_loss < self.metrics['best_loss']:
                self.metrics['best_loss'] = avg_loss
                self.metrics['best_epoch'] = epoch + 1
                self.save_checkpoint(epoch, avg_loss, is_best=True)

            if (epoch + 1) % self.args.save_interval_epochs == 0:
                self.save_checkpoint(epoch, avg_loss)

            Logger(f"\nEpoch {epoch + 1} 完成")
            Logger(f"平均损失: {avg_loss:.6f}")
            Logger(f"最佳损失: {self.metrics['best_loss']:.6f} (epoch {self.metrics['best_epoch']})")
            Logger(f"Epoch用时: {epoch_time / 60:.2f}分钟")

        final_path = self.save_checkpoint(self.args.epochs - 1, avg_loss)
        Logger(f"\n{'=' * 60}")
        Logger(f"微调完成!")
        Logger(f"最佳模型: epoch {self.metrics['best_epoch']}, 损失: {self.metrics['best_loss']:.6f}")
        Logger(f"最终模型: {final_path}")
        Logger(f"{'=' * 60}")

        self.save_training_report()


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V RefCOCO微调")

    parser.add_argument("--save_dir", type=str, default="../experiments/refcoco_finetune", help="保存目录")
    parser.add_argument("--epochs", type=int, default=10, help="微调轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="微调学习率")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="数据类型")

    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层大小")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="层数")
    parser.add_argument("--max_seq_len", type=int, default=1536, help="最大序列长度")
    parser.add_argument("--use_moe", type=int, default=0, help="是否使用MoE")

    parser.add_argument("--dataset_type", type=str, default="refcoco", choices=["refcoco", "refcocog", "refcoco+"])
    parser.add_argument("--image_dir", type=str, default="../dataset/refcoco/images", help="图像目录")
    parser.add_argument("--ann_file", type=str, default="../dataset/refcoco/instances.json", help="标注文件")
    parser.add_argument("--ref_file", type=str, default="../dataset/refcoco/refs(unc).p", help="指代表达文件")

    parser.add_argument("--from_weight", type=str, default="sft_vlm", help="基础模型权重")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="检查点目录")

    parser.add_argument("--freeze_vision", action="store_true", help="冻结视觉编码器")
    parser.add_argument("--train_lm_only", action="store_true", help="只训练语言模型")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")

    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=50, help="日志间隔")
    parser.add_argument("--save_interval_epochs", type=int, default=2, help="保存间隔(epoch)")

    args = parser.parse_args()

    finetuner = RefCOCOFinetuner(args)
    finetuner.setup()
    finetuner.finetune()


if __name__ == "__main__":
    main()