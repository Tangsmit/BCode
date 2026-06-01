# trainer/train_vlm_refcoco.py
import os
import sys
import json
import time
import numpy as np
from datetime import datetime

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split
from transformers import AutoTokenizer

from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from dataset.refcoco_dataset import RefCOCODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, \
    vlm_checkpoint, SkipBatchSampler

warnings.filterwarnings('ignore')


class TrainingMetrics:

    def __init__(self):
        self.metrics = {
            'loss': [],
            'perplexity': [],
            'learning_rate': [],
            'grad_norm': [],
            'training_time': [],
            'best_loss': float('inf'),
        }
        self.current_epoch = 0
        self.start_time = time.time()

    def update(self, loss, lr, grad_norm=None):
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(lr)

        perplexity = np.exp(min(loss, 10)) 
        self.metrics['perplexity'].append(perplexity)

        if grad_norm is not None:
            self.metrics['grad_norm'].append(grad_norm)

        if loss < self.metrics['best_loss']:
            self.metrics['best_loss'] = loss
            return True  
        else:
            return False  

    def get_summary(self):
        total_time = time.time() - self.start_time
        self.metrics['training_time'].append(total_time)

        summary = {
            'current_epoch': self.current_epoch,
            'avg_loss': np.mean(self.metrics['loss'][-100:]) if self.metrics['loss'] else 0,
            'best_loss': self.metrics['best_loss'],
            'avg_perplexity': np.mean(self.metrics['perplexity'][-100:]) if self.metrics['perplexity'] else 0,
            'current_lr': self.metrics['learning_rate'][-1] if self.metrics['learning_rate'] else 0,
            'total_time_hours': total_time / 3600,
            'grad_norm_avg': np.mean(self.metrics['grad_norm'][-100:]) if self.metrics['grad_norm'] else 0
        }
        return summary

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load(self, load_path):
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                self.metrics = json.load(f)


def create_experiment_folder(base_dir, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hash(frozenset(config.items())) % 10000

    folder_name = f"exp_{timestamp}_{config_hash:04d}"
    experiment_dir = os.path.join(base_dir, folder_name)

    os.makedirs(experiment_dir, exist_ok=True)

    subdirs = ['checkpoints', 'logs', 'metrics', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    return experiment_dir


def save_config(args, experiment_dir):
    config_path = os.path.join(experiment_dir, 'configs', 'training_config.json')
    config_dict = vars(args)

    config_dict['experiment_start_time'] = datetime.now().isoformat()
    config_dict['experiment_dir'] = experiment_dir

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    Logger(f"训练配置已保存到: {config_path}")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, task_type='sft', metrics=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    epoch_losses = []

    for step, batch in enumerate(loader, start=start_step + 1):
        if task_type == 'referring':
            X, Y, loss_mask, pixel_values, bbox_labels = batch
            bbox_labels = bbox_labels.to(args.device) if bbox_labels is not None else None
        else:
            X, Y, loss_mask, pixel_values = batch

        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            if task_type == 'referring':
                res = model(X, pixel_values=pixel_values, task_type='referring')
            else:
                res = model(X, pixel_values=pixel_values)

            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        grad_norm = None
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * args.accumulation_steps
        epoch_losses.append(current_loss)

        if metrics and step % args.log_interval == 0:
            has_improved = metrics.update(current_loss, lr, grad_norm)
            if has_improved and step > 0:
                Logger(f"Step {step}: 损失改进到 {current_loss:.6f}")

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            log_msg = (f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) '
                       f'loss:{current_loss:.6f} lr:{current_lr:.12f} '
                       f'task:{task_type} epoch_Time:{eta_min}min')

            if grad_norm is not None:
                log_msg += f' grad_norm:{grad_norm:.4f}'

            Logger(log_msg)

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "lr": current_lr,
                    "epoch_Time": eta_min,
                    "task": task_type,
                    "grad_norm": grad_norm if grad_norm else 0,
                    "step": step + epoch * iters
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(epoch, step, current_loss, experiment_dir)

        if task_type == 'referring':
            del X, Y, loss_mask, pixel_values, res, loss, bbox_labels
        else:
            del X, Y, loss_mask, pixel_values, res, loss

    avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
    return avg_epoch_loss


def save_checkpoint(epoch, step, loss, experiment_dir):
    model.eval()

    checkpoint_name = f"checkpoint_epoch{epoch + 1}_step{step + 1}_loss{loss:.4f}"

    moe_suffix = '_moe' if vlm_config.use_moe else ''
    ckp_path = os.path.join(experiment_dir, 'checkpoints', f'{checkpoint_name}.pth')

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    clean_state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith('vision_encoder.')
    }
    clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
    torch.save(clean_state_dict, ckp_path)

    full_checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': vlm_config.__dict__,
        'training_args': vars(args)
    }

    full_ckp_path = os.path.join(experiment_dir, 'checkpoints', f'{checkpoint_name}_full.pt')
    torch.save(full_checkpoint, full_ckp_path)

    vlm_checkpoint(vlm_config, weight=checkpoint_name, model=model, optimizer=optimizer,
                   epoch=epoch, step=step, wandb=wandb, save_dir=experiment_dir, scaler=scaler)

    Logger(f"检查点已保存: {ckp_path}")
    model.train()
    del state_dict, clean_state_dict


def evaluate_model(model, dataloader, device, task_type='sft'):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if task_type == 'referring':
                X, Y, loss_mask, pixel_values, _ = batch
            else:
                X, Y, loss_mask, pixel_values = batch

            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            pixel_values = pixel_values.to(device)

            res = model(X, pixel_values=pixel_values)

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    perplexity = np.exp(min(avg_loss, 10))

    model.train()
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'samples': total_samples
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Training")

    parser.add_argument("--save_dir", type=str, default="../experiments", help="实验保存根目录")
    parser.add_argument('--save_weight', default='vlm_trained', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1536, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")

    parser.add_argument("--data_path", type=str, default="../dataset/sft_data.parquet", help="训练数据路径")
    parser.add_argument("--dataset_type", type=str, default="sft", choices=["sft", "refcoco", "refcocog", "refcoco+"],
                        help="数据集类型")
    parser.add_argument("--image_dir", type=str, default="../refcoco/images", help="图像目录")
    parser.add_argument("--ann_file", type=str, default="../refcoco/instances.json", help="标注文件")
    parser.add_argument("--ref_file", type=str, default="../refcoco/refs(unc).p", help="指代表达文件")

    parser.add_argument("--eval_interval", type=int, default=500, help="评估间隔")
    parser.add_argument("--eval_samples", type=int, default=1000, help="评估样本数")

    parser.add_argument('--from_weight', default='sft_vlm', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Training", help="wandb项目名")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 创建实验文件夹 ==========
    config_dict = {
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'max_seq_len': args.max_seq_len,
        'use_moe': args.use_moe,
        'dataset_type': args.dataset_type,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }

    experiment_dir = create_experiment_folder(args.save_dir, config_dict)
    Logger(f"实验文件夹: {experiment_dir}")

    # 保存训练配置
    save_config(args, experiment_dir)

    # 初始化指标跟踪器
    metrics = TrainingMetrics()
    metrics_path = os.path.join(experiment_dir, 'metrics', 'training_metrics.json')

    # ========== 3. 配置模型参数、检查ckp ==========
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )

    # 检查点恢复
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints')

    # ========== 4. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 5. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import wandb as wb

            wandb_id = ckp_data.get('wandb_id') if ckp_data else None
            resume = 'must' if wandb_id else None
            wandb_run_name = f"MiniMind-V-{args.dataset_type}-E{args.epochs}-BS{args.batch_size}"
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                id=wandb_id,
                resume=resume,
                config=vars(args)
            )
            wandb = wb
            wandb.save(os.path.join(experiment_dir, 'configs', 'training_config.json'))
        except ImportError:
            Logger("警告: wandb未安装，将不会记录训练指标")

    # ========== 6. 加载模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,  # 使用修改后的默认值 sft_vlm_512
        device=args.device
    )

    # 根据数据集类型选择数据集
    if args.dataset_type == "sft":
        dataset = VLMDataset(
            args.data_path,
            tokenizer,
            preprocess=preprocess,
            image_special_token=vlm_config.image_special_token,
            max_length=vlm_config.max_seq_len
        )
        task_type = 'sft'
    else:  # refcoco系列数据集
        dataset = RefCOCODataset(
            image_dir=args.image_dir,
            ann_file=args.ann_file,
            ref_file=args.ref_file,
            dataset_type=args.dataset_type,
            tokenizer=tokenizer,
            processor=preprocess,
            max_length=vlm_config.max_seq_len,
            image_special_token=vlm_config.image_special_token
        )
        task_type = 'referring'

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    Logger(f"数据集大小: {total_size}")
    Logger(f"训练集大小: {len(train_ds)} (70%)")
    Logger(f"验证集大小: {len(val_ds)} (30%)")
    Logger(f"任务类型: {task_type}")

    # 训练集数据加载器
    train_sampler = None
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_ds)

    # 验证集数据加载器
    val_sampler = None
    if dist.is_initialized():
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    # 修复Windows多进程问题：在Windows上强制使用单进程
    if sys.platform == "win32":
        num_workers = 0
        Logger("Windows平台检测到，禁用多进程数据加载")
    else:
        num_workers = min(args.num_workers, 4)  # 验证集使用较少的worker

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False if val_sampler else False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False  # Windows上禁用持久化工作进程
    )

    # 优化器和梯度缩放
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 7. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

        # 加载之前的指标
        if os.path.exists(metrics_path):
            metrics.load(metrics_path)
            metrics.current_epoch = start_epoch

        Logger(f"从检查点恢复: epoch={start_epoch}, step={start_step}")

    # ========== 8. DDP包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练 ==========
    Logger(f"\n{'=' * 60}")
    Logger(f"开始训练 - 实验目录: {experiment_dir}")
    Logger(f"将完整训练 {args.epochs} 个epoch")
    Logger(f"{'=' * 60}\n")

    best_loss = float('inf')
    best_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        metrics.current_epoch = epoch

        if train_sampler:
            train_sampler.set_epoch(epoch)

        # Windows平台下多进程问题修复
        if sys.platform == "win32":  # Windows上设置为0避免多进程问题
            train_num_workers = 0
        else:  # 非Windows平台使用多进程
            train_num_workers = args.num_workers

        # 处理断点续训
        epoch_start_time = time.time()
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)),
                args.batch_size,
                start_step + 1
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=False  # Windows上禁用持久化工作进程
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            avg_loss = train_epoch(epoch, train_loader, len(train_loader) + start_step + 1, start_step, wandb,
                                   task_type, metrics)
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=False  # Windows上禁用持久化工作进程
            )
            avg_loss = train_epoch(epoch, train_loader, len(train_loader), 0, wandb, task_type, metrics)

        epoch_time = time.time() - epoch_start_time

        # 每个epoch结束后进行完整验证（修改：只在epoch结束后验证）
        val_metrics = evaluate_model(model, val_loader, args.device, task_type)
        Logger(f"\nEpoch {epoch + 1} 验证结果:")
        Logger(f"  验证损失: {val_metrics['loss']:.6f}")
        Logger(f"  验证困惑度: {val_metrics['perplexity']:.2f}")
        Logger(f"  验证样本数: {val_metrics['samples']}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            Logger(f"  新的最佳验证损失: {best_val_loss:.6f}")

            # 保存最佳验证模型
            best_val_model_path = os.path.join(experiment_dir, 'checkpoints', 'best_val_model.pth')
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, best_val_model_path)
            Logger(f"  最佳验证模型已保存: {best_val_model_path}")

        # 打印epoch摘要
        summary = metrics.get_summary()
        Logger(f"\n{'=' * 50}")
        Logger(f"Epoch {epoch + 1}/{args.epochs} 完成")
        Logger(f"训练平均损失: {avg_loss:.6f}")
        Logger(f"验证损失: {val_metrics['loss']:.6f}")
        Logger(f"最佳损失: {summary['best_loss']:.6f}")
        Logger(f"平均困惑度: {summary['avg_perplexity']:.2f}")
        Logger(f"当前学习率: {summary['current_lr']:.8f}")
        Logger(f"Epoch用时: {epoch_time / 60:.2f}分钟")
        Logger(f"总用时: {summary['total_time_hours']:.2f}小时")
        Logger(f"{'=' * 50}\n")

        # 检查是否为最佳训练损失模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

            # 保存最佳模型
            best_model_path = os.path.join(experiment_dir, 'checkpoints', 'best_train_model_512.pth')
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, best_model_path)
            Logger(f"新的最佳训练模型已保存 (epoch {best_epoch}): {best_model_path} (损失: {best_loss:.6f})")

        # 定期保存指标
        if is_main_process():
            metrics.save(metrics_path)

            # 保存训练摘要
            summary_path = os.path.join(experiment_dir, 'logs', f'epoch_{epoch + 1}_summary.json')
            epoch_summary = {
                **summary,
                'val_loss': val_metrics['loss'],
                'val_perplexity': val_metrics['perplexity'],
                'val_samples': val_metrics['samples'],
                'epoch': epoch + 1
            }
            with open(summary_path, 'w') as f:
                json.dump(epoch_summary, f, indent=2)

    # ========== 10. 训练完成 ==========
    if is_main_process():
        # 保存最终模型
        final_model_path = os.path.join(experiment_dir, 'checkpoints', 'final_model.pth')
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, final_model_path)

        # 生成最终报告
        final_summary = {
            'experiment_dir': experiment_dir,
            'total_epochs': args.epochs,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_val_loss': best_val_loss,
            'final_loss': avg_loss if 'avg_loss' in locals() else 0,
            'training_completed': True,
            'completion_time': datetime.now().isoformat(),
            'total_training_hours': time.time() - metrics.start_time,
            'training_metrics': summary
        }

        final_report_path = os.path.join(experiment_dir, 'logs', 'final_report.json')
        with open(final_report_path, 'w') as f:
            json.dump(final_summary, f, indent=2)

        Logger(f"\n{'=' * 60}")
        Logger(f"训练完成!")
        Logger(f"实验目录: {experiment_dir}")
        Logger(
            f"最佳训练模型 (epoch {best_epoch}): {os.path.join(experiment_dir, 'checkpoints', 'best_train_model_512.pth')}")
        Logger(f"最佳训练损失: {best_loss:.6f}")
        Logger(f"最佳验证模型: {os.path.join(experiment_dir, 'checkpoints', 'best_val_model.pth')}")
        Logger(f"最佳验证损失: {best_val_loss:.6f}")
        Logger(f"最终模型: {final_model_path}")
        Logger(f"最终报告: {final_report_path}")
        Logger(f"{'=' * 60}")

        if wandb:
            wandb.finish()