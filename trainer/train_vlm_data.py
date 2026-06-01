# trainer/train_vlm_refcoco.py
import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from PIL import Image
import glob

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from pathlib import Path

from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from dataset.refcoco_dataset import RefCOCODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, \
    vlm_checkpoint, SkipBatchSampler

warnings.filterwarnings('ignore')


class TrainingMetrics:
    """训练指标跟踪器"""

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
        """更新指标"""
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


class CustomRefCOCODataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, tokenizer, processor, max_length=1536,
                 image_special_token="<image>", dataset_type="custom"):

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_special_token = image_special_token
        self.dataset_type = dataset_type

        self.samples = self._load_data()

        Logger(f"加载了 {len(self.samples)} 个样本")
        if len(self.samples) == 0:
            Logger(f"警告: 没有加载到任何样本，请检查数据目录: {data_dir}")
            if self.data_dir.exists():
                Logger(f"目录内容: {list(self.data_dir.iterdir())}")
            else:
                Logger(f"目录不存在: {data_dir}")

    def _load_data(self):
        samples = []

        json_files = list(self.data_dir.glob("*.json"))
        Logger(f"找到 {len(json_files)} 个JSON文件")

        for json_path in json_files:
            try:
                Logger(f"处理JSON文件: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_name = None

                if 'shapes' in data and len(data['shapes']) > 0:
                    if 'image_name' in data['shapes'][0]:
                        image_name = data['shapes'][0]['image_name']
                        Logger(f"从shapes中获取图片名: {image_name}")

                if not image_name:
                    image_name = json_path.stem
                    possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                    found_image = False

                    for ext in possible_extensions:
                        possible_image_path = self.data_dir / f"{image_name}{ext}"
                        if possible_image_path.exists():
                            image_name = f"{image_name}{ext}"
                            found_image = True
                            Logger(f"找到对应图片: {image_name}")
                            break

                    if not found_image:
                        Logger(f"警告: 未找到 {json_path.stem} 对应的图片文件")
                        continue

                image_path = self.data_dir / image_name
                if not image_path.exists():
                    Logger(f"警告: 图片文件不存在: {image_path}")
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        alt_path = self.data_dir / f"{json_path.stem}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            Logger(f"找到替代图片: {alt_path}")
                            break
                    else:
                        Logger(f"错误: 无法找到 {json_path.stem} 的任何图片文件")
                        continue

                texts = data.get('text', [])
                if not texts:
                    Logger(f"警告: JSON文件 {json_path} 中没有文本描述")
                    continue

                shapes = data.get('shapes', [])
                if not shapes:
                    Logger(f"警告: JSON文件 {json_path} 中没有边界框信息")
                    continue

                for text in texts:
                    for shape in shapes:
                        bbox = shape.get('bbox', [])
                        if len(bbox) == 4:
                            if bbox[2] > 0 and bbox[3] > 0:  
                                samples.append({
                                    'image_path': str(image_path),
                                    'text': text,
                                    'bbox': bbox,
                                    'image_name': image_path.name,
                                    'json_file': json_path.name
                                })
                            else:
                                Logger(f"警告: 无效的bbox: {bbox}")
                        else:
                            Logger(f"警告: bbox长度不为4: {bbox}")

            except json.JSONDecodeError as e:
                Logger(f"JSON解析错误 {json_path}: {e}")
            except Exception as e:
                Logger(f"加载标注文件 {json_path} 时出错: {e}")
                import traceback
                Logger(f"错误详情: {traceback.format_exc()}")

        Logger(f"总共加载了 {len(samples)} 个样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image_path = sample['image_path']
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            Logger(f"加载图像失败: {sample['image_path']}, 错误: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        try:
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"]

            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1) 
            elif pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0).unsqueeze(1)  

            pixel_values = pixel_values.squeeze(0) 

        except Exception as e:
            Logger(f"图像预处理失败: {e}")
            pixel_values = torch.zeros((1, 3, 224, 224))

        # 文本编码
        text_with_image_token = f"{self.image_special_token} {sample['text']}"
        text_ids = self.tokenizer(
            text_with_image_token,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )["input_ids"].squeeze(0)

        loss_mask = torch.zeros_like(text_ids, dtype=torch.float32)

        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_special_token)
        if image_token_id is None:
            Logger(f"警告: 无法找到图像token ID: {self.image_special_token}")
            image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            if image_token_id is None:
                Logger(f"警告: 仍然无法找到图像token ID")
                image_token_id = text_ids[0]

        image_token_positions = torch.where(text_ids == image_token_id)[0]

        if len(image_token_positions) > 0:
            start_idx = image_token_positions[0] + 1
            loss_mask[start_idx:] = 1.0

        bbox = sample['bbox']
        try:
            bbox_tensor = torch.tensor([
                bbox[0] / image.width,  
                bbox[1] / image.height,  
                (bbox[0] + bbox[2]) / image.width,  
                (bbox[1] + bbox[3]) / image.height  
            ], dtype=torch.float32)
        except Exception as e:
            Logger(f"bbox处理失败: {e}, bbox={bbox}, image size={image.size}")
            bbox_tensor = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

        labels = torch.clone(text_ids)
        labels[loss_mask == 0] = -100  

        return text_ids, labels, loss_mask, pixel_values, bbox_tensor


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
    bbox_loss_fct = nn.MSELoss()  
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
                if pixel_values.dim() == 4:
                    pixel_values = pixel_values.unsqueeze(1)  
                    Logger(f"修复后pixel_values形状: {pixel_values.shape}")

                res = model(X, pixel_values=pixel_values, task_type='referring')

                text_loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                text_loss = (text_loss * loss_mask).sum() / loss_mask.sum()

                if bbox_labels is not None and hasattr(res, 'bbox_pred'):
                    bbox_pred = res.bbox_pred
                    if bbox_pred.shape[0] == bbox_labels.shape[0]:
                        bbox_loss = bbox_loss_fct(bbox_pred, bbox_labels)
                        loss = text_loss + args.bbox_loss_weight * bbox_loss
                        if hasattr(res, 'aux_loss'):
                            loss += res.aux_loss
                    else:
                        Logger(f"警告: bbox_pred形状{bbox_pred.shape}与bbox_labels形状{bbox_labels.shape}不匹配")
                        if hasattr(res, 'aux_loss'):
                            loss = text_loss + res.aux_loss
                        else:
                            loss = text_loss
                else:
                    if hasattr(res, 'aux_loss'):
                        loss = text_loss + res.aux_loss
                    else:
                        loss = text_loss

                loss = loss / args.accumulation_steps
            else:
                res = model(X, pixel_values=pixel_values)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                if hasattr(res, 'aux_loss'):
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

            if task_type == 'referring' and bbox_labels is not None and hasattr(res, 'bbox_pred'):
                if 'bbox_loss' in locals():
                    log_msg += f' bbox_loss:{bbox_loss.item():.6f}'

            Logger(log_msg)

            if wandb:
                log_dict = {
                    "loss": current_loss,
                    "lr": current_lr,
                    "epoch_Time": eta_min,
                    "task": task_type,
                    "grad_norm": grad_norm if grad_norm else 0,
                    "step": step + epoch * iters
                }
                if task_type == 'referring' and bbox_labels is not None and hasattr(res, 'bbox_pred'):
                    if 'bbox_loss' in locals():
                        log_dict["bbox_loss"] = bbox_loss.item()
                wandb.log(log_dict)


        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(epoch, step, current_loss, experiment_dir)


        if task_type == 'referring':
            del X, Y, loss_mask, pixel_values, res, loss
            if 'bbox_labels' in locals():
                del bbox_labels
            if 'bbox_loss' in locals():
                del bbox_loss
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Training")

    parser.add_argument("--save_dir", type=str, default="../experiments", help="实验保存根目录")
    parser.add_argument('--save_weight', default='vlm_trained', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--bbox_loss_weight", type=float, default=0.1, help="边界框损失权重")

    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1536, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")

    parser.add_argument("--data_path", type=str, default="../data/train", help="训练数据路径（包含图片和对应的JSON文件）")
    parser.add_argument("--dataset_type", type=str, default="custom",
                        choices=["sft", "refcoco", "refcocog", "refcoco+", "custom"],
                        help="数据集类型")

    parser.add_argument('--from_weight', default='sft_vlm', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Training", help="wandb项目名")

    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    config_dict = {
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'max_seq_len': args.max_seq_len,
        'use_moe': args.use_moe,
        'dataset_type': args.dataset_type,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'bbox_loss_weight': args.bbox_loss_weight
    }

    experiment_dir = create_experiment_folder(args.save_dir, config_dict)
    Logger(f"实验文件夹: {experiment_dir}")

    save_config(args, experiment_dir)

    metrics = TrainingMetrics()
    metrics_path = os.path.join(experiment_dir, 'metrics', 'training_metrics.json')

    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )

    ckp_data = None
    if args.from_resume == 1:
        ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints')

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import wandb as wb

            wandb_id = ckp_data.get('wandb_id') if ckp_data else None
            resume = 'must' if wandb_id else None
            wandb_run_name = f"MiniMind-V-Custom-RefCOCO-E{args.epochs}-BS{args.batch_size}"
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

    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        device=args.device
    )

    if args.dataset_type == "custom":
        Logger(f"加载自定义数据集，数据路径: {args.data_path}")
        data_path = Path(args.data_path)
        if not data_path.exists():
            Logger(f"错误: 数据路径不存在: {args.data_path}")
            Logger(f"当前工作目录: {os.getcwd()}")
            sys.exit(1)

        dataset = CustomRefCOCODataset(
            data_dir=args.data_path,
            tokenizer=tokenizer,
            processor=preprocess,
            max_length=vlm_config.max_seq_len,
            image_special_token=vlm_config.image_special_token,
            dataset_type="custom"
        )
        task_type = 'referring'
    elif args.dataset_type == "sft":
        dataset = VLMDataset(
            args.data_path,
            tokenizer,
            preprocess=preprocess,
            image_special_token=vlm_config.image_special_token,
            max_length=vlm_config.max_seq_len
        )
        task_type = 'sft'
    else:  
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

    Logger(f"数据集大小: {len(dataset)}")
    Logger(f"任务类型: {task_type}")
    Logger(f"不进行验证集划分，使用全部数据训练")

    if len(dataset) == 0:
        Logger(f"错误: 数据集为空，无法训练")
        Logger(f"请检查以下内容:")
        Logger(f"1. 数据路径是否正确: {args.data_path}")
        Logger(f"2. 数据目录是否包含图片和JSON文件")
        Logger(f"3. JSON文件格式是否正确")
        Logger(f"4. 文件命名是否符合规范")
        sys.exit(1)

    train_sampler = None
    if dist.is_initialized():
        train_sampler = DistributedSampler(dataset)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

        if os.path.exists(metrics_path):
            metrics.load(metrics_path)
            metrics.current_epoch = start_epoch

        Logger(f"从检查点恢复: epoch={start_epoch}, step={start_step}")

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    Logger(f"\n{'=' * 60}")
    Logger(f"开始训练 - 实验目录: {experiment_dir}")
    Logger(f"将完整训练 {args.epochs} 个epoch")
    Logger(f"使用全部 {len(dataset)} 个样本进行训练")
    Logger(f"{'=' * 60}\n")

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        metrics.current_epoch = epoch

        if train_sampler:
            train_sampler.set_epoch(epoch)

        if sys.platform == "win32":  
            train_num_workers = 0
        else:  
            train_num_workers = args.num_workers

        epoch_start_time = time.time()
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(dataset)),
                args.batch_size,
                start_step + 1
            )
            train_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=False  
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            avg_loss = train_epoch(epoch, train_loader, len(train_loader) + start_step + 1, start_step, wandb,
                                   task_type, metrics)
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=False  
            )
            avg_loss = train_epoch(epoch, train_loader, len(train_loader), 0, wandb, task_type, metrics)

        epoch_time = time.time() - epoch_start_time

        summary = metrics.get_summary()
        Logger(f"\n{'=' * 50}")
        Logger(f"Epoch {epoch + 1}/{args.epochs} 完成")
        Logger(f"训练平均损失: {avg_loss:.6f}")
        Logger(f"最佳损失: {summary['best_loss']:.6f}")
        Logger(f"平均困惑度: {summary['avg_perplexity']:.2f}")
        Logger(f"当前学习率: {summary['current_lr']:.8f}")
        Logger(f"Epoch用时: {epoch_time / 60:.2f}分钟")
        Logger(f"总用时: {summary['total_time_hours']:.2f}小时")
        Logger(f"{'=' * 50}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

            best_model_path = os.path.join(experiment_dir, 'checkpoints',
                                           f'best_model_epoch{best_epoch}_loss{best_loss:.4f}.pth')
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, best_model_path)
            Logger(f"新的最佳训练模型已保存 (epoch {best_epoch}): {best_model_path} (损失: {best_loss:.6f})")

        if is_main_process():
            metrics.save(metrics_path)

            summary_path = os.path.join(experiment_dir, 'logs', f'epoch_{epoch + 1}_summary.json')
            epoch_summary = {
                **summary,
                'epoch': epoch + 1,
                'avg_epoch_loss': avg_loss,
                'best_loss_so_far': best_loss,
                'best_epoch_so_far': best_epoch
            }
            with open(summary_path, 'w') as f:
                json.dump(epoch_summary, f, indent=2)

        if is_main_process():
            checkpoint_name = f"epoch_{epoch + 1}_final"
            epoch_checkpoint_path = os.path.join(experiment_dir, 'checkpoints', f'{checkpoint_name}.pth')
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, epoch_checkpoint_path)
            Logger(f"Epoch {epoch + 1} 检查点已保存: {epoch_checkpoint_path}")

    if is_main_process():
        final_model_path = os.path.join(experiment_dir, 'checkpoints', 'final_model.pth')
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, final_model_path)

        final_summary = {
            'experiment_dir': experiment_dir,
            'total_epochs': args.epochs,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'final_loss': avg_loss if 'avg_loss' in locals() else 0,
            'training_completed': True,
            'completion_time': datetime.now().isoformat(),
            'total_training_hours': (time.time() - metrics.start_time) / 3600,
            'total_samples': len(dataset),
            'training_metrics': summary
        }

        final_report_path = os.path.join(experiment_dir, 'logs', 'final_report.json')
        with open(final_report_path, 'w') as f:
            json.dump(final_summary, f, indent=2)

        Logger(f"\n{'=' * 60}")
        Logger(f"训练完成!")
        Logger(f"实验目录: {experiment_dir}")
        Logger(
            f"最佳训练模型 (epoch {best_epoch}): {os.path.join(experiment_dir, 'checkpoints', f'best_model_epoch{best_epoch}_loss{best_loss:.4f}.pth')}")
        Logger(f"最佳训练损失: {best_loss:.6f}")
        Logger(f"最终模型: {final_model_path}")
        Logger(f"最终报告: {final_report_path}")
        Logger(f"{'=' * 60}")

        if wandb:
            wandb.finish()