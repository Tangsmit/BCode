#!/usr/bin/env python3
# train_referring.py - RefCOCO指代任务训练脚本

import os
import json
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import argparse
import wandb
from datetime import datetime

from model_vlm import MiniMindVLM, VLMConfig


class RefCOCODataset(Dataset):

    def __init__(self,
                 annotations_path: str,
                 images_dir: str,
                 processor,
                 tokenizer,
                 max_length: int = 128,
                 max_images: Optional[int] = None,
                 split: str = 'train'):
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        self.data = self._load_annotations()

        if max_images is not None:
            self.data = self.data[:max_images]

        print(f"Loaded {len(self.data)} samples from {split} split")

    def _load_annotations(self):
        with open(self.annotations_path, 'r') as f:
            annotations = json.load(f)

        data = []

        if 'annotations' in annotations:
            anns = annotations['annotations']
            images = {img['id']: img for img in annotations['images']}

            for ann in anns:
                image_id = ann['image_id']
                bbox = ann['bbox']

                if image_id in images:
                    image_info = images[image_id]
                    image_path = os.path.join(self.images_dir, image_info['file_name'])

                    if not os.path.exists(image_path):
                        continue

                    if 'sentences' in ann:
                        for sentence in ann['sentences']:
                            data.append({
                                'image_path': image_path,
                                'image_id': image_id,
                                'bbox': bbox,
                                'text': sentence['sent'],
                                'width': image_info['width'],
                                'height': image_info['height']
                            })
                    elif 'caption' in ann:
                        data.append({
                            'image_path': image_path,
                            'image_id': image_id,
                            'bbox': bbox,
                            'text': ann['caption'],
                            'width': image_info['width'],
                            'height': image_info['height']
                        })
        else:
            for item in annotations:
                if 'image_path' in item:
                    image_path = item['image_path']
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.images_dir, item.get('image_id', '') + '.jpg')

                    if not os.path.exists(image_path):
                        continue

                    data.append({
                        'image_path': image_path,
                        'image_id': item.get('image_id', 0),
                        'bbox': item['bbox'],
                        'text': item['text'],
                        'width': item.get('width', 640),
                        'height': item.get('height', 480)
                    })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item['image_path']).convert('RGB')
            width, height = image.size

            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'][0]

            text = item['text']
            text_encoding = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )

            bbox = torch.tensor(item['bbox'], dtype=torch.float32)

            if len(bbox) == 4:
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[0] + bbox[2]
                y_max = bbox[1] + bbox[3]
                bbox = torch.tensor([x_min, y_min, x_max, y_max])

            bbox = bbox / torch.tensor([width, height, width, height])

            input_ids = text_encoding['input_ids'][0]
            attention_mask = text_encoding['attention_mask'][0]

            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'bbox_labels': bbox,
                'image_id': item['image_id'],
                'text': text,
                'original_width': width,
                'original_height': height
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return {
                'pixel_values': torch.randn(3, 224, 224),
                'input_ids': torch.randint(0, 100, (self.max_length,)),
                'attention_mask': torch.ones(self.max_length),
                'bbox_labels': torch.tensor([0.1, 0.1, 0.5, 0.5]),
                'image_id': 0,
                'text': 'placeholder',
                'original_width': 640,
                'original_height': 480
            }


class ReferringTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self._set_seed(config.seed)

        self.model = self._init_model()
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_learning_rate
        )

        self.scaler = GradScaler() if config.use_amp else None

        self.criterion = {
            'bbox': nn.SmoothL1Loss(),
            'iou': lambda pred, target: 1 - self.model.compute_iou(pred, target).mean()
        }

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_accuracy': []
        }

        if config.use_wandb:
            wandb.init(project=config.wandb_project, name=config.experiment_name)
            wandb.config.update(config)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_model(self):
        model_config = VLMConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            use_referring_head=True,
            has_cls_token=True,
            referring_head_hidden_size=self.config.referring_head_hidden_size
        )

        model = MiniMindVLM(
            params=model_config,
            vision_model_path=self.config.vision_model_path,
            referring_head_hidden_size=self.config.referring_head_hidden_size
        )

        if self.config.pretrained_path and os.path.exists(self.config.pretrained_path):
            print(f"Loading pretrained weights from {self.config.pretrained_path}")
            checkpoint = torch.load(self.config.pretrained_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return model

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_bbox_loss = 0
        total_iou_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            for key in ['pixel_values', 'input_ids', 'attention_mask', 'bbox_labels']:
                batch[key] = batch[key].to(self.device)

            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    bbox_labels=batch['bbox_labels'],
                    task_type='referring'
                )

                loss = outputs.loss
                total_loss += loss.item()

                bbox_loss = self.criterion['bbox'](outputs.bbox_pred, batch['bbox_labels'])
                iou_loss = self.criterion['iou'](outputs.bbox_pred, batch['bbox_labels'])
                total_bbox_loss += bbox_loss.item()
                total_iou_loss += iou_loss.item()

            self.optimizer.zero_grad()

            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bbox': f'{bbox_loss.item():.4f}',
                'iou': f'{iou_loss.item():.4f}'
            })

            if self.config.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_bbox_loss': bbox_loss.item(),
                    'train_iou_loss': iou_loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })

        self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_bbox_loss = total_bbox_loss / len(train_loader)
        avg_iou_loss = total_iou_loss / len(train_loader)

        self.metrics['train_loss'].append(avg_loss)

        return {
            'loss': avg_loss,
            'bbox_loss': avg_bbox_loss,
            'iou_loss': avg_iou_loss
        }

    def evaluate(self, val_loader, epoch=None):
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                for key in ['pixel_values', 'input_ids', 'attention_mask', 'bbox_labels']:
                    batch[key] = batch[key].to(self.device)

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    bbox_labels=batch['bbox_labels'],
                    task_type='referring'
                )

                loss = outputs.loss
                total_loss += loss.item()

                pred_boxes = outputs.bbox_pred
                target_boxes = batch['bbox_labels']

                iou = self.compute_batch_iou(pred_boxes, target_boxes)
                total_iou += iou.sum().item()

                correct = (iou > 0.5).sum().item()
                total_correct += correct
                total_samples += pred_boxes.size(0)

        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / total_samples
        accuracy = total_correct / total_samples

        self.metrics['val_loss'].append(avg_loss)
        self.metrics['val_iou'].append(avg_iou)
        self.metrics['val_accuracy'].append(accuracy)

        if self.config.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'val_iou': avg_iou,
                'val_accuracy': accuracy
            })

        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'accuracy': accuracy
        }

    def compute_batch_iou(self, pred_boxes, target_boxes):
        pred_boxes = pred_boxes.clamp(0, 1)
        target_boxes = target_boxes.clamp(0, 1)

        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-6

        iou = inter_area / union_area
        return iou

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {metrics['accuracy']:.4f}")

    def train(self, train_loader, val_loader):
        best_accuracy = 0

        for epoch in range(self.config.num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 50}")

            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"BBox: {train_metrics['bbox_loss']:.4f}, "
                  f"IoU: {train_metrics['iou_loss']:.4f}")

            val_metrics = self.evaluate(val_loader, epoch)
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")

            if self.config.save_checkpoints:
                is_best = val_metrics['accuracy'] > best_accuracy
                if is_best:
                    best_accuracy = val_metrics['accuracy']

                self.save_checkpoint(epoch, val_metrics, is_best)

        if self.config.save_final_model:
            final_path = os.path.join(self.config.checkpoint_dir, 'final_model.pt')
            torch.save(self.model.state_dict(), final_path)

        if self.config.use_wandb:
            wandb.finish()

        return self.metrics


def create_data_loaders(config):
    from transformers import AutoTokenizer

    processor = CLIPProcessor.from_pretrained(config.vision_model_path)

    class SimpleTokenizer:
        def __init__(self):
            self.vocab_size = config.vocab_size
            self.pad_token_id = 0
            self.eos_token_id = 1

        def __call__(self, text, return_tensors=None, max_length=None, padding=None, truncation=None):
            tokens = [ord(c) % self.vocab_size for c in text[:max_length]]
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

            return {
                'input_ids': torch.tensor([tokens]),
                'attention_mask': torch.tensor([[1 if t != self.pad_token_id else 0 for t in tokens]])
            }

    tokenizer = SimpleTokenizer()

    train_dataset = RefCOCODataset(
        annotations_path=config.train_annotations,
        images_dir=config.images_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_images=config.max_train_images,
        split='train'
    )

    val_dataset = RefCOCODataset(
        annotations_path=config.val_annotations,
        images_dir=config.images_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_images=config.max_val_images,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Referring Expression Comprehension Model")

    parser.add_argument('--train_annotations', type=str, required=True, help='训练标注文件路径')
    parser.add_argument('--val_annotations', type=str, required=True, help='验证标注文件路径')
    parser.add_argument('--images_dir', type=str, required=True, help='图像目录路径')

    parser.add_argument('--vision_model_path', type=str, default='./model/vision_model/clip-vit-base-patch16',
                        help='视觉模型路径')
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层大小')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='隐藏层层数')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--referring_head_hidden_size', type=int, default=256, help='指代头隐藏层大小')
    parser.add_argument('--vocab_size', type=int, default=32000, help='词汇表大小')

    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')

    parser.add_argument('--max_length', type=int, default=128, help='最大文本长度')
    parser.add_argument('--max_train_images', type=int, default=None, help='最大训练图像数')
    parser.add_argument('--max_val_images', type=int, default=None, help='最大验证图像数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作线程数')

    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--pretrained_path', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--save_checkpoints', action='store_true', help='保存检查点')
    parser.add_argument('--save_final_model', action='store_true', help='保存最终模型')

    parser.add_argument('--use_wandb', action='store_true', help='使用wandb')
    parser.add_argument('--wandb_project', type=str, default='referring-vlm', help='wandb项目名')
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称')
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')

    args = parser.parse_args()

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f'referring_{timestamp}'

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(args)

    print("Creating trainer...")
    trainer = ReferringTrainer(args)

    print("Starting training...")
    metrics = trainer.train(train_loader, val_loader)

    metrics_path = os.path.join(args.checkpoint_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining completed! Results saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()