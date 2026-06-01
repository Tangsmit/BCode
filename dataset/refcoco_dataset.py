# dataset/refcoco_dataset.py
import torch
from torch.utils.data import Dataset
import os
import json
import pickle
from PIL import Image
import random
import numpy as np


class RefCOCODataset(Dataset):

    def __init__(self, image_dir, ann_file, ref_file, dataset_type='refcoco',
                 tokenizer=None, processor=None, max_length=1536,
                 image_special_token='@' * 196):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_special_token = image_special_token

        # 添加特殊token到tokenizer
        if tokenizer:
            if image_special_token not in tokenizer.get_vocab():
                special_tokens_dict = {'additional_special_tokens': [image_special_token]}
                tokenizer.add_special_tokens(special_tokens_dict)

        with open(ann_file, 'r') as f:
            self.coco_anns = json.load(f)

        self.image_id_to_anns = {}
        for ann in self.coco_anns['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_anns:
                self.image_id_to_anns[image_id] = []
            self.image_id_to_anns[image_id].append(ann)

        self.image_id_to_info = {img['id']: img for img in self.coco_anns['images']}

        with open(ref_file, 'rb') as f:
            self.ref_data = pickle.load(f)

        self.samples = []
        for ref in self.ref_data:
            if dataset_type == 'refcoco' and 'refcoco' not in str(ref_file):
                continue
            elif dataset_type == 'refcocog' and 'refcocog' not in str(ref_file):
                continue
            elif dataset_type == 'refcoco+' and 'refcoco+' not in str(ref_file):
                continue

            if ref['split'] != 'train':
                continue

            image_info = self.image_id_to_info.get(ref['image_id'])
            if not image_info:
                continue

            image_path = os.path.join(image_dir, image_info['file_name'])
            if not os.path.exists(image_path):
                continue

            anns = [ann for ann in self.image_id_to_anns.get(ref['image_id'], [])
                    if ann['id'] == ref['ann_id']]
            if not anns:
                continue

            self.samples.append({
                'ref': ref,
                'image_path': image_path,
                'image_info': image_info,
                'ann': anns[0]  
            })

        print(f"加载 {len(self.samples)} 个{dataset_type}样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref = sample['ref']
        image_path = sample['image_path']
        ann = sample['ann']

        image = Image.open(image_path).convert('RGB')

        sentences = ref['sentences']
        selected_sentence = random.choice(sentences)
        text = selected_sentence['sent']

        bbox = ann['bbox']  # [x, y, width, height]
        bbox_norm = [
            (bbox[0] + bbox[2] / 2) / sample['image_info']['width'],  # x_center
            (bbox[1] + bbox[3] / 2) / sample['image_info']['height'],  # y_center
            bbox[2] / sample['image_info']['width'],  # width
            bbox[3] / sample['image_info']['height']  # height
        ]

        input_text = f"描述图像中的对象: {text} {self.image_special_token}"

        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        input_ids = inputs['input_ids'].squeeze() 

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        if self.processor:
            pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)  # [3, H, W]
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)

        bbox_labels = torch.tensor(bbox_norm, dtype=torch.float32)

        return input_ids, labels, loss_mask, pixel_values, bbox_labels


def refcoco_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    loss_mask = torch.stack([item[2] for item in batch])
    pixel_values = torch.stack([item[3] for item in batch])
    bbox_labels = torch.stack([item[4] for item in batch])

    return input_ids, labels, loss_mask, pixel_values, bbox_labels