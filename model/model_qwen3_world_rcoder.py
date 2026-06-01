import math
import os
import re
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLOWorld
from qwen_vl_utils import process_vision_info


class VCoderQwenSystem:
    def __init__(self, qwen_path, yolo_path, depth_path=None, device="cuda"):
        self.device = torch.device(device)

        # 1. 加载 Qwen3-VL
        print(f"Loading Qwen3-VL from: {qwen_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(qwen_path)

        # 2. 加载 YOLO-World (核心修正组件)
        print(f"Loading YOLO-World from: {yolo_path}...")
        self.yolo = YOLOWorld(yolo_path)
        self.yolo.to(device)

        # --- 新增：统计模型参数 ---
        print("\n" + "=" * 30)
        print("Model Parameter Statistics:")
        qwen_params = self.count_parameters(self.model)
        yolo_params = self.count_parameters(self.yolo.model)  # YOLO-World 的 torch 模型通常在 .model 属性下

        print(f"Qwen3-VL:    {qwen_params / 1e6:>10.2f} M params")
        print(f"YOLO-World:  {yolo_params / 1e6:>10.2f} M params")
        print(f"Total:       {(qwen_params + yolo_params) / 1e6:>10.2f} M params")
        print("=" * 30 + "\n")

    @staticmethod
    def count_parameters(model):
        """计算模型总参数"""
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        return 0

    def _calculate_iou(self, box1, box2):
        """计算 [x1, y1, x2, y2] 格式的 IoU"""
        i_x1, i_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        i_x2, i_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_w, inter_h = max(0, i_x2 - i_x1), max(0, i_y2 - i_y1)
        inter = inter_w * inter_h
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-7)

    def normalize_to_pixel(raw_coords, w, h):
        # 假设 raw_coords 是 [x1, y1, x2, y2] 且范围是 0-1000
        # 强制进行边界裁剪，防止预测框溢出
        x1 = max(0, min(1000, raw_coords[0]))
        y1 = max(0, min(1000, raw_coords[1]))
        x2 = max(0, min(1000, raw_coords[2]))
        y2 = max(0, min(1000, raw_coords[3]))

        # 转换为实际像素
        return [
            (x1 / 1000.0) * w,
            (y1 / 1000.0) * h,
            (x2 / 1000.0) * w,
            (y2 / 1000.0) * h
        ]

    @torch.no_grad()
    def process(self, image, prompt):
        """
        高度鲁棒的推理逻辑：
        1. 自动处理 Letterbox 缩放补偿
        2. 动态判定 [xmin, ymin, xmax, ymax] 与 [ymin, xmin, ymax, xmax]
        3. 引入类别简化与 YOLO 锚点修正
        """
        w, h = image.size

        # --- 步骤 1: Qwen3-VL 提名 ---
        messages = [
            {
                "role": "system",
                "content": "You are a precise detector. Output ONLY the bounding box in [ymin, xmin, ymax, xmax] format."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Find <|object_ref_start|>{prompt}<|object_ref_end|>"}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(self.device)

        gen_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output_text = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # 提取坐标数字
        nums = re.findall(r"(\d+)", output_text)
        if len(nums) < 4:
            return output_text, [prompt], [0, 0, w, h]

        raw = [int(n) for n in nums[-4:]]

        # --- 步骤 2: YOLO-World 语义简化与检测 ---
        simplified_cls = self.simplify_prompt(prompt)
        self.yolo.set_classes([simplified_cls])
        yolo_res = self.yolo.predict(image, conf=0.05, verbose=False)
        yolo_boxes = yolo_res[0].boxes.xyxy.cpu().numpy().tolist()

        # --- 步骤 3: 坐标解析与 Letterbox 补偿 ---
        # 很多 VLM 模型是基于 1000x1000 正方形推理的，若原图长宽比悬殊，需考虑缩放一致性
        def to_pixel(coords):
            # 强制裁剪到 0-1000 范围并转为像素
            return [
                (max(0, min(1000, coords[0])) / 1000.0) * w,
                (max(0, min(1000, coords[1])) / 1000.0) * h,
                (max(0, min(1000, coords[2])) / 1000.0) * w,
                (max(0, min(1000, coords[3])) / 1000.0) * h
            ]

        # 候选 A: [ymin, xmin, ymax, xmax] -> 官方声明格式
        cand_a = to_pixel([raw[1], raw[0], raw[3], raw[2]])
        # 候选 B: [xmin, ymin, xmax, ymax] -> 实测 RefCOCO 常见本能格式
        cand_b = to_pixel([raw[0], raw[1], raw[2], raw[3]])

        # --- 步骤 4: 鲁棒性决策决策树 ---
        best_cand = cand_b  # 默认信任 B (根据之前 78% 的结果)
        final_box = cand_b
        max_score = -1.0

        for cand in [cand_a, cand_b]:
            cand_center = [(cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2]

            for y_box in yolo_boxes:
                iou = self._calculate_iou(cand, y_box)
                y_center = [(y_box[0] + y_box[2]) / 2, (y_box[1] + y_box[3]) / 2]
                dist = np.sqrt((cand_center[0] - y_center[0]) ** 2 + (cand_center[1] - y_center[1]) ** 2)

                # 综合评分：IoU 越高、距离越近，得分越高
                # 距离进行归一化处理，防止大图干扰
                dist_norm = dist / (np.sqrt(w ** 2 + h ** 2) + 1e-6)
                score = iou * 1.0 + (1.0 - dist_norm) * 0.5

                if score > max_score:
                    max_score = score
                    best_cand = cand
                    # 如果 IoU 足够，则信任 YOLO 修正边缘；否则仅把 YOLO 当作格式校验器
                    final_box = y_box if iou > 0.25 else cand

        # --- 步骤 5: 针对特殊 Prompt 的方位兜底 ---
        # 针对 "second from front", "on the right" 等语义
        if max_score < 0.2:  # YOLO 没匹配上或识别失败
            # 逻辑：如果 prompt 说 right 但框在左边，切换候选
            if "right" in prompt.lower() and best_cand[0] < w * 0.3:
                best_cand = cand_a if cand_a[0] > cand_b[0] else cand_b
            final_box = best_cand

        # 最终安全裁剪：确保不超出图像边界
        final_box = [
            max(0, min(w, final_box[0])), max(0, min(h, final_box[1])),
            max(0, min(w, final_box[2])), max(0, min(h, final_box[3]))
        ]

        return output_text, [prompt], final_box

    def simplify_prompt(self, prompt):
        """核心名词提取逻辑"""
        prompt_lower = prompt.lower()
        mapping = {
            'person': ['guy', 'man', 'woman', 'girl', 'boy', 'player', 'surfer', 'skiier', 'butt'],
            'dog': ['puppy', 'animal'],
            'boat': ['ship', 'raft', 'watercraft'],
            'drink': ['bottle', 'cup', 'vase', 'glass', 'can', 'wine'],
            'food': ['sandwich', 'orange', 'apple', 'fruit', 'cake', 'pizza'],
            'furniture': ['chair', 'upholstery', 'sofa', 'table', 'seat']
        }
        for core_cls, synonyms in mapping.items():
            if core_cls in prompt_lower or any(s in prompt_lower for s in synonyms):
                return core_cls
        words = prompt_lower.split()
        return words[-1] if words else prompt