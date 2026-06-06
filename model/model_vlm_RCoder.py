import torch
import re
import numpy as np
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor
from ultralytics import YOLOWorld, SAM

from .model_vlm import MiniMindVLM, VLMConfig


class VCoderVLMSystem:
    def __init__(
        self,
        vlm_path,
        yolo_path,
        sam_path,
        depth_path=None,
        device="cuda",
        weight_path="./out/best_val.pth",
        hidden_size=768,
        num_hidden_layers=16,
        use_moe=False
    ):
        self.device = torch.device(device)

        print("🚀 初始化 VCoderVLMSystem ...")

        # =========================
        # 1️⃣ tokenizer（仅用于文本）
        # =========================
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./model/vision_model/clip-vit-base-patch16",
            trust_remote_code=True
        )

        # =========================
        # 2️⃣ processor（独立创建，避免 None 问题）
        # =========================
        self.processor = AutoProcessor.from_pretrained(
            "./model/vision_model/clip-vit-base-patch16"
        )

        # =========================
        # 3️⃣ 初始化 VLM
        # =========================
        self.model = MiniMindVLM(
            VLMConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                use_moe=bool(use_moe)
            ),
            vision_model_path="./model/vision_model/clip-vit-base-patch16"
        )

        # ⚠️ 不再依赖 model.processor
        self.model.processor = self.processor

        # =========================
        # 4️⃣ 加载权重
        # =========================
        print(f"📦 加载权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location=self.device)

        self.model.load_state_dict(
            {k: v for k, v in state_dict.items() if 'mask' not in k},
            strict=False
        )

        self.model = self.model.to(self.device).eval()

        # =========================
        # 5️⃣ YOLO & SAM
        # =========================
        print(f"🔍 加载 YOLO-World: {yolo_path}")
        self.yolo = YOLOWorld(yolo_path)

        # 🔥 强制统一 device（关键）
        self.yolo.to(device)

        # 🔥 关键补丁：让 CLIP 也到 GPU
        if hasattr(self.yolo.model, "model") and hasattr(self.yolo.model.model, "to"):
            self.yolo.model.model.to(device)

        print("🧠 加载 SAM...")
        self.sam = SAM(sam_path)

        print("✅ 初始化完成\n")

    # =========================
    # IoU
    # =========================
    def _calculate_iou(self, box1, box2):
        i_x1, i_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        i_x2, i_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        inter_area = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (area1 + area2 - inter_area + 1e-7)

    def simplify_prompt(self, prompt):
        words = re.findall(r'\w+', prompt.lower())
        return words[-1] if words else "object"

    # =========================
    # 主流程
    # =========================
    @torch.no_grad()
    def process(self, image, prompt, mode="all"):
        w, h = image.size

        # =========================
        # 1️⃣ 图像处理（稳定版）
        # =========================
        inputs_img = self.processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = {"pixel_values": inputs_img["pixel_values"].to(self.device).unsqueeze(1)}

        # =========================
        # 2️⃣ 文本（不使用 chat_template，避免报错）
        # =========================
        text = prompt.replace("<image>", "<image>")

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        # =========================
        # 3️⃣ 推理
        # =========================
        gen_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=100,
            do_sample=False
        )

        output_text = self.tokenizer.decode(
            gen_ids[0],
            skip_special_tokens=True
        )

        # =========================
        # 4️⃣ 解析 bbox
        # =========================
        nums = re.findall(r"(\d+)", output_text)

        if len(nums) >= 4:
            raw = [float(n) for n in nums[-4:]]
            pred_box = [
                (raw[1] / 1000) * w,
                (raw[0] / 1000) * h,
                (raw[3] / 1000) * w,
                (raw[2] / 1000) * h
            ]
        else:
            pred_box = [0, 0, w, h]

        # =========================
        # 5️⃣ YOLO 辅助（兜底）
        # =========================
        try:
            target_cls = self.simplify_prompt(prompt)
            self.yolo.set_classes([target_cls])

            y_res = self.yolo.predict(image, conf=0.1, verbose=False)[0]
            yolo_boxes = y_res.boxes.xyxy.cpu().numpy().tolist()

            if yolo_boxes:
                best_box = yolo_boxes[0]
                pred_box = best_box
        except:
            pass

        return output_text, [prompt], pred_box