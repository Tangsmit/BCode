import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw
import time
from model.model_vlm_RCoder import VCoderVLMSystem


def calculate_metrics(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # IoU
    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-7

    iou = inter_area / union_area

    # GIoU
    cw_x1, cw_y1 = min(b1_x1, b2_x1), min(b1_y1, b2_y1)
    cw_x2, cw_y2 = max(b1_x2, b2_x2), max(b1_y2, b2_y2)
    c_area = (cw_x2 - cw_x1) * (cw_y2 - cw_y1) + 1e-7
    giou = iou - (c_area - union_area) / c_area

    # DIoU
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    rho_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    c_diag_sq = (cw_x2 - cw_x1) ** 2 + (cw_y2 - cw_y1) ** 2 + 1e-7
    diou = iou - rho_sq / c_diag_sq

    return {"iou": iou, "giou": giou, "diou": diou}


class VLMRefCOCOEvaluator:
    def __init__(self, system, dataset_root, dataset_name='refcoco'):
        self.system = system
        self.image_dir = os.path.join(dataset_root, 'images', 'train2014')

        print(f"正在载入数据集: {dataset_name}...")
        with open(os.path.join(dataset_root, dataset_name, 'refs(google).p'), 'rb') as f:
            self.refs = pickle.load(f)
        with open(os.path.join(dataset_root, dataset_name, 'instances.json'), 'r') as f:
            instances = json.load(f)
            self.ann_id_to_bbox = {ann['id']: ann['bbox'] for ann in instances['annotations']}

        timestamp = datetime.now().strftime('%m%d_%H%M')
        self.save_dir = f"./RCoder/eval_output_{timestamp}"
        self.vis_dir = os.path.join(self.save_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        self.detailed_results = []

    def draw_and_save(self, img, gt_box, pred_box, metrics, prompt, img_id):
        draw = ImageDraw.Draw(img)

        draw.rectangle(gt_box, outline="lime", width=3)
        draw.rectangle(pred_box, outline="red", width=3)

        text = f"Prompt: {prompt}\nIoU: {metrics['iou']:.3f}"
        draw.text((10, 10), text, fill="yellow")

        save_path = os.path.join(self.vis_dir, f"{img_id}_vis.jpg")
        img.save(save_path)

    def run_eval(self, limit=20):
        results_stats = {"iou": [], "giou": [], "diou": []}
        correct_at_05 = 0
        total_time = 0

        val_samples = [r for r in self.refs if r['split'] == 'val'][:limit]

        for ref in tqdm(val_samples, desc="Evaluating"):
            img_id_num = ref['image_id']
            img_id_str = str(img_id_num).zfill(12)
            img_path = os.path.join(self.image_dir, f"COCO_train2014_{img_id_str}.jpg")

            if not os.path.exists(img_path):
                continue

            gt_wh = self.ann_id_to_bbox[ref['ann_id']]
            gt_box = [
                float(gt_wh[0]),
                float(gt_wh[1]),
                float(gt_wh[0] + gt_wh[2]),
                float(gt_wh[1] + gt_wh[3])
            ]

            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            prompt = ref['sentences'][0]['sent']

            start_time = time.time()
            output_text, _, pred_box = self.system.process(img, prompt, mode="all")
            end_time = time.time()
            total_time += (end_time - start_time)

            m = calculate_metrics(gt_box, pred_box)
            for k in results_stats:
                results_stats[k].append(m[k])

            if m['iou'] >= 0.5:
                correct_at_05 += 1

            record = {
                "img_id": img_id_str,
                "img_size": [w, h],
                "prompt": prompt,
                "model_raw_text": output_text,
                "gt_box_x1y1x2y2": [round(x, 2) for x in gt_box],
                "pred_box_x1y1x2y2": [round(x, 2) for x in pred_box],
                "inference_time": end_time - start_time,
                "metrics": {k: round(v, 4) for k, v in m.items()}
            }
            self.detailed_results.append(record)

            self.draw_and_save(img.copy(), gt_box, pred_box, m, prompt, img_id_str)

        self._save_detailed_log()

        print(f"\n" + "=" * 40)
        print(f"📊 评测汇总 (N={len(results_stats['iou'])})")
        print(f"Mean IoU:  {np.mean(results_stats['iou']):.4f}")
        print(f"Mean GIoU: {np.mean(results_stats['giou']):.4f}")
        print(f"Mean DIoU: {np.mean(results_stats['diou']):.4f}")
        print(f"Acc@0.5:   {(correct_at_05 / len(results_stats['iou'])) * 100:.2f}%")
        print(f"平均耗时:  {total_time / len(results_stats['iou']):.2f} 秒/图")
        print(f"结果已保存至: {self.save_dir}")
        print("=" * 40)

    def _save_detailed_log(self):
        log_path = os.path.join(self.save_dir, "detailed_results.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    CONFIG = {
        "vlm_path": "model",  
        "depth_path": os.path.join(BASE_MODEL_DIR, "Depth/depth-anything-small-hf"),
        "sam_path": os.path.join(BASE_DIR, "model", "facebook/sam3/sam3.pt"),
        "dataset_root": os.path.join(BASE_DIR, "dataset")
    }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> 初始化系统...")

    vlm_system = VCoderVLMSystem(
        vlm_path=CONFIG["vlm_path"],
        yolo_path=CONFIG["depth_path"],
        sam_path=CONFIG["sam_path"],
        device=DEVICE
    )

    evaluator = VLMRefCOCOEvaluator(
        vlm_system,
        CONFIG["dataset_root"]
    )

    evaluator.run_eval(limit=2)