import os
import json
import cv2
import argparse
import urllib.request
import torch
import torch.nn as nn
import torchvision.models as models
import math

CLASSES = ["person", "car", "dog", "cat", "chair"]

WEIGHTS_URL = "https://huggingface.co/BuiMinhQuan/my_yolo/resolve/main/best.pth"
DEFAULT_WEIGHTS_PATH = "./models/best.pth"

def download_weights(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading weights from {url}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            urllib.request.urlretrieve(url, save_path)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download weights: {e}")
            exit(1)
    else:
        print(f"Weights already exist at {save_path}. Skipping download.")


# Model definition
class DecoupledHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=256, B=2, C=5):
        super(DecoupledHead, self).__init__()
        
        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1)
        )

        # 2. Localization branch
        self.reg_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, B * 5, kernel_size=1)
        )

        # 3. Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, C, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)
        
        reg_out = self.reg_branch(x) 
        cls_out = self.cls_branch(x) 

        out = torch.cat([reg_out, cls_out], dim=1) 

        return torch.sigmoid(out)

class YOLOv1ResNet(nn.Module):
    def __init__(self, S=14, B=2, C=5):
        super(YOLOv1ResNet, self).__init__()
        self.S = S 
        self.B = B 
        self.C = C 

        # 1. Backbone 
        resnet = models.resnet101(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Decoupled Head 
        self.yolo_head = DecoupledHead(in_channels=2048, hidden_channels=256, B=self.B, C=self.C)

    def forward(self, x):
        x = self.backbone(x)
        x = self.yolo_head(x)
        x = x.permute(0, 2, 3, 1) 
        return x

# Utility functions for post-processing
def compute_iou_1d(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6
    return inter_area / union_area



def non_max_suppression(bboxes, iou_threshold=0.4, conf_threshold=0.05):
    bboxes = [box for box in bboxes if box[4] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)
        bboxes = [box for box in bboxes if box[5] != chosen_box[5] or compute_iou_1d(chosen_box[:4], box[:4]) < iou_threshold]
    return bboxes_after_nms


def soft_nms(bboxes, iou_threshold=0.4, conf_threshold=0.05, sigma=0.5):
    bboxes = [box for box in bboxes if box[4] > conf_threshold]
    bboxes_after_nms = []

    while len(bboxes) > 0:
        max_idx = max(range(len(bboxes)), key=lambda i: bboxes[i][4])
        chosen_box = bboxes.pop(max_idx)
        bboxes_after_nms.append(chosen_box)

        for box in bboxes:
            if box[5] != chosen_box[5]:
                continue
                
            iou = compute_iou_1d(chosen_box[:4], box[:4])

            weight = math.exp(-(iou * iou) / sigma)
            
            box[4] = box[4] * weight

        bboxes = [box for box in bboxes if box[4] > conf_threshold]

    return bboxes_after_nms

def decode_yolo_predictions(predictions, S=14, B=2, C=5, image_size=448, conf_thresh=0.05):
    bboxes = []
    cell_size = image_size / S
    for i in range(S):
        for j in range(S):
            class_probs = predictions[i, j, 10:15]
            class_id = torch.argmax(class_probs).item()
            class_score = torch.max(class_probs).item()
            for b in range(B):
                box_idx = b * 5
                confidence = predictions[i, j, box_idx + 4].item() * class_score
                if confidence < conf_thresh: continue
                x_cell = predictions[i, j, box_idx + 0].item()
                y_cell = predictions[i, j, box_idx + 1].item()
                w_norm = predictions[i, j, box_idx + 2].item()
                h_norm = predictions[i, j, box_idx + 3].item()
                
                x_center = (j + x_cell) * cell_size
                y_center = (i + y_cell) * cell_size
                w = w_norm * image_size
                h = h_norm * image_size
                
                xmin = x_center - w / 2
                ymin = y_center - h / 2
                xmax = x_center + w / 2
                ymax = y_center + h / 2
                bboxes.append([xmin, ymin, xmax, ymax, confidence, class_id])
    return bboxes

def predict_with_tta(model, image, device, args):
    orig_h, orig_w = image.shape[:2]
    
    img_resized = cv2.resize(image, (448, 448))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    tensor_orig = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor_orig = tensor_orig.to(device)

    img_flipped = cv2.flip(img_rgb, 1)
    tensor_flip = torch.from_numpy(img_flipped).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor_flip = tensor_flip.to(device)



    with torch.no_grad():
        out_orig = model(tensor_orig)[0].cpu()
        out_flip = model(tensor_flip)[0].cpu()

    boxes_orig = decode_yolo_predictions(out_orig, S=14, B=2, C=5, image_size=448, conf_thresh=args.conf_thresh)
    boxes_flip = decode_yolo_predictions(out_flip, S=14, B=2, C=5, image_size=448, conf_thresh=args.conf_thresh)

    for box in boxes_flip:
        old_xmin = box[0]
        old_xmax = box[2]
        
        box[0] = 448 - old_xmax
        box[2] = 448 - old_xmin


    combined_boxes = boxes_orig + boxes_flip
    
    final_boxes = soft_nms(combined_boxes, iou_threshold=args.iou_thresh, conf_threshold=args.conf_thresh)

    return final_boxes




# Main function for inference
def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script for YOLOv1ResNet model")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing input images')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file for predictions.json')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='Confidence threshold (default: 0.2)')
    parser.add_argument('--iou_thresh', type=float, default=0.4, help='NMS IoU threshold (default: 0.4)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Download weights if not exist
    download_weights(WEIGHTS_URL, DEFAULT_WEIGHTS_PATH)

    # 2. Load model and weights
    model = YOLOv1ResNet(S=14, B=2, C=5)
    model.load_state_dict(torch.load(DEFAULT_WEIGHTS_PATH))
    model.to(device)
    model.eval()

    # 3. Process images and make predictions
    results = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    image_files = [
        f for f in os.listdir(args.image_dir) 
        if f.lower().endswith(valid_extensions)
    ]

    print(f"Found {len(image_files)} images in {args.image_dir}. Processing...")

    for filename in image_files:
        image_path  = os.path.join(args.image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        orig_h, orig_w = image.shape[:2]

        bboxes_nms = predict_with_tta(model, image, device, args)

        formatted_boxes = []
        scale_x = orig_w / 448.0
        scale_y = orig_h / 448.0

        for box in bboxes_nms:
            xmin = max(0, int(box[0] * scale_x))
            ymin = max(0, int(box[1] * scale_y))
            xmax = min(orig_w, int(box[2] * scale_x))
            ymax = min(orig_h, int(box[3] * scale_y))
            conf = round(box[4], 4)
            class_id = CLASSES[box[5]]
            
            formatted_boxes.append({
                "class": class_id,
                "confidence": conf,
                "bbox": [xmin, ymin, xmax, ymax]
            })

        results.append({
            "image_id": filename,
            "boxes": formatted_boxes
        })
    
    # 4. Save results to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()



