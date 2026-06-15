import os
import json
import cv2
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import math

class YOLOAugmentation:
    def __init__(self, output_size=448):
        self.output_size = output_size 

    def __call__(self, image, boxes, labels):
        image = self.random_photometric(image)
        image, boxes, labels = self.random_geometric(image, boxes, labels)
        image, boxes = self.resize(image, boxes)

        return image, boxes, labels

    def random_photometric(self, image):
        if random.random() < 0.5:
            # Random Brightness & Contrast
            alpha = random.uniform(0.7, 1.3) # Contrast
            beta = random.uniform(-30, 30)   # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        if random.random() < 0.5:
            # Random Hue & Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] += random.uniform(-10, 10) # Hue shift
            hsv[:, :, 1] *= random.uniform(0.7, 1.3) # Saturation shift
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if random.random() < 0.3:
            # Random Blur
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        if random.random() < 0.3:
            # Random Gaussian Noise
            noise = np.random.normal(0, 15, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        return image
    
    
    def random_geometric(self, image, boxes, labels):
        h, w = image.shape[:2]

        # Random horizonal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

            if len(boxes) > 0:
                new_xmin = w - boxes[:, 2] # W - xmax
                new_xmax = w - boxes[:, 0] # W - xmin
                boxes[:, 0] = new_xmin
                boxes[:, 2] = new_xmax
            
        # Random vertical flip
        if random.random() < 0.1:
            image = cv2.flip(image, 0)
            if len(boxes) > 0:
                new_ymin = h - boxes[:, 3]
                new_ymax = h - boxes[:, 1]
                boxes[:, 1] = new_ymin
                boxes[:, 3] = new_ymax

        # Random Scale and Crop
        if random.random() < 0.5 and len(boxes) > 0:
            scale = random.uniform(0.7, 0.9) 
            new_h, new_w = int(h * scale), int(w * scale)

            # Pick a random cropping point
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

            image = image[top: top + new_h, left: left + new_w]

            # Move bbox coordinates
            boxes[:, 0] -= left
            boxes[:, 2] -= left 
            boxes[:, 1] -= top 
            boxes[:, 3] -= top 

            # Remove bbox parts that are out of the cropping region
            boxes[:, 0] = np.clip(boxes[:, 0], 0, new_w)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, new_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, new_h)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, new_h)

            # Remove boxes that are cut too much
            valid_indices = (boxes[:, 2] - boxes[:, 0] > 5) & (boxes[:, 3] - boxes[:, 1] > 5)
            boxes = boxes[valid_indices]
            labels = np.array(labels)[valid_indices].tolist()

        
        return image, boxes, labels 
    

    def resize(self, image, boxes):
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.output_size, self.output_size))

        if len(boxes) > 0:
            scale_x = self.output_size / w
            scale_y = self.output_size / h
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            
        return image, boxes


class YOLODataset(Dataset):
    def __init__(self, json_file, img_dir, augmentor, image_size=448, S=7, B=2, C=5, is_train=False):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.image_size = image_size

        self.augmentor = augmentor

        self.S = S
        self.B = B
        self.C = C

        self.is_train = is_train

        # Mapping:
        # person -> 0
        # car    -> 1
        # ...
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.data["classes"])}

        self.img_to_anns = {img["id"]: [] for img in self.data["images"]}

        for ann in self.data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        self.images_info = self.data["images"]

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        # =====================================================
        # 1. Read images and annotations
        # =====================================================
        img_info = self.images_info[idx]
        file_name = os.path.basename(img_info["file_name"])
        img_path = os.path.join(self.img_dir, file_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        for ann in self.img_to_anns[img_info["id"]]:
            boxes.append(ann["bbox"])
            labels.append(self.class_to_idx[ann["class"]])

        boxes = np.array(boxes, dtype=np.float32)

        # =====================================================
        # 2. AUGMENTATION & RESIZE 
        # =====================================================
        if getattr(self, 'is_train', False):
            image, boxes, labels = self.augmentor(image, boxes, labels)
        else:
            image, boxes = self.augmentor.resize(image, boxes)

        # Chuyển ảnh sang Tensor cho PyTorch
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # =====================================================
        # 3. Create target tensor [S, S, 5 + C]
        # =====================================================
        target_tensor = torch.zeros((self.S, self.S, 5 + self.C), dtype=torch.float32)

        if len(boxes) == 0:
            return image_tensor, target_tensor

        boxes = torch.tensor(boxes, dtype=torch.float32)


        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box

            # Relative width and height [0, 1]
            w_norm = (xmax - xmin) / self.image_size
            h_norm = (ymax - ymin) / self.image_size

            # Relative center coordinates [0, 1]
            x_center = ((xmin + xmax) / 2.0) / self.image_size
            y_center = ((ymin + ymax) / 2.0) / self.image_size

            # Determine which cell the center falls into
            j = min(int(self.S * x_center), self.S - 1)
            i = min(int(self.S * y_center), self.S - 1)

            # Calculate the position of the box relative to the cell
            x_cell = self.S * x_center - j
            y_cell = self.S * y_center - i

            # Ignore if the cell is already assigned to another box 
            if target_tensor[i, j, 4].item() != 0:
                continue

            # Assign the box parameters and class label to the target tensor
            target_tensor[i, j, 0] = x_cell
            target_tensor[i, j, 1] = y_cell
            target_tensor[i, j, 2] = w_norm
            target_tensor[i, j, 3] = h_norm
            target_tensor[i, j, 4] = 1.0
            target_tensor[i, j, 5 + label] = 1.0

        return image_tensor, target_tensor
    

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


class YOLOLoss(nn.Module):
    def __init__(self, S=14, B=2, C=5, lambda_coord=5, lambda_noobj=0.5, gamma=2.0):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.gamma = gamma # Focal Loss gamma parameter

    def decode_to_global(self, box, device):
        """
        Convert box coordinates from cell-relative to global image-relative [0, 1]
        """
        B, S, _, _ = box.shape
        # Create grid of cell offsets
        grid_y, grid_x = torch.meshgrid([torch.arange(S, device=device), torch.arange(S, device=device)], indexing='ij')
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        x_global = (box[..., 0] + grid_x) / S
        y_global = (box[..., 1] + grid_y) / S
        w = box[..., 2]
        h = box[..., 3]
        
        return torch.stack([x_global, y_global, w, h], dim=-1)

    def compute_ciou(self, box1, box2):
        """Compute CIoU (Complete IoU) and IoU on global image coordinates [0, 1]"""
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2

        # 1. Calculate Intersection (IoU)
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + 1e-6
        iou = inter_area / union_area

        # 2. Calculate Center Distance Squared (d^2)
        d_sq = (box1[..., 0] - box2[..., 0])**2 + (box1[..., 1] - box2[..., 1])**2

        # 3. Calculate Diagonal of the smallest enclosing box squared (c^2)
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        c_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + 1e-6

        # 4. Calculate Aspect Ratio Penalty (v)
        w1, h1 = box1[..., 2], box1[..., 3]
        w2, h2 = box2[..., 2], box2[..., 3]
        
        # v = (4 / pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6)), 2)

        # 5. Calculate Alpha
        with torch.no_grad():
            alpha = v / ((1.0 - iou) + v + 1e-6)

        # 6. Calculate Final CIoU
        ciou = iou - (d_sq / c_sq + alpha * v)
        
        return iou, ciou

    def forward(self, predictions, target):
        device = predictions.device
        
        # 1. Extract components from predictions and target
        target_boxes = target[..., 0:4]
        target_obj = target[..., 4].unsqueeze(3)
        target_class = target[..., 5:]

        pred_box1 = predictions[..., 0:4]
        pred_conf1 = predictions[..., 4].unsqueeze(3)
        pred_box2 = predictions[..., 5:9] 
        pred_conf2 = predictions[..., 9].unsqueeze(3)
        pred_class = predictions[..., 10:]

        # --- Convert to global coordinates ---
        target_boxes_global = self.decode_to_global(target_boxes, device)
        pred_box1_global = self.decode_to_global(pred_box1, device)
        pred_box2_global = self.decode_to_global(pred_box2, device)

        # 2. Find best box for each cell based on CIoU (thay vì GIoU)
        iou1, ciou1 = self.compute_ciou(pred_box1_global, target_boxes_global)
        iou2, ciou2 = self.compute_ciou(pred_box2_global, target_boxes_global)
        
        iou1 = iou1.unsqueeze(3)
        iou2 = iou2.unsqueeze(3)
        ciou1 = ciou1.unsqueeze(3)
        ciou2 = ciou2.unsqueeze(3)

        # Mask better box for each cell
        best_box = (iou1 > iou2).float()

        pred_best_conf = best_box * pred_conf1 + (1 - best_box) * pred_conf2
        best_ciou = best_box * ciou1 + (1 - best_box) * ciou2
        
        best_ious = torch.max(iou1, iou2)
        target_best_conf = target_obj * best_ious

        # 3. Masks
        obj_mask = target_obj
        noobj_mask = 1 - obj_mask

        # =================================================================
        # 4. Calculate losses 
        # =================================================================

        # A. COORDINATE LOSS 
        loss_coord = self.lambda_coord * torch.sum(obj_mask * (1.0 - best_ciou))

        # B. CONFIDENCE LOSS 
        loss_conf_obj = torch.sum(obj_mask * F.binary_cross_entropy(pred_best_conf, target_best_conf, reduction='none'))
        
        bce_noobj1 = F.binary_cross_entropy(pred_conf1, torch.zeros_like(pred_conf1), reduction='none')
        bce_noobj2 = F.binary_cross_entropy(pred_conf2, torch.zeros_like(pred_conf2), reduction='none')
        
        focal_weight1 = (pred_conf1 ** self.gamma)
        focal_weight2 = (pred_conf2 ** self.gamma)
        
        loss_conf_noobj = self.lambda_noobj * torch.sum(noobj_mask * (focal_weight1 * bce_noobj1 + focal_weight2 * bce_noobj2))

        # C. CLASSIFICATION LOSS 
        loss_class = torch.sum(obj_mask * F.binary_cross_entropy(pred_class, target_class, reduction='none'))

        total_loss = loss_coord + loss_conf_obj + loss_conf_noobj + loss_class
        return total_loss / predictions.shape[0]


def train_model(model, train_loader, val_loader, optimizer, criterion,
                scheduler=None, epochs=50, device="cpu", eval_interval=2, save_dir='./models'):
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        model.train() 
        train_loss = 0.0
        

        current_lr = optimizer.param_groups[0]['lr']


        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Loss calculation
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step() 

            train_loss += loss.item()


        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")


        if scheduler is not None:
            scheduler.step()

        if epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch}/{epochs}] - Val Loss:   {avg_val_loss:.4f}")


            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss 
                save_path = os.path.join(save_dir, 'best.pth')

                torch.save(model.state_dict(), save_path)

                print(f"-> Saved best model at Epoch {epoch} (Val Loss: {best_val_loss:.4f})")



def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 with ResNet backbone")

    # Must-have arguments
    parser.add_argument('--train_data', type=str, required=True, help="Path to the training JSON file")
    parser.add_argument('--val_data', type=str, required=True, help="Path to the validation JSON file")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing all images")
    parser.add_argument('--val_image_dir', type=str, required=True, help="Directory containing validation images")
    parser.add_argument('--checkpoint_dir', type=str, default='./models', help="Directory to save model checkpoints")

    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--eval_interval', type=int, default=2, help="Evaluate on validation set every N epochs")

    return parser.parse_args()

def main():
    # 1. Parse command-line arguments
    args = parse_args()

    # 2. Setup device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Create dataset and dataloader
    augmentor = YOLOAugmentation(output_size=448)
    train_dataset = YOLODataset(
        json_file=args.train_data, 
        img_dir=args.image_dir, 
        augmentor=augmentor, 
        is_train=True,
        S=14)
    
    val_dataset = YOLODataset(
        json_file=args.val_data, 
        img_dir=args.val_image_dir, 
        augmentor=augmentor, 
        is_train=False, 
        S=14)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 4. Create model, loss function, and optimizer
    model = YOLOv1ResNet(S=14, B=2, C=5)

    # Load trained model in "models/best.pth" if exists
    if os.path.exists(os.path.join(args.checkpoint_dir, 'best.pth')):
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best.pth')))
        print(f"Loaded pretrained model from {os.path.join(args.checkpoint_dir, 'best.pth')}")
    

    criterion = YOLOLoss(S=14, B=2, C=5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        device=device,            
        eval_interval=args.eval_interval,
        save_dir=args.checkpoint_dir,
        scheduler=scheduler
    )
    
if __name__ == "__main__":
    main()