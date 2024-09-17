from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
import os
import json
import cv2

class EvaluateModel:
    def __init__(self, yolo_path, pred_detections, img_width=None, img_height=None, temp_gt_path=os.path.join('ObjectDetection','temp_gt.json'), temp_pred_path=os.path.join('ObjectDetection','temp_pred.json')):
        """
        Initializes the evaluator with ground truth and detection results.
        
        Args:
            yolo_path (str): Path to the directory containing YOLO formatted data.
            img_width (int, optional): Width of the images. If None, the width will be inferred from the images.
            img_height (int, optional): Height of the images. If None, the height will be inferred from the images.
        """
        self.yolo_path = yolo_path
        self.img_width = img_width
        self.img_height = img_height

        # Load and align ground truth detections
        self.gt_detections = self.load_gt_detections(yolo_path)

        # Align image IDs of predicted detections with ground truth
        self.pred_detections = self.align_image_ids(pred_detections, self.gt_detections)

        # Save ground truth to temp file
        with open(temp_gt_path, 'w') as coco_file:
            json.dump(self.gt_detections, coco_file)
        self.gt_coco = COCO(temp_gt_path)

        # Save predicted detections to temp file
        with open(temp_pred_path, 'w') as coco_file:
            json.dump(self.pred_detections, coco_file)
        self.pred_coco = COCO(temp_pred_path)

        # Initialize COCO evaluation
        self.coco_eval = COCOeval(self.gt_coco, self.pred_coco, 'bbox')

        # remove temporary files
        os.remove(temp_gt_path)
        os.remove(temp_pred_path)

    def load_gt_detections(self, yolo_path):
        """
        Load YOLO annotations and convert them to COCO format for evaluation.
        """
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "meerkat"}]  # meerkat as the single category
        }

        annotation_id = 0
        image_dir = os.path.join(yolo_path, "images", "test")

        for image_id, image_name in enumerate(os.listdir(image_dir)):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(image_dir, image_name)
                img = cv2.imread(image_path)
                height, width, _ = img.shape

                # Image info for COCO format
                image_info = {
                    "id": image_id,
                    "file_name": image_name,
                    "height": height,
                    "width": width
                }
                coco_format["images"].append(image_info)

                # Load YOLO annotations
                txt_file = os.path.join(yolo_path, "labels", "test", image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
                if not os.path.exists(txt_file):
                    continue

                with open(txt_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                        x_min = (x_center - bbox_width / 2) * width
                        y_min = (y_center - bbox_height / 2) * height
                        bbox_width = bbox_width * width
                        bbox_height = bbox_height * height

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0,
                            "segmentation": []
                        }
                        coco_format["annotations"].append(annotation_info)
                        annotation_id += 1
        return coco_format

    def align_image_ids(self, predicted_coco, ground_truth_coco):
        """
        Align image IDs between predicted and ground truth datasets based on file names.
        """
        file_name_to_id = {img['file_name']: img['id'] for img in ground_truth_coco['images']}

        for pred_img in predicted_coco['images']:
            file_name = pred_img['file_name']
            if file_name in file_name_to_id:
                new_id = file_name_to_id[file_name]
                pred_img['id'] = new_id

                for annotation in predicted_coco['annotations']:
                    if annotation['image_id'] == pred_img['id']:
                        annotation['image_id'] = new_id

        return predicted_coco

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def compute_precision_recall(self, predictions, ground_truths, iou_threshold=0.6):
        tp, fp, fn = 0, 0, 0
        used_gt = set()

        # Convert COCO annotations to dictionary for easier access
        gt_by_id = {gt['id']: gt for gt in ground_truths['annotations']}
        
        for pred in predictions['annotations']:
            pred_box = pred['bbox']
            pred_class = pred['category_id']
            
            best_iou = 0
            best_gt = None
            
            for gt_id, gt in gt_by_id.items():
                if gt_id in used_gt:
                    continue
                
                gt_box = gt['bbox']
                gt_class = gt['category_id']
                
                if pred_class == gt_class:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            
            if best_gt is not None and best_iou >= iou_threshold:
                tp += 1
                used_gt.add(best_gt['id'])
            else:
                fp += 1
        
        fn = len(ground_truths['annotations']) - len(used_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

    def run_evaluation(self):
        """
        Run COCO evaluation and print results summary.
        """
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        precision, recall, f1 = self.compute_precision_recall(self.pred_detections,self.gt_detections)

        # Return evaluation stats with adjusted names
        results = {
            'mAP5095': self.coco_eval.stats[0],
            'mAP50': self.coco_eval.stats[1],
            'mAP75': self.coco_eval.stats[2],
            'AR5095': self.coco_eval.stats[8],
            'precision':precision,
            'recall':recall,
            'f1':f1
        }
        
        return results