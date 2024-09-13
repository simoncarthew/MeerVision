import numpy as np
from collections import defaultdict
import cv2
import glob
import os
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, BoundingBox

class EvaluateModel:
    def __init__(self, iou_threshold=0.5):
        """
        Initialize the evaluation class with IoU threshold.
        :param iou_threshold: IoU threshold to use for evaluation.
        """
        self.iou_threshold = iou_threshold

    def load_gt_detections(self, yolo_path, img_width=None, img_height=None):
        """
        Loads YOLO format detections from text files and converts them into
        the required format with absolute bounding box coordinates.
        """
        all_detections = []

        img_files = glob.glob(os.path.join(yolo_path, "images", "test", '*.jpg'))
        if img_width is None or img_height is None:
            img = cv2.imread(img_files[0])
            img_height, img_width = img.shape[:2]

        lbl_files = glob.glob(os.path.join(yolo_path, "labels", "test", '*.txt'))

        for lbl_file in lbl_files:
            detections = []
            img_file = os.path.join(yolo_path, "images", "test", os.path.basename(lbl_file).replace('.txt', '.jpg'))

            if not os.path.exists(img_file):
                print(f"Warning: Image file {img_file} not found for label {lbl_file}")
                continue

            with open(lbl_file, 'r') as file:
                lines = file.readlines()
                img_file = os.path.basename(lbl_file).replace(".txt", ".jpg")

                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.split())

                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    detections.append({
                        'image_id': img_file,
                        'box': (x1, y1, x2, y2),
                        'class': int(class_id),
                        'confidence': 1.0  # Assuming all detections have maximum confidence if not provided
                    })

            all_detections += detections

        return all_detections

    def prepare_bounding_boxes(self, gt_list, pred_list):
        gt_boxes = []
        pred_boxes = []
        
        # Load ground truth boxes
        for gt in gt_list:
            x1, y1, x2, y2 = gt['box']
            gt_boxes.append(BoundingBox.of_bbox(
                image=gt['image_id'],
                category=gt['class'],
                xtl=x1,
                ytl=y1,
                xbr=x2,
                ybr=y2
            ))
        
        # Load predicted boxes
        for pred in pred_list:
            x1, y1, x2, y2 = pred['box']
            pred_boxes.append(BoundingBox.of_bbox(
                image=pred['image_id'],
                category=pred['class'],
                xtl=x1,
                ytl=y1,
                xbr=x2,
                ybr=y2,
                score=pred['confidence']
            ))
        
        return gt_boxes, pred_boxes

    def full_evaluation(self, pred_list, yolo_path, img_width=None, img_height=None):
        gt_list = self.load_gt_detections(yolo_path=yolo_path, img_height=img_height, img_width=img_width)

        gt_boxes, pred_boxes = self.prepare_bounding_boxes(gt_list, pred_list)
        
        # Calculate metrics at IoU threshold of 0.5
        results_50 = get_pascal_voc_metrics(gt_boxes, pred_boxes, iou_threshold=0.5)
        
        # Calculate precision, recall, and mAP50
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for cls, metric in results_50.items():
            total_tp += np.sum(metric.tp)
            total_fp += np.sum(metric.fp)
            total_fn += metric.num_groundtruth - np.sum(metric.tp)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        mAP50 = MetricPerClass.mAP(results_50)

        # Calculate metrics at IoU thresholds of 0.5 to 0.9
        results_50_90 = {}
        for iou_thresh in np.arange(0.5, 1.0, 0.05):
            results_50_90.update(get_pascal_voc_metrics(gt_boxes, pred_boxes, iou_threshold=iou_thresh))
        
        mAP50_90 = MetricPerClass.mAP(results_50_90)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"mAP50: {mAP50:.4f}")
        print(f"mAP50-90: {mAP50_90:.4f}")

        return precision, recall, mAP50, mAP50_90

if __name__ == "__main__":
    eval = EvaluateModel()
    eval.full_evaluation(pred_list=[], yolo_path='Data/Formated/yolo', img_height=640, img_width=640)
