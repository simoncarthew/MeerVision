import numpy as np
from collections import defaultdict
import cv2
import glob
import os

class EvaluateModel:
    def __init__(self, iou_thresholds=(0.5,), iou_5090=True):
        """
        Initialize the evaluation class with IoU thresholds.
        :param iou_thresholds: A tuple of IoU thresholds to use for mAP evaluation (default is 0.5).
        :param iou_5090: Boolean to calculate mAP5090 (IoU from 0.5 to 0.9).
        """
        self.iou_thresholds = iou_thresholds
        self.iou_5090 = iou_5090

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.
        Each box is represented as a tuple of (x1, y1, x2, y2).
        :param box1: First bounding box.
        :param box2: Second bounding box.
        :return: IoU value.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def match_detections(self, detected_boxes, gt_boxes):
        """
        Match predicted boxes to ground truth boxes using IoU.
        :param detected_boxes: List of detected boxes (list of dicts with 'box', 'confidence', and 'class').
        :param gt_boxes: List of ground truth boxes (list of dicts with 'box' and 'class').
        :return: Matches and unmatched boxes.
        """
        matches = []
        unmatched_detections = []
        unmatched_gts = gt_boxes.copy()

        for det in detected_boxes:
            best_iou = 0
            best_gt = None
            for gt in unmatched_gts:
                iou = self.calculate_iou(det['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= self.iou_thresholds[0]:  # Default IoU threshold is 0.5
                matches.append((det, best_gt))
                unmatched_gts.remove(best_gt)
            else:
                unmatched_detections.append(det)

        return matches, unmatched_detections, unmatched_gts

    def precision_recall(self, matches, unmatched_detections, unmatched_gts):
        """
        Calculate precision and recall.
        :param matches: Matched boxes.
        :param unmatched_detections: Detections not matched with ground truth.
        :param unmatched_gts: Ground truth boxes not matched with detections.
        :return: Precision, Recall.
        """
        true_positives = len(matches)
        false_positives = len(unmatched_detections)
        false_negatives = len(unmatched_gts)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

        return precision, recall

    def calculate_ap(self, precisions, recalls):
        """
        Calculate the Average Precision (AP) given precision and recall values.
        :param precisions: List of precision values.
        :param recalls: List of recall values.
        :return: Average Precision (AP).
        """
        precisions = np.concatenate(([0.], precisions, [0.]))
        recalls = np.concatenate(([0.], recalls, [1.]))

        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        return ap

    def evaluate(self, det_boxes, gt_boxes):
        """
        Evaluate the model's detections against ground truth boxes.
        :param detected_boxes: Detections from the model (list of dicts).
        :param gt_boxes: Ground truth boxes (list of dicts).
        :return: Precision, Recall, mAP50, mAP5090.
        """
        # Match the detections to ground truth boxes
        matches, unmatched_detections, unmatched_gts = self.match_detections(det_boxes, gt_boxes)

        # Calculate precision and recall
        precision, recall = self.precision_recall(matches, unmatched_detections, unmatched_gts)

        # Calculate mAP for IoU thresholds
        if self.iou_5090:
            iou_thresholds = np.linspace(0.5, 0.9, 10)
        else:
            iou_thresholds = self.iou_thresholds

        aps = []
        for iou_thresh in iou_thresholds:
            self.iou_thresholds = (iou_thresh,)
            matches, unmatched_detections, unmatched_gts = self.match_detections(det_boxes, gt_boxes)
            precisions = [self.precision_recall(m, ud, ug)[0] for m, ud, ug in zip(matches, unmatched_detections, unmatched_gts)]
            recalls = [self.precision_recall(m, ud, ug)[1] for m, ud, ug in zip(matches, unmatched_detections, unmatched_gts)]
            ap = self.calculate_ap(precisions, recalls)
            aps.append(ap)

        # Return metrics
        mAP50 = aps[0] if aps else 0
        mAP5090 = np.mean(aps) if aps else 0

        return precision, recall, mAP50, mAP5090
    
    def load_detections(self, yolo_path, img_width=None, img_height=None):
        """
        Loads YOLO format detections from a text file and converts them into
        the required format with absolute bounding box coordinates.
        
        Args:
        - yolo_path (str): Path to the folder containing YOLO images and labels.
        - img_width (int, optional): The width of the image in pixels. 
                                    If not provided, it will be inferred.
        - img_height (int, optional): The height of the image in pixels.
                                    If not provided, it will be inferred.
        
        Returns:
        - dict: A dictionary where the key is the image file name and the value 
                is a list of dictionaries with detections in the format 
                {'box': (x1, y1, x2, y2), 'class': class_id}.
        - list: A list of image file paths corresponding to the labels.
        """
        # Load the image files
        img_files = glob.glob(os.path.join(yolo_path, "images", "test", '*.jpg'))
        all_detections = {}

        # Load the image to get its dimensions if not provided
        if img_width is None or img_height is None:
            img = cv2.imread(img_files[0])
            img_height, img_width = img.shape[:2]

        # Iterate over all labeled images
        lbl_files = glob.glob(os.path.join(yolo_path, "labels", "test", '*.txt'))

        for lbl_file in lbl_files:
            detections = []
            img_file = os.path.join(yolo_path, "images", "test", os.path.basename(lbl_file).replace('.txt', '.jpg'))

            # Ensure the corresponding image exists
            if not os.path.exists(img_file):
                print(f"Warning: Image file {img_file} not found for label {lbl_file}")
                continue

            # Open and read the label file
            with open(lbl_file, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    # Parse YOLO format: class_id x_center y_center width height
                    class_id, x_center, y_center, width, height = map(float, line.split())

                    # Convert YOLO format to (x1, y1, x2, y2)
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Append the detection in the required format
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'class': int(class_id)
                    })

            # Store detections for the corresponding image
            all_detections[os.path.basename(img_file)] = detections

        return all_detections
    
    def full_evaluation(self,yolo_path,detections, img_width=None, img_height=None):
        # load the ground truth
        gt_detections = self.load_detections(yolo_path=yolo_path,img_height=img_height,img_width=img_width)

        # test image path
        test_img_path = os.path.join(yolo_path,"images","test")

        for img_file, gt_detection in gt_detections.items():
            precision, recall, mAP50, mAP5090 = self.evaluate(det_boxes=detections[img_file],gt_boxes=gt_detection)
            # {'Suricata_suricatta_3512.jpg': [{'box': (130, 436, 149, 486), 'class': 0}, {'box': (377, 448, 407, 492), 'class': 0}
        pass

if __name__ == "__main__":
    eval = EvaluateModel()
    print(eval.load_detections(yolo_path='Data/Formated/yolo',img_height=640,img_width=640))