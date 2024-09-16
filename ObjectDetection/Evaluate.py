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
        self.gt_detections = COCO(temp_gt_path)

        # Save predicted detections to temp file
        with open(temp_pred_path, 'w') as coco_file:
            json.dump(self.pred_detections, coco_file)
        self.pred_detections = COCO(temp_pred_path)

        # Initialize COCO evaluation
        self.coco_eval = COCOeval(self.gt_detections, self.pred_detections, 'bbox')

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

    def run_evaluation(self):
        """
        Run COCO evaluation and print results summary.
        """
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        # Return evaluation stats with adjusted names
        results = {
            'mAP5095': self.coco_eval.stats[0],
            'mAP50': self.coco_eval.stats[1],
            'mAP75': self.coco_eval.stats[2],
            'AR5095': self.coco_eval.stats[8],
        }
        
        return results
