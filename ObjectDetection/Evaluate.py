from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class EvaluateModel:
    def __init__(self, annotation_file, detection_results_file):
        """
        Initializes the COCOEvaluator with the ground truth annotations and detection results.
        
        Args:
            annotation_file (str): Path to the COCO ground truth annotations file.
            detection_results_file (str): Path to the detection results file in COCO format.
        """
        self.annotation_file = annotation_file
        self.detection_results_file = detection_results_file
        self.coco_gt = COCO(annotation_file)
        self.coco_dt = self.coco_gt.loadRes(detection_results_file)
        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')

    def run_evaluation(self):
        """
        Runs the COCO evaluation and prints the summary of results.
        """
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

    def get_results(self):
        """
        Returns the evaluation results.
        
        Returns:
            dict: A dictionary containing the evaluation results.
        """
        results = {
            'AP': self.coco_eval.stats[0],
            'AP@0.5': self.coco_eval.stats[1],
            'AP@0.75': self.coco_eval.stats[2],
            'AP (small)': self.coco_eval.stats[3],
            'AP (medium)': self.coco_eval.stats[4],
            'AP (large)': self.coco_eval.stats[5],
            'AR@1': self.coco_eval.stats[6],
            'AR@10': self.coco_eval.stats[7],
            'AR@100': self.coco_eval.stats[8],
            'AR (small)': self.coco_eval.stats[9],
            'AR (medium)': self.coco_eval.stats[10],
            'AR (large)': self.coco_eval.stats[11],
        }
        return results
