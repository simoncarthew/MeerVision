import os
import pandas as pd
from Mega import Mega

# variable variables
versions = ["a","b"]
classifys = [True, False]

# static variables
test_yolo_path = os.path.join("Data","Formated","yolo")
class_name = "resnet50"
class_path = os.path.join("ObjectDetection", "Megadetector", "resnet_test.pth")
results_save_path = os.path.join("ObjectDetection", "Megadetector", "results.csv")

# results variables
columns = ['version', 'classify', 'classify_name', 'mAP5095', 'mAP75', 'mAP50', 'precision', 'recall', 'f1', 'inference']
results_df = pd.DataFrame(columns=columns)

# main loop
for version in versions:
    for classify in classifys:
        mega = Mega(version=version, class_path=class_path, class_name=class_name)
        results = mega.evaluate(yolo_path=test_yolo_path,classify=classify)
        results_df.loc[len(results_df)] = [version, classify, class_name, results['mAP5095'], results['mAP75'], results['mAP50'], results['precision'], results['recall'], results['f1'], results['inference']]
        results_df.to_csv(results_save_path, index=False)
        del mega