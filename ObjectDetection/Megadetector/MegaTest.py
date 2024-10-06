import pandas as pd
from Mega import Mega
import os

RESULTS_FOLDER = os.path.join("ObjectDetection", "Megadetector","Results")

if __name__ == "__main__":
    # set altered parameters
    versions = ["a","b"]
    classifiers = [True, False]

    # create pandas df
    columns = ['version', 'classify', 'mAP5095', 'mAP50', 'mAP75', 'AR5095', 'precision', 'recall', 'f1', 'inference']
    results_df = pd.DataFrame(columns=columns)

    for version in versions:
        for classifier in classifiers:
            mega = Mega(version,device="cpu")
            results = mega.evaluate(yolo_path="Data/Formated/yolo",classify=classifier)
            row = [version, classifier]
            for column in columns:
                if column != 'version' and column != 'classify':
                    row.append(results[column])
            results_df.loc[len(results_df)] = row
            results_df.to_csv(os.path.join(RESULTS_FOLDER,"results.csv"))
            del mega