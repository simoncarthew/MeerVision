import cv2
import torch
from PytorchWildlife.models import detection as pw_detection

class Mega:
    def __init__(self, version=5, img_size=(1280, 1280)):
        if version == 5:
            self.model = pw_detection.MegaDetectorV5()
        else:
            raise ValueError("Invalid mega version.")
        
        self.img_size = img_size

    def detect(self, image_path, show=False, conf_thresh=0):
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, self.img_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            results = self.model.single_image_detection(img_tensor)

        # Process results if necessary
        detected_boxes = []
        for detection in results:
            for box in detection:
                conf = box['confidence'].item()
                if conf >= conf_thresh:
                    x1, y1, x2, y2 = map(int, box['box'])
                    detected_boxes.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': box['class']
                    })

        if show:
            self.draw_detection(img_resized, detected_boxes)

        return detected_boxes

    def draw_detection(self, img, detected_boxes, thresh=0):
        for detected in detected_boxes:
            x1, y1, x2, y2 = detected['box']
            conf = detected['confidence']
            cls = detected['class']
            label = f'Class {cls}'

            if conf > thresh:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Detection Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mega = Mega()
    print(mega.detect(image_path="Data/Formated/yolo/images/test/Suricata_suricatta_1657.jpg"))