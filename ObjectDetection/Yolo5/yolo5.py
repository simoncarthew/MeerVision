import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
from yolov5 import train, detect, val  # Make sure YOLOv5 repository is in your PYTHONPATH

class Yolo5:
    def __init__(self, model_size='s', model_path = None, pretrained = True, device=None):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_path is not None:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            model_size = "yolov5" + model_size
            self.model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=pretrained).to(self.device)
        
        # Determine the model's precision
        self.model_dtype = next(self.model.parameters()).dtype
    
    def freeze_layers(self, freeze):
        # Print the layers that will be frozen
        for k, v in self.model.named_parameters():
            if any(f"model.{x}." in k for x in range(freeze)):
                v.requires_grad = False
            else:
                v.requires_grad = True

    def train(self, data_path, lr = 0.01, epochs=30, batch_size=16, img_sz=640, freeze = 10, optimizer = 'SGD', augment = True, save_path = "ObjectDetection/Yolo5"):
        self.freeze_layers(freeze)

        train.run(
            data=data_path,
            lr0 = lr,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_sz,
            device=self.device,
            project = save_path,
            optimizer = optimizer,
            augment=augment,
            name = "train"
        )
    
    def detect_video(self, video_path, output_path=None, conf_thres=0.25):
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Set up video writer (optional)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = self.model(frame)

            # Render results on the frame
            rendered_frame = results.render()[0]  # returns a list of images

            # Display the frame
            cv2.imshow('YOLOv5 Detection', rendered_frame)

            # Write the frame to the output video
            if output_path: out.write(rendered_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def evaluate_model(self, data_path, model_path, save_path, task="test"):
        results = val.run(
            data=data_path,
            weights=model_path,
            device=self.device,
            project=save_path,
            task=task
        )
        
        # Unpack the results
        mp, mr, map50, map, *losses = results[0]  # Unpack the first tuple containing metrics
        maps = results[1]  # Class-specific mAPs
        times = results[2]  # Inference time and NMS time
        
        # Create the results dictionary
        results_dict = {
            'metrics/precision(B)': mp,
            'metrics/recall(B)': mr,
            'metrics/mAP50(B)': map50,
            'metrics/mAP50-95(B)': map,
            'loss/box_loss': losses[0],  # Assuming losses are [box_loss, obj_loss, cls_loss]
            'loss/obj_loss': losses[1],
            'loss/cls_loss': losses[2],
            'class_specific_mAPs': maps.tolist(),  # Convert to list if needed
            'times/inference_time': times[0],
            'times/nms_time': times[1]
        }
        
        return results_dict

if __name__ == "__main__":

    # Example usage
    yolo = Yolo5(model_size='s',model_path="ObjectDetection/Yolo5/train/weights/best.pt")
    # yolo.train(data_path='Data/Formated/yolo/dataset.yaml', epochs=1)
    print(yolo.evaluate_model("Data/Formated/yolo/dataset.yaml","ObjectDetection/Yolo5/train/weights/best.pt"))
    # yolo.detect_video(video_path='Data/YoutubeCameraTrap/A Wide shot of three Juvenile Meerkat or Suricate, Suricata suricatta  just outside their burrow..mp4')

