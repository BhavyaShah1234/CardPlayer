import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11m.pt', task='detection')
    model.train(data='data.yaml', epochs=5, imgsz=512, device=0, pretrained=True, batch=8, exist_ok=True, plots=True, conf=0.4, amp=False)
