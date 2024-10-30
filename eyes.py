from ultralytics import YOLO

class Detector:
    def __init__(self, weights='weights\best.pt'):
        self.weights = weights
        self.model = YOLO(model=self.weights, task='detect', verbose=False)

    def detect(self, image, min_conf=0.5, max_iou=0.5, device='gpu'):
        results = self.model.predict(image, conf=min_conf, iou=max_iou, device=0 if device == 'gpu' else 'cpu', verbose=False)[0]
        bboxes = results.boxes.data.cpu().numpy().tolist()
        bboxes = [[int(xmin), int(ymin), int(xmax), int(ymax), round(conf, 2), results.names[c]] for xmin, ymin, xmax, ymax, conf, c in bboxes]
        return bboxes

    def post_process(self, bboxes):
        results = {}
        for xmin, ymin, xmax, ymax, conf, c in bboxes:
            if c not in results:
                results[c] = [xmin, ymin, xmax, ymax, conf]
            elif conf > results[c][4]:
                results[c] = [xmin, ymin, xmax, ymax, conf]
        return results
