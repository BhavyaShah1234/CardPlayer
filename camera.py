import ast
import cv2 as cv
import threading as t

class Camera(t.Thread):
    def __init__(self, camera_id, img_h, img_w, logger):
        super(Camera, self).__init__(name='camera_processor')
        self.camera_id = camera_id
        self.img_h = img_h
        self.img_w = img_w
        self.logger = logger
        self.cap = cv.VideoCapture(self.camera_id)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.img_h)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.img_w)
        # self.cap.set(cv.CAP_PROP_EXPOSURE, 10)
        self.exit_event = False
        self.current_frame = None
        # self.cards = {}
        self.key = -1

    def update_text(self, texts: dict):
        self.texts = {}
        index = {'A': 0, 'B': 1, 'C': 2, 'ME': 3}
        for text_id in texts:
            if texts[text_id]['color'] == 'good':
                color = (0, 255, 0)
            elif texts[text_id]['color'] == 'bad':
                color = (0, 0, 255)
            else:
                color = (255, 0, 255)
            if texts[text_id]['position'] == 'game':
                position = (10, 20)
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'total':
                position = (self.img_w - 150, 20)
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'trump':
                position = (self.img_w - 150, self.img_h - 20)
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'turn':
                position = (10, self.img_h - 20)
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'deal':
                position = (int(self.img_w / 4), int(self.img_h / 2))
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'decision':
                position = (100, int(self.img_h / 4))
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'forbidden':
                position = (100, int(self.img_h / 3))
                text = texts[text_id]['text']
            elif texts[text_id]['position'][:5] == 'hands':
                player = texts[text_id]['position'][6:]
                position = (10, 300 + (index[player] * 30))
                text = texts[text_id]['text']
            elif texts[text_id]['position'] == 'expected':
                position = (int(self.img_w / 4), 20)
                text = texts[text_id]['text']
            elif texts[text_id]['position'][:6] == 'played':
                player = texts[text_id]['position'][7:]
                position = (self.img_w - 100, 300 + (index[player] * 30))
                text = texts[text_id]['text']
            else:
                position = (int(self.img_w / 2), int(self.img_h / 2))
                text = texts[text_id]['text']
            self.texts[text_id] = {'text': text, 'position': position, 'color': color}

    def set_detections(self, cards):
        self.cards = cards

    def set_exit_event(self):
        self.exit_event = True

    def run(self):
        cv.namedWindow('JUDGEMENT')
        while self.cap.isOpened():
            if self.exit_event:
                break
            ret, frame = self.cap.read()
            if not ret:
                print('CAPTURE FAILED. TRYING AGAIN....')
                self.cap = cv.VideoCapture(self.camera_id)
                self.run()
            self.current_frame = cv.resize(frame, (512, 512))
            # for c in self.cards:
            #     xmin, ymin, xmax, ymax, conf = self.cards[c]
            #     frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
            #     frame = cv.putText(frame, f'{c}:{conf}', (xmin, ymin - 10), 1, cv.FONT_HERSHEY_PLAIN, (255, 0, 255), 1)
            for text_id in self.texts:
                frame = cv.putText(frame, self.texts[text_id]['text'], self.texts[text_id]['position'], cv.FONT_HERSHEY_PLAIN, 1, self.texts[text_id]['color'], 1)   
            cv.imshow('JUDGEMENT', frame)
            self.key = cv.waitKey(1)
        self.cap.release()

    def start_capturing(self):
        self.start()

    def join_thread(self):
        self.join()
