import threading
import cv2
import numpy as np
import time
import random
import pathlib
from datetime import timedelta
from .sort import Sort
from django.conf import settings
from detection.utils import get_location
from notification.models import DetectionNotificationTable


def img_save(table, frame, count, save_img_count, video_time):
    save_path = str(table.date).replace('-', '').replace(':', '').replace(' ', '').split('+')[0]
    path = pathlib.Path(settings.MEDIA_ROOT, 'detect_objects', save_path)

    if not path.exists():
        pathlib.Path.mkdir(path)

    image = pathlib.Path(path, f'{save_img_count}.jpg')
    db_path = pathlib.Path('detect_objects', save_path, f'{save_img_count}.jpg')

    cv2.imwrite(str(image), frame[0: -110, 0: :])
    try:
        location = get_location(video_time)

        table.images.create(
            url=str(db_path),
            latitude=location.latitude,
            longitude=location.longitude,
            count_objects=count
        )

    except:
        table.images.create(
            url=str(db_path),
            count_objects=count
        )

class PotholeDetection:

    def __init__(self, weights_path, cfg_path, obj_names, frame_size, table):

        self.sort = Sort(max_age=30, min_hits=10)
        self.table = table

        self.CONFIDENCE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        self.class_names = []
        with open(obj_names, "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        net = cv2.dnn.readNet(weights_path, cfg_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(frame_size, frame_size), scale=1 / 255, swapRB=True)

        self.vc = cv2.VideoCapture(self.table.video.path)

        self.trackerIds = np.zeros(0)

        self.save_img_count = 0
        self.object_detected = 0
        self.time_video = None

        self.color_list = []
        for j in range(1000):
            self.color_list.append(
                ((int)(random.randrange(255)), (int)(random.randrange(255)), (int)(random.randrange(255))))

    def tracking(self, frame, scores, boxes):

        if len(scores) > 0:
            detect = np.zeros((len(scores), 5))
            count = 0
            bl = None
            for (score, box) in zip(scores, boxes):
                # print(int(box[0]))
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int((box[0] + box[2]))
                y2 = int((box[1] + box[3]))

                bb = np.array([x1, y1, x2, y2, score])
                detect[count, :] = bb[:]
                count += 1

            tracker = self.sort.update(detect)

            for d in tracker:
                diff = np.in1d(self.trackerIds, tracker[:, 4])
                bl = np.in1d(True, diff).all()
                cv2.rectangle(frame, ((int)(d[0]), (int)(d[1])), ((int)(d[2]), (int)(d[3])),
                              self.color_list[(int)(d[4])], 2)

            if not bl:
                if len(tracker) > 0:
                    self.trackerIds = np.append(self.trackerIds, tracker[:, 4])
                    self.save_img_count += 1
                    self.object_detected += len(tracker[:, 4])

                    t = threading.Thread(target=img_save, args=(self.table, frame, len(tracker[:, 4]),
                                                                self.save_img_count, self.time_video,))
                    t.start()

    def get_time(self, milliseconds):
        seconds_t = milliseconds // 1000
        minutes_t = 0
        hours_t = 0

        if seconds_t >= 60:
            minutes_t = seconds_t // 60
            seconds_t = seconds_t % 60

        if minutes_t >= 60:
            hours_t = minutes_t // 60
            minutes_t = minutes_t % 60

        return hours_t, minutes_t, seconds_t

    def run(self):

        start_time = self.table.date - timedelta(hours=3)

        frame_width = int(self.vc.get(3))
        frame_height = int(self.vc.get(4))

        size = (frame_width, frame_height)

        fourcc = cv2.VideoWriter_fourcc(*'H264')

        outputStream = cv2.VideoWriter('project1.mp4', fourcc, 30.0, size)



        while True:
            grabbed, frame = self.vc.read()
            if not grabbed:
                break

            milliseconds = self.vc.get(cv2.CAP_PROP_POS_MSEC)
            hours, minutes, seconds = self.get_time(milliseconds=milliseconds)
            self.time_video = start_time + timedelta(hours=hours, minutes=minutes, seconds=seconds)

            start = time.time()
            _, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
            end = time.time()

            self.tracking(frame, scores, boxes)
            start_drawing = time.time()

            end_drawing = time.time()

            fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
                1 / (end - start), (end_drawing - start_drawing) * 1000)
            cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("detections", frame[0: -110, 0: :])

            outputStream.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.vc.release()
        outputStream.release()

        cv2.destroyAllWindows()


        self.table.in_processing = False
        self.table.save()
        DetectionNotificationTable(type='detection finished', detection_id=self.table, recipient=self.table.creator).save()
