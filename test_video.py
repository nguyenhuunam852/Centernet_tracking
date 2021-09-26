from deepsort import deepsort_rbc
import numpy as np
import cv2
from centernet import centernet_detection
from deep_sort import *
from face_recognition import face_locations

PATH_TO_CFG = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\pipeline.config'
PATH_TO_CKPT = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\Centernet-992021-1129-faces\ckpt-17'
PATH_TO_LABELS = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\label_map.txt'
PATH_TO_Model = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\checkpoints'


def get_mask(filename):
    mask = cv2.imread(filename, 0)
    mask = mask / 255.0
    return mask


list_capture = []


class Person:
    def __init__(self, x, id):
        self.apearance = 0
        self.x = x
        self.id = id


if __name__ == '__main__':
    detector = centernet_detection(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)

    cap = cv2.VideoCapture(
        'Video/Pier Park Panama City_ Hour of Watching People Walk By.mp4')

    cv2.namedWindow('frame', 0)
    cv2.resizeWindow('frame', 512, 512)

    frame_id = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('centernet_out_3.avi', fourcc, 20.0, (512, 512))

    while True:
        ret, frame = cap.read()
        if ret is False:
            frame_id += 1
            break
        if frame_id == 3:
            frame_id = 0
            height, width, _ = frame.shape
            frame = cv2.resize(frame, (512, 512))
            detection_scores, detection_classes, detection_boxes = detector.predict(
                frame)
            for index, face in enumerate(detection_boxes):
                face = np.array(face)
                if detection_scores[index] > 0.6:
                    y_min = 512*face[0]
                    x_min = 512*face[1]
                    y_max = 512*face[2]
                    x_max = 512*face[3]
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(
                        x_max), int(y_max)), (255, 255, 255), 2)

            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id = 1
        else:
            frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
