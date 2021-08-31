from deepsort import deepsort_rbc
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import cv2
import pickle
import sys
from centernet import centernet_detection
from deep_sort import *

PATH_TO_CFG = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\pipeline.config'
PATH_TO_CKPT = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\CenterNet-8242021-141\ckpt-26'
PATH_TO_LABELS = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\label_map.txt'
PATH_TO_Model = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\networks\mars-small128.pb'


def get_mask(filename):
    mask = cv2.imread(filename, 0)
    mask = mask / 255.0
    return mask


if __name__ == '__main__':
    import time
    detector = centernet_detection(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)
    deepsort = deepsort_rbc(PATH_TO_Model)
    cap = cv2.VideoCapture(
        'production ID_4750076.mp4')

    cv2.namedWindow('frame', 0)
    cv2.resizeWindow('frame', 1024, 600)

    frame_id = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('centernet_out_3.avi', fourcc, 20.0, (1024, 600))

    while True:
        ret, frame = cap.read()
        if ret is False:
            frame_id += 1
            break
        height, width, _ = frame.shape

        start_time = time.time()
        out_scores, classes, detections = detector.predict(frame)
        print("1.--- %s seconds ---" % (time.time() - start_time))

        if detections is None:
            print("No dets")
            frame_id += 1
            continue

        detections = np.array(detections)
        out_scores = np.array(out_scores)

        ymin = height*detections[:, 0]  # ymin
        xmin = width*detections[:, 1]  # xmin
        ymax = height*detections[:, 2] - ymin  # ymax
        xmax = width*detections[:, 3] - xmin  # xmax

        ymin = np.reshape(ymin, (20, 1))  # ymin
        xmin = np.reshape(xmin, (20, 1))   # xmin
        xmax = np.reshape(xmax, (20, 1))
        ymax = np.reshape(ymax, (20, 1))

        detections = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
        start_time = time.time()
        tracker, detections_class = deepsort.run_deep_sort(
            frame, out_scores, detections)
        print("2.--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id_num = str(track.track_id)

            features = track.features

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                bbox[2]), int(bbox[3])), (255, 255, 255), 2)

            cv2.putText(frame, str(id_num), (int(bbox[0]), int(
                bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            # Draw bbox from detector. Just to compare.
            for det in detections_class:
                bbox = det.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                    bbox[2]), int(bbox[3])), (255, 255, 0), 2)
        print("3.--- %s seconds ---" % (time.time() - start_time))

        frame = cv2.resize(frame, (1024, 600))
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
