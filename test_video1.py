from deepsort import deepsort_rbc
import numpy as np
import cv2
from centernet import centernet_detection
from deep_sort import *
from random import randint
from random import seed
import os
from glob import glob
from face_recognition import face_locations
import threading
import re
import shutil

PATH_TO_CFG = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\pipeline.config'
PATH_TO_CKPT = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\CenterNet-8242021-141\ckpt-26'
PATH_TO_CKPT_FACE = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\Centernet-992021-1129-faces\ckpt-17'

PATH_TO_LABELS = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\label_map.txt'
PATH_TO_Model = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\networks\mars-small128.pb'

seed(1)


def get_mask(filename):
    mask = cv2.imread(filename, 0)
    mask = mask / 255.0
    return mask


class Person:
    def __init__(self, body, id):
        self.apearance = 0
        self.body = body
        self.id = id


def _image_read(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _extract_face(image, bbox, face_scale_thres=(20, 20)):
    h, w = image.shape[:2]
    try:
        (startY, startX, endY, endX) = bbox
    except:
        return None
    minX, maxX = min(startX, endX), max(startX, endX)
    minY, maxY = min(startY, endY), max(startY, endY)
    face = image[minY:maxY, minX:maxX].copy()
    (fH, fW) = face.shape[:2]
    if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
        return None
    else:
        return face


def camera_monitor():
    import time
    detector = centernet_detection(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)
    deepsort = deepsort_rbc(PATH_TO_Model)
    cap = cv2.VideoCapture(
        'Video/Pier Park Panama City_ Hour of Watching People Walk By.mp4')

    cv2.namedWindow('frame', 0)
    cv2.resizeWindow('frame', 512, 512)

    persons = []
    frame_id = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('centernet_out_3.avi', fourcc, 20.0, (512, 512))
    prev_time = 0
    while True:
        start_time = time.time()
        fps = 1/(start_time-prev_time)
        prev_time = start_time

        ret, frame = cap.read()
        if ret is False:
            frame_id += 1
            break
        elif(frame_id == 3):
            frame_id = 0
            height, width, _ = frame.shape

            out_scores, classes, detections = detector.predict(frame)

            if detections is None:
                print("No dets")
                continue

            detections = np.array(detections)
            out_scores = np.array(out_scores)

            ymin = height*detections[:, 0]
            xmin = width*detections[:, 1]
            ymax = height*detections[:, 2] - ymin
            xmax = width*detections[:, 3] - xmin

            ymin = np.reshape(ymin, (20, 1))
            xmin = np.reshape(xmin, (20, 1))
            xmax = np.reshape(xmax, (20, 1))
            ymax = np.reshape(ymax, (20, 1))

            detections = np.concatenate((xmin, ymin, xmax, ymax), axis=1)

            tracker, detections_class = deepsort.run_deep_sort(
                frame, out_scores, detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                id_num = str(track.track_id)

                body = frame[int(bbox[1]):int(bbox[3]),
                             int(bbox[0]):int(bbox[2])]

                if(any(x.id == id_num for x in persons)):
                    try:
                        newperson = next(
                            (x for x in persons if x.id == id_num), None)
                        if not os.path.exists("test/{0}".format(str(id_num))):
                            os.makedirs(
                                "test/{0}".format(str(id_num)))
                        cv2.imwrite(
                            "test/{0}/frame{1}-{2}.jpg".format(str(id_num), str(id_num), str(randint(0, 1000))), newperson.body)
                        newperson.body = body
                    except Exception as e:
                        print(e)
                else:
                    newperson = Person(body, id_num)
                    persons.append(newperson)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                    bbox[2]), int(bbox[3])), (255, 255, 255), 2)

                cv2.putText(frame, str(id_num), (int(bbox[0]), int(
                    bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (100, 255, 0), 3, cv2.LINE_AA)

            frame = cv2.resize(frame, (512, 512))
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def get_face():
    while True:
        list_foler = glob("./test/*")
        if(len(list_foler) == 0):
            continue
        foler = list_foler.pop(0)
        foldername = foler.split('\\')[1]
        list_image1 = glob(foler+"/*.jpg")
        try:
            for item in list_image1:
                image = _image_read(item)
                bboxs = face_locations(image)
                if(len(bboxs) == 0):
                    continue
                bbox = bboxs[0]
                face = _extract_face(image, bbox)
                directory = "./faces/{0}".format(foldername)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(
                    "./faces/{0}/{1}.jpg".format(foldername, randint(0, 1000)), face)
            shutil.rmtree(foler, ignore_errors=True)

        except Exception as e:
            print(e)


if __name__ == '__main__':
    t1 = threading.Thread(target=camera_monitor, args=())
    t2 = threading.Thread(target=get_face, args=())
    t1.start()
    t2.start()
    t1.join()
    t2.join()
