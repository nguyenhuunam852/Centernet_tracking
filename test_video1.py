from deepsort import deepsort_rbc
import numpy as np
import cv2
from centernet import centernet_detection
from random import randint
from random import seed
import os
from glob import glob
from face_recognition import face_locations
import threading
import shutil
from pathlib import Path

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
    def __init__(self, body, Id):
        self.apearance = 0
        self.body = body
        self.id = Id


def _image_read(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _extract_face(image, bbox, face_scale_thres=(20, 20)):
    try:
        (startY, startX, endY, endX) = bbox
    except Exception as ex:
        print(ex)
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
    deep_sort = deepsort_rbc(PATH_TO_Model)
    cap = cv2.VideoCapture(
        'Video/Pier Park Panama City_ Hour of Watching People Walk By.mp4')

    cv2.namedWindow('frame', 0)
    cv2.resizeWindow('frame', 512, 512)

    persons = []
    frame_id = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Centernet_out_3.avi', fourcc, 20.0, (512, 512))
    prev_time = 0
    while True:
        start_time = time.time()
        fps = 1/(start_time-prev_time)
        prev_time = start_time

        ret, frame = cap.read()
        if ret is False:
            frame_id += 1
            break
        elif frame_id == 3:
            frame_id = 0
            height, width, _ = frame.shape

            out_scores, classes, detections = detector.predict(frame)

            if detections is None:
                print("No dets")
                continue

            detections = np.array(detections)
            out_scores = np.array(out_scores)

            y_min = height*detections[:, 0]
            x_min = width*detections[:, 1]
            y_max = height*detections[:, 2] - y_min
            x_max = width*detections[:, 3] - x_min

            y_min = np.reshape(y_min, (20, 1))
            x_min = np.reshape(x_min, (20, 1))
            x_max = np.reshape(x_max, (20, 1))
            y_max = np.reshape(y_max, (20, 1))

            detections = np.concatenate((x_min, y_min, x_max, y_max), axis=1)

            tracker, detections_class = deep_sort.run_deep_sort(
                frame, out_scores, detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                id_num = str(track.track_id)

                body = frame[int(bbox[1]):int(bbox[3]),
                             int(bbox[0]):int(bbox[2])]

                if any(x.id == id_num for x in persons):
                    try:
                        new_person = next(
                            (x for x in persons if x.id == id_num), None)
                        if not os.path.exists("test/{0}".format(str(id_num))):
                            os.makedirs(
                                "test/{0}".format(str(id_num)))
                        cv2.imwrite(
                            "test/{0}/frame{1}-{2}.jpg".format(str(id_num), str(id_num), str(randint(0, 1000))), new_person.body)
                        new_person.body = body
                    except Exception as e:
                        print(e)
                else:
                    new_person = Person(body, id_num)
                    persons.append(new_person)

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
        list_folder = sorted(Path('./test/').iterdir(), key=os.path.getmtime)
        if len(list_folder) == 0:
            continue
        folder = list_folder.pop(0)
        folder = str(folder)
        folder_name = folder.split('\\')[1]
        list_image1 = glob(folder+"/*.jpg")
        try:
            for item in list_image1:
                image = _image_read(item)
                boxes = face_locations(image)
                if len(boxes) == 0:
                    continue
                bbox = boxes[0]
                face = _extract_face(image, bbox)
                directory = "./faces/{0}".format(folder_name)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(
                    "./faces/{0}/{1}.jpg".format(folder_name, randint(0, 1000)), face)
            shutil.rmtree(folder, ignore_errors=True)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    t1 = threading.Thread(target=camera_monitor, args=())
    #t2 = threading.Thread(target=get_face, args=())
    t1.start()
    #t2.start()
    t1.join()
    #t2.join()
