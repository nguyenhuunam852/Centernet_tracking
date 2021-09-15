from threading import Thread
import threading
import time


def cal_square(numbers):
    print("calculate square number")
    for n in numbers:
        time.sleep(0.2)
        print('square:', n.id)


def cal_cube(numbers):
    print("calculate cube number \n")
    for n in numbers:
        time.sleep(0.2)
        print('cube:', n.id)


class person:
    def __init__(self, id):
        self.id = id


arr = []

try:
    p1 = person(1)
    p2 = person(2)

    arr.append(p2)
    arr.append(p1)
    t = time.time()
    t1 = threading.Thread(target=cal_square, args=(arr,))
    t2 = threading.Thread(target=cal_cube, args=(arr,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("done in ", time.time() - t)
except:
    print("error")

    # elif (frame_id == 4):
    #     frame_id = 0
    #     if(len(persons) == 0):
    #         continue
    #     list_image = []
    #     for person in persons:
    #         image = cv2.resize(person.body, (512, 512))
    #         list_image.append(image)

    #     list_image = np.array(list_image)
    #     out_scores, classes, detections = face_detector.face_predict(
    #         list_image)

    #     detections = np.array(detections)
    #     i = 0
    #     for image, score, detection in zip(list_image, out_scores, detections):
    #         for bbox, con in zip(detection, score):
    #             if(con > 0.4):
    #                 bbox[1] = bbox[1]*image.shape[0]
    #                 bbox[0] = bbox[0]*image.shape[1]
    #                 bbox[3] = bbox[3]*image.shape[1]
    #                 bbox[2] = bbox[2]*image.shape[0]

    #                 face = image[int(bbox[1]):int(bbox[3]),
    #                              int(bbox[0]):int(bbox[2])]
    #                 cv2.imwrite("test/frame{0}.jpg".format(str(i)), face)
    #                 i += 1
