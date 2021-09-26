
# %%
import glob
from re import L
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')

PATH_TO_CFG = 'pipeline.config'
PATH_TO_CKPT = r'/home/nam/Desktop/Mlproject/Centernet_tracking/Centernet-9252021-1734-faces/ckpt-173'
PATH_TO_LABELS = 'label_map.txt'


class Detector(object):
    def __init__(self, path_config, path_ckpt, path_to_labels):
        self.path_config = path_config
        self.path_ckpt = path_ckpt
        self.label_path = path_to_labels

        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        self.detection_model = self.load_model()

        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def detect_fn(self, image):
        with tf.device('/device:GPU:0'):
            image, shapes = self.detection_model.preprocess(image)
            prediction_dict = self.detection_model.predict(image, shapes)
            detections = self.detection_model.postprocess(
                prediction_dict, shapes)
        return detections

    def load_model(self):
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.path_config)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.path_ckpt).expect_partial()

        return detection_model

    def predict(self, image):
        original_img = np.copy(image)

        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        # num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        self.detection_scores = detections['detection_scores']
        self.detection_classes = detections['detection_classes']
        self.detection_boxes = detections['detection_boxes']

        # draw bounding boxes and labels
        image, coordinate_dict = self.draw(image)

        return image, original_img, coordinate_dict

    def draw(self, img):
        coordinate_dict = dict()
        height, width, _ = img.shape
        li = []

        for i, score in enumerate(self.detection_scores):
            if score < 0.3:
                continue

            self.detection_classes[i] += 1
            # if background, ignore
            if self.detection_classes[i] == 0:
                continue

            label = str(self.category_index[self.detection_classes[i]]['name'])
            ymin, xmin, ymax, xmax = self.detection_boxes[i]
            real_xmin, real_ymin, real_xmax, real_ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(
                ymax * height)

            curr = real_xmax * real_ymax - real_ymin * real_xmin
            status = check_overlap(curr, li)
            # if status == 1:
            #     continue
            li.append(real_xmax * real_ymax - real_ymin * real_xmin)
            # check overlap bounding boxes

            cv2.rectangle(img, (real_xmin, real_ymin),
                          (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(img, label, (real_xmin, real_ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.5)
            coordinate_dict[label] = (
                real_xmin, real_ymin, real_xmax, real_ymax)

        return img, coordinate_dict


def check_overlap(curr, li):
    for va in li:
        # overlap
        if abs(va - curr) < 1000:
            return 1
    return 0


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('centernet_out_3.avi', fourcc, 20.0, (512, 512))


class CamApp:
    def __init__(self, detect):
        self.video = cv2.VideoCapture(
            'rtsp://admin:nam781999@192.168.1.207:554/cam/realmonitor?channel=1&subtype=0')

        self.detector = detect

    def showVideo(self):
        frame_id = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('out_3.avi', fourcc, 20.0, (512, 512))

        while self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                if frame_id == 3:
                    frame_id = 0
                    start_time = time.time()
                    image = cv2.resize(frame, (512, 512))
                    image, original_image, coordinate_dict = self.detector.predict(
                        image)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print("1.--- %s seconds ---" % (time.time() - start_time))

                    cv2.imshow('test', image)
                    out.write(image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    frame_id += 1
        self.video.release()


detector = Detector(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)

app = CamApp(detector)
cv2.namedWindow("cam-test", cv2.WINDOW_AUTOSIZE)
app.showVideo()

# list_image = glob.glob(
#     r'D:\train2017\KhoaLuanTotNghiep\RarFile\WIDER_val\WIDER_val\images\0--Parade\*.jpg')

# for index, image in enumerate(list_image):
#     frame = cv2.imread(image)
#     frame = cv2.resize(frame, (512, 512))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image, original_image, coordinate_dict = app.detector.predict(frame)
#     cv2.imwrite('./testfolder/{0}.png'.format(str(index)), image)


# detector = Detector(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)
# image = cv2.imread(image_path)

# frame = cv2.imread(
#     'test/luissuarez_pepi.jpg')
# image = cv2.resize(frame, (512, 512))

# image, original_image, coordinate_dict = app.detector.predict(image)
# cv2.imwrite('./testfolder/{0}.png'.format('test'), image)

# image = cv2.resize(frame, (512, 512))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image, original_image, coordinate_dict = app.detector.predict(image)
# for item in list_faces:
#     coor = item['box']
#     cv2.rectangle(image, (coor[0], coor[1]),
#                   (coor[0]+coor[2], coor[1]+coor[3]), (255, 0, 0), 2)

# cv2.imwrite('./testfolder/{0}.png'.format('test'), image)
# start = time.time()
# image, original_image, coordinate_dict = detector.predict(image)
# end = time.time()
# print("Estimated time: ", end - start)
# cv2.imwrite('corner_test.png', image)
# cv2.imshow('test', image)
# cv2.waitKey(0)

# %%
