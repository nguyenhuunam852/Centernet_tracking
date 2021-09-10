from re import L
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


class centernet_detection():
    def __init__(self, path_config, path_ckpt, path_to_labels):
        self.path_config = path_config
        self.path_ckpt = path_ckpt
        self.label_path = path_to_labels

        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        self.detection_model = self.load_model()

    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        with tf.device('/GPU:0'):
            prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(
            prediction_dict, shapes)
        return detections

    def load_model(self):
        configs = config_util.get_configs_from_pipeline_file(self.path_config)
        model_config = configs['model']

        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.path_ckpt).expect_partial()
        return detection_model

    def predict(self, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)
        detection_scores = detections['detection_scores']
        detection_classes = detections['detection_classes']
        detection_boxes = detections['detection_boxes']

        return detection_scores, detection_classes, detection_boxes
