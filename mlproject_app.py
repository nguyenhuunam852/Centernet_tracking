from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from centernet import centernet_detection
from deepsort import deepsort_rbc
import time

import keras
import keras.api
import keras.api._v1
import keras.api._v2
import keras.engine.base_layer_v1

PATH_TO_CFG = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\pipeline.config'
PATH_TO_CKPT = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\CenterNet-8242021-141\ckpt-26'
PATH_TO_CKPT_FACE = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\Centernet-992021-1129-faces\ckpt-17'
PATH_TO_LABELS = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\label_map.txt'
PATH_TO_Model = r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\networks\mars-small128.pb'

detector = centernet_detection(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)
deep_sort = deepsort_rbc(PATH_TO_Model)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        frame_id = 1
        cap = cv2.VideoCapture(
            r'D:\train2017\KhoaLuanTotNghiep\Person_tracking_centernet\Video\Pier Park Panama City_ Hour of Watching People Walk By.mp4')
        prev_time = 0
        while self._run_flag:
            start_time = time.time()
            fps = 1 / (start_time - prev_time)
            prev_time = start_time

            ret, cv_img = cap.read()
            if ret is False:
                frame_id += 1
                break
            if ret:
                if frame_id == 3:
                    frame_id = 0
                    height, width, _ = cv_img.shape
                    out_scores, classes, detections = detector.predict(cv_img)
                    if detections is None:
                        print("No dets")
                        continue
                    detections = np.array(detections)
                    out_scores = np.array(out_scores)

                    y_min = height * detections[:, 0]
                    x_min = width * detections[:, 1]
                    y_max = height * detections[:, 2] - y_min
                    x_max = width * detections[:, 3] - x_min

                    y_min = np.reshape(y_min, (20, 1))
                    x_min = np.reshape(x_min, (20, 1))
                    x_max = np.reshape(x_max, (20, 1))
                    y_max = np.reshape(y_max, (20, 1))

                    detections = np.concatenate((x_min, y_min, x_max, y_max), axis=1)

                    tracker, detections_class = deep_sort.run_deep_sort(
                        cv_img, out_scores, detections)

                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue

                        bbox = track.to_tlbr()
                        id_num = str(track.track_id)

                        cv2.rectangle(cv_img, (int(bbox[0]), int(bbox[1])), (int(
                            bbox[2]), int(bbox[3])), (255, 255, 255), 2)

                        cv2.putText(cv_img, str(id_num), (int(bbox[0]), int(
                            bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

                    cv2.putText(cv_img, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (100, 255, 0), 3, cv2.LINE_AA)

                    cv_img = cv2.resize(cv_img, (512, 512))
                    self.change_pixmap_signal.emit(cv_img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    frame_id += 1
            else:
                frame_id += 1
        cap.release()

    def stop(self):
        self._run_flag = False


class MyTabWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Geeks")
        self.tabs.addTab(self.tab2, "For")
        self.tabs.addTab(self.tab3, "Geeks")

        # Create first tab
        self.tab1.layout = QVBoxLayout(self)

        self.tab1.layout.addWidget(parent.image_label)
        self.tab1.layout.addWidget(parent.button)
        self.tab1.setLayout(self.tab1.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 600
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.button = QtWidgets.QPushButton(self)
        self.button.move(650, 500)
        self.button.setText('Stop')
        self.button.clicked.connect(self.activate_thread)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.tab_widget = MyTabWidget(self)
        self.setCentralWidget(self.tab_widget)

    def closeEvent(self):
        self.thread.stop()
        self.__init__()
        self.button.clicked.connect(self.activate_thread)
        # event.accept()

    def activate_thread(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.button.clicked.connect(self.thread.stop)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
