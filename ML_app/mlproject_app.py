from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import time
from re import L
import cv2
import time
import numpy as np
from centernet import Detector
import warnings


import keras
import keras.api
import keras.api._v1
import keras.api._v2
import keras.engine.base_layer_v1

PATH_TO_CFG = 'config/pipeline.config'
PATH_TO_CKPT = r'/home/nam/Desktop/Mlproject/Centernet_tracking/Centernet-9252021-1734-faces/ckpt-173'
PATH_TO_LABELS = 'config/label_map.txt'

detector = Detector(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)
# deep_sort = deepsort_rbc(PATH_TO_Model)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, detect):
        super().__init__()
        self.detector = detect
        self.video = cv2.VideoCapture(
            'rtsp://admin:nam781999@192.168.1.207:554/cam/realmonitor?channel=1&subtype=0')

    def run(self):
        frame_id = 0
        while self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                if frame_id == 3:
                    frame_id = 0
                    start_time = time.time()
                    image = cv2.resize(frame, (512, 512))

                    image, original_image, coordinate_dict = self.detector.predict(
                        image)
                    print("1.--- %s seconds ---" % (time.time() - start_time))

                    self.change_pixmap_signal.emit(image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    frame_id += 1
        self.video.release()

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

        detector = Detector(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)

        self.thread = VideoThread(detector)
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
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    a = App()
    a.show()
    sys.exit(app.exec_())
