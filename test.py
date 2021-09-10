# %%
import time
from mtcnn import MTCNN
import cv2
import tensorflow as tf
from face_recognition import face_locations

# %%
img = cv2.cvtColor(cv2.imread("test/test1.jpg"), cv2.COLOR_BGR2RGB)

start_time = time.time()
with tf.device('/GPU:1'):
    faces = face_locations(img)

print("--- %s seconds ---" % (time.time() - start_time))

# %%
