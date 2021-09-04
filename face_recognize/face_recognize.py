# %%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import tensorflow as tf
import os
from face_recognition import face_locations
import matplotlib.pyplot as plt
import pickle
import numpy as np
from imutils import paths
import re
import math
IMAGE_TEST = "../Dataset/khanh/001.jpg"
DATASET_PATH = "../Dataset/"
# %%
label = []
faces = []


def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
            'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]

    return meta_file, ckpt_file


def load_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.compat.v1.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.compat.v1.train.import_meta_graph(os.path.join(
            model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(),
                      os.path.join(model_exp, ckpt_file))


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1),
                      np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1-sz2+v):(sz1+sz2+v), (sz1-sz2+h):(sz1+sz2+h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def _blobImage(image, out_size=(300, 300), scaleFactor=1.0, mean=(104.0, 177.0, 123.0)):
    imageBlob = cv2.dnn.blobFromImage(image,
                                      scalefactor=scaleFactor,   # Scale image
                                      size=out_size,  # Output shape
                                      mean=mean,  # Trung bình kênh theo RGB
                                      swapRB=False,  # Trường hợp ảnh là BGR thì set bằng True để chuyển qua RGB
                                      crop=False)
    return imageBlob


def load_data(image_paths, do_random_crop, do_random_flip, image_size, face_scale_thres=(20, 20), do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = face_locations(img)
        if(len(bbox) == 0):
            continue
        box = np.array(bbox[0])
        (startY, startX, endY, endX) = box
        minX, maxX = min(startX, endX), max(startX, endX)
        minY, maxY = min(startY, endY), max(startY, endY)
        face = img[minY:maxY, minX:maxX].copy()

        (fH, fW) = face.shape[:2]
        if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
            continue
        # face = crop(face, do_random_crop, image_size)
        # face = flip(face, do_random_flip)
        label.append((image_paths[i].split("/")[2]).split("\\")[0])
        faceBlob = _blobImage(face, out_size=(
            160, 160), scaleFactor=1/255.0, mean=(0, 0, 0))
        faces.append(np.transpose(faceBlob[0], [1, 2, 0]))

        images[i, :, :, :] = np.transpose(faceBlob[0], [1, 2, 0])
    return images


# %%
dataset = get_dataset('./Dataset/')
paths1, labels = get_image_paths_and_labels(dataset)
nrof_images = len(paths1)
emb_array = np.zeros((nrof_images, 512))
BatchSize = 20
with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:

        load_model('./20180402-114759/20180402-114759.pb')
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph(
        ).get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        print('Calculating features for images')
        nrof_images = len(paths1)
        nrof_batches_per_epoch = int(
            math.ceil(1.0*nrof_images / BatchSize))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i*BatchSize
            end_index = min((i+1)*BatchSize, nrof_images)
            paths_batch = paths1[start_index:end_index]
            face_scale_thres = (20, 20)

            images = load_data(
                paths_batch, False, False, 160, face_scale_thres)

            feed_dict = {images_placeholder: images,
                         phase_train_placeholder: False}

            emb_array[start_index:end_index, :] = sess.run(
                embeddings, feed_dict=feed_dict)

emb_array = emb_array[:len(label)]


_save_pickle(label, "./y_labels.pkl")
_save_pickle(emb_array, "./embed_blob_faces.pkl")
_save_pickle(faces, "./faces.pkl")

# %%
embed_faces = _load_pickle("./embed_blob_faces.pkl")
y_labels = _load_pickle("./y_labels.pkl")

ids = np.arange(len(y_labels))

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    np.stack(embed_faces), y_labels, ids, test_size=0.2, stratify=y_labels)
#X_train = np.squeeze(X_train, axis=1)
#X_test = np.squeeze(X_test, axis=1)

print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))

_save_pickle(id_train, "./id_train.pkl")
_save_pickle(id_test, "./id_test.pkl")

# %%


def _most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label


vec = X_test[1].reshape(1, -1)
_most_similarity(X_train, vec, y_train)

# %%

y_preds = []
for vec in X_test:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(X_train, vec, y_train)
    y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))
# %%
