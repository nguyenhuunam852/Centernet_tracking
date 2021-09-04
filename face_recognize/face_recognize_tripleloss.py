# %%
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import numpy as np
import pickle
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
# %%


def _most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label


def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def _base_network():
    model = VGG16(include_top=True, weights=None)
    dense = Dense(128)(model.layers[-4].output)
    norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense)
    model = Model(inputs=[model.input], outputs=[norm2])
    return model


# %%
faces = _load_pickle("./faces.pkl")
labels = _load_pickle("./y_labels.pkl")

faceResizes = []
for face in faces:
    face_rz = cv2.resize(face, (224, 224))
    faceResizes.append(face_rz)

X = np.stack(faceResizes)

# %%
id_train = _load_pickle("./id_train.pkl")
id_test = _load_pickle("./id_test.pkl")
labels = np.array(labels)

X_train, X_test = X[id_train], X[id_test]
y_train, y_test = labels[id_train], labels[id_test]

print(X_train.shape)
print(X_test.shape)

# %%
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = _base_network()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())

gen_train = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).repeat().shuffle(1024).batch(32)

# %%
history = model.fit(
    gen_train,
    steps_per_epoch=50,
    epochs=15)

# %%


X_train_vec = model.predict(X_train)
X_test_vec = model.predict(X_test)

y_preds = []
for vec in X_test_vec:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(X_train_vec, vec, y_train)
    y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))
