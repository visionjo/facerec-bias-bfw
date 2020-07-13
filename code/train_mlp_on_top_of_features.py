import pickle
import numpy as np
from pathlib import Path
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow as tf

from facebias.iotools import prepare_training_features, load_utk_unlabeled_test

train_mlp = True
do_gender = True
do_ethnicity = False

f_tr_features = "/Volumes/MyBook/bfw/bfw-cropped-aligned-features/features.pkl"
dir_val_features = '/Users/jrobby/GitHub/facerec-bias-bfw/data/utkface/features/'

dir_logs = 'logs/gender/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path_weights = Path('train_models')
path_weights.mkdir(exist_ok=True, parents=True)
top_gender_model_weights_path = path_weights / "gender_fc_model.h5"
top_ethnicity_model_weights_path = path_weights / "ethnicity_fc_model.h5"
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 100
batch_size = 16

path_tr_features = Path(f_tr_features)

# Load data, both train (BFW) and validation (UTK-Face) sets
path_val_features = Path(dir_val_features)
train_ref, train_features, train_labels = prepare_training_features(path_tr_features)
val_features, val_labels = load_utk_unlabeled_test(path_val_features)

if train_mlp:
    if do_gender:
        model = Sequential()
        model.add(Flatten(input_shape=train_features.shape[1:]))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_logs)  # , histogram_freq=1, write_images=False)

        model.fit(
            train_features,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_features, val_labels),
            callbacks=[tensorboard_callback]
        )
        model.save_weights(str(top_gender_model_weights_path))
    if do_ethnicity:
        pass
