import pickle
import numpy as np
from pathlib import Path
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from facebias.iotools import prepare_training_features, load_utk_unlabeled_test


def get_mlp_definition(output_size=1, output_activation='signmoid', optimizer="rmsprop",
                       loss="binary_crossentropy", metrics=["accuracy"]):
    """
    Define MLP and compile loss on top. Default settings are for a binary classifier (i.e., gender recognition).
    @param output_size: number of output predictions (i.e., number of classes or just 1 for binary)
    @type output_size: int
    @param output_activation:   activation function used to normalize the output
    @param optimizer:   optimization schema
    @param loss:    type of loss function
    @param metrics:     metrics to optimize according to
    @type metrics: list
    @return: Keras model compiled with loss on top of MLP

    """
    model = Sequential()
    model.add(Flatten(input_shape=train_features.shape[1:]))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_size, activation=output_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


train_mlp = True
do_gender = True
do_ethnicity = False

f_tr_features = "/Volumes/MyBook/bfw/bfw-cropped-aligned-features/features.pkl"
dir_val_features = Path('./').home() / '/GitHub/facerec-bias-bfw/data/utkface/features/'

dir_gender_logs = 'logs/gender/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_ethnicity_logs = 'logs/ethnicity/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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
        model = get_mlp_definition('gender')

        tensorboard_callback = TensorBoard(log_dir=dir_gender_logs)  # , histogram_freq=1, write_images=False)

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
        model = get_mlp_definition('ethnicity')

        tensorboard_callback = TensorBoard(log_dir=dir_ethnicity_logs)  # , histogram_freq=1, write_images=False)

        model.fit(
            train_features,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_features, val_labels),
            callbacks=[tensorboard_callback]
        )
        model.save_weights(str(top_ethnicity_model_weights_path))
