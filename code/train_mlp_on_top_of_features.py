import numpy as np
from pathlib import Path
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from facebias.iotools import split_bfw_features
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop


def get_mlp_definition(input_shape, output_size=1, output_activation='sigmoid', optimizer=Adam(1e-4),  # "rmsprop",
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
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    # model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_size, activation=output_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def plot_summary(N, filepath):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), model.history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), model.history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(filepath)


do_gender = True
do_ethnicity = True

f_tr_features = "/Volumes/MyBook/bfw/bfw-cropped-aligned-features/features.pkl"
dir_val_features = '/Users/jrobby/GitHub/facerec-bias-bfw/data/utkface/features/'  # Path('./').home() /
dir_features = str(Path(f_tr_features).parent)
f_meta = '/Users/jrobby/datasets/BFW-v0_1_5/meta/bfw-fold-meta-lut.csv'
dir_gender_logs = 'logs/gender/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_ethnicity_logs = 'logs/ethnicity/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

path_weights = Path('train_models')

path_weights.mkdir(exist_ok=True, parents=True)
top_gender_model_weights_path = path_weights / "gender_fc_model.h5"
top_ethnicity_model_weights_path = path_weights / "ethnicity_fc_model.h5"
train_data_dir = "data/train"
validation_data_dir = "data/validation"
# nb_train_samples = 2000
# nb_validation_samples = 800
epochs = 100
batch_size = 16

path_tr_features = Path(f_tr_features)

# Load data, both train (BFW) and validation (UTK-Face) sets
path_val_features = Path(dir_val_features)

# train_ref, train_features, train_labels = prepare_training_features(path_tr_features)
# val_features, val_labels = load_utk_unlabeled_test(path_val_features)
optimizer = 'adam'

if do_gender:
    train_ref, train_features, train_labels, val_ref, val_features, val_labels = split_bfw_features(f_meta, dir_features)
    opt = Adam(lr=1e-4) if optimizer == 'adam' else RMSprop(0.0001, decay=1e-6)
    model = get_mlp_definition(train_features.shape[1:], optimizer=opt)  # use default settings to define gender recognizer.
    tensorboard_callback = TensorBoard(log_dir=dir_gender_logs)

    model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_features, val_labels),
        callbacks=[tensorboard_callback]
    )
    model.save_weights(str(top_gender_model_weights_path))
    plot_summary(epochs, dir_gender_logs + '/plot_summary.pdf')
if do_ethnicity:
    train_ref, train_features, train_labels, val_ref, val_features, val_labels = split_bfw_features(f_meta, dir_features, 'ethnicity')
    # perform one-hot encoding on the labels
    # lb = LabelBinarizer()
    # train_labels = lb.fit_transform(train_labels)
    # train_labels = to_categorical(train_labels,num_classes=4)
    # lb = LabelBinarizer()
    # val_labels = lb.fit_transform(val_labels)
    # val_labels = to_categorical(val_labels,num_classes=4)

    opt = Adam(lr=1e-4) if optimizer == 'adam' else RMSprop(0.0001, decay=1e-6)
    model = get_mlp_definition(train_features.shape[1:], optimizer=opt, loss='sparse_categorical_crossentropy',
                               output_activation='softmax', output_size=4)

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
    plot_summary(epochs, dir_ethnicity_logs + '/plot_summary.pdf')
