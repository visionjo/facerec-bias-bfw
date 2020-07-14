import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from facebias.iotools import split_bfw_features
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten
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
    plt.plot(np.arange(0, N), model.history.history["accuracy"],
             label="train_acc")
    plt.plot(np.arange(0, N), model.history.history["val_accuracy"],
             label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(filepath)


parser = argparse.ArgumentParser(prog="Train MLP Classifier(s)")
parser.add_argument('-i', '--input', type=str,
                    default='bfw/features/sphereface/features.pkl',
                    help='path to the datatable, i.e., meta file (CSV)')
parser.add_argument('-d', '--datatable', type=str,
                    default='data/bfw/meta/bfw-fold-meta-lut.csv',
                    help='path to the datatable, i.e., meta file (CSV)')
parser.add_argument('-l', '--logs', type=str, default='logs',
                    help='directory to write log output and plots')
parser.add_argument('--epochs', type=int, default=100, help='N epochs (train)')
parser.add_argument('--batch', type=int, default=16, help='Size mini-batch')
parser.add_argument('-g', '--gender', action="store_true", type=bool,
                    help='train gender classifier')
parser.add_argument('-e', '--ethnicity', action="store_true", type=bool,
                    help='train ethnicity classifier')
parser.add_argument('-w', '--weights', type=str, default='train_models',
                    help='path to dump trained weights')
parser.add_argument('-o', '--optimizer', type=str, default='adam',
                    help='optimizer to train with (i.e., "adam" or not)')
args = parser.parse_args()

dir_features = str(Path(args.input).parent)
f_meta = args.input

output_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

path_weights = Path(args.weights)
path_weights.mkdir(exist_ok=True, parents=True)

epochs = args.epochs
batch_size = args.batch

if args.gender:
    train_ref, train_features, train_labels, val_ref, val_features, val_labels \
        = split_bfw_features(f_meta, dir_features)

    opt = Adam(lr=1e-4) if args.optimizer == 'adam' \
        else RMSprop(0.0001, decay=1e-6)
    model = get_mlp_definition(train_features.shape[1:], optimizer=opt)
    tb_callback = TensorBoard(log_dir=f"{args.logs}/gender/{output_tag}")

    model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_features, val_labels),
        callbacks=[tb_callback]
    )
    model.save_weights(str(path_weights / "gender_fc_model.h5"))
    plot_summary(epochs, args.logs + '/gender/plot_summary.pdf')

if args.ethnicity:
    train_ref, train_features, train_labels, val_ref, val_features, val_labels \
        = split_bfw_features(f_meta, dir_features, 'ethnicity')

    opt = Adam(lr=1e-4) if args.optimizer == 'adam' \
        else RMSprop(0.0001, decay=1e-6)
    model = get_mlp_definition(train_features.shape[1:], optimizer=opt,
                               loss='sparse_categorical_crossentropy',
                               output_activation='softmax', output_size=4)

    tb_callback = TensorBoard(log_dir=f"{args.logs}/ethnicity/{output_tag}")

    model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_features, val_labels),
        callbacks=[tb_callback]
    )
    model.save_weights(str(path_weights / "ethnicity_fc_model.h5"))
    plot_summary(epochs, args.logs + '/ethnicity/plot_summary.pdf')
