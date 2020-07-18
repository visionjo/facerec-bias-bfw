import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop

dir_package = str(Path('./facebias').resolve())
if dir_package not in sys.path:
    print(f"adding {dir_package} to path")
    sys.path.append(dir_package)
else:
    print(dir_package)

from facebias.models.mlp import get_finetuned_mlp
from facebias.iotools import split_bfw_features

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

N_FOLDS = 5


def plot_summary(N, filepath):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    try:
        plt.plot(np.arange(0, N), model.history.history["accuracy"],
                 label="train_acc")
        plt.plot(np.arange(0, N), model.history.history["val_accuracy"],
                 label="val_acc")
    except KeyError:
        plt.plot(np.arange(0, N), model.history.history["acc"],
                 label="train_acc")
        plt.plot(np.arange(0, N), model.history.history["val_acc"],
                 label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(filepath)


parser = argparse.ArgumentParser(prog="Train MLP Classifier(s)")
parser.add_argument('-i', '--input', type=str,
                    default='../data/bfw/features/sphereface/features.pkl',
                    help='path to the datatable, i.e., meta file (CSV)')
parser.add_argument('-d', '--datatable', type=str,
                    default='../data/bfw/meta/bfw-fold-meta-lut.csv',
                    help='path to the datatable, i.e., meta file (CSV)')
parser.add_argument('-l', '--logs', type=str, default='logs',
                    help='directory to write log output and plots')
parser.add_argument('--epochs', type=int, default=35, help='N epochs (train)')
parser.add_argument('--batch', type=int, default=64, help='Size mini-batch')
parser.add_argument('-g', '--gender', action="store_true",
                    help='train gender classifier')
parser.add_argument('-e', '--ethnicity', action="store_true",
                    help='train ethnicity classifier')
parser.add_argument('-w', '--weights', type=str, default='train_models',
                    help='path to dump trained weights')
parser.add_argument('-o', '--optimizer', type=str, default='adam',
                    help='optimizer to train with (i.e., "adam" or not)')
args = parser.parse_args()


dir_features = str(Path(args.input).parent)
f_meta = args.datatable

output_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

path_weights = Path(args.weights)
path_weights.mkdir(exist_ok=True, parents=True)

epochs = args.epochs
batch_size = args.batch

if args.gender:
    for val_fold in range(1, N_FOLDS + 1):
        print(f"fold {val_fold}")
        ref_tr, features_tr, labels_tr, ref_val, features_val, labels_val \
            = split_bfw_features(f_meta, dir_features, val_fold=val_fold)

        opt = Adam(lr=1e-4) if args.optimizer == 'adam' \
            else RMSprop(0.0001, decay=1e-6)
        model = get_finetuned_mlp(features_tr.shape[1:], optimizer=opt)
        dir_out = f"{args.logs}/gender/{output_tag}/{val_fold}"
        tb_callback = TensorBoard(log_dir=dir_out)

        model.fit(
            features_tr,
            labels_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(features_val, labels_val),
            callbacks=[tb_callback]
        )
        path_out = path_weights / str(val_fold) / "gender_fc_model.h5"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        model.save_weights(str(path_out))
        f_out = f"{args.logs}/gender/{output_tag}/{val_fold}/plot_summary"
        plot_summary(epochs, f"{f_out}.pdf")

if args.ethnicity:
    for val_fold in range(1, N_FOLDS + 1):
        ref_tr, features_tr, labels_tr, ref_val, features_val, labels_val \
            = split_bfw_features(f_meta, dir_features, 'ethnicity')

        opt = Adam(lr=1e-4) if args.optimizer == 'adam' \
            else RMSprop(0.0001, decay=1e-6)
        model = get_finetuned_mlp(features_tr.shape[1:], optimizer=opt,
                                  loss='sparse_categorical_crossentropy',
                                  output_activation='softmax', output_size=4)
        dir_out = f"{args.logs}/ethnicity/{output_tag}/{val_fold}"
        tb_callback = TensorBoard(log_dir=dir_out)

        model.fit(
            features_tr,
            labels_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(features_val, labels_val),
            callbacks=[tb_callback]
        )
        path_out = path_weights / str(val_fold) / "ethnicity_fc_model.h5"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        model.save_weights(str(path_out))

        f_out = f"{args.logs}/ethnicity/{output_tag}/{val_fold}/plot_summary"
        plot_summary(epochs, f"{f_out}.pdf")
