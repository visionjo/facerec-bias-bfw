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


parser = argparse.ArgumentParser(prog="Evaluate MLP Classifier(s)")
parser.add_argument('-i', '--input', type=str,
                    default='../data/bfw/features/sphereface/features.pkl',
                    help='path to the datatable, i.e., meta file (CSV)')
parser.add_argument('-d', '--datatable', type=str,
                    default='../data/bfw/meta/bfw-fold-meta-lut.csv',
                    help='path to the datatable, i.e., meta file (CSV)')
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


path_weights = Path(args.weights)
path_weights.mkdir(exist_ok=True, parents=True)


batch_size = args.batch

if args.gender:
    for val_fold in range(1, N_FOLDS + 1):
        print(f"fold {val_fold}")
        ref_tr, features_tr, labels_tr, ref_val, features_val, labels_val \
            = split_bfw_features(f_meta, dir_features, val_fold=val_fold)

        opt = Adam(lr=1e-4) if args.optimizer == 'adam' \
            else RMSprop(0.0001, decay=1e-6)
        model = get_finetuned_mlp(features_tr.shape[1:], optimizer=opt)

        model.fit(
            features_tr,
            labels_tr,
            batch_size=batch_size,
            validation_data=(features_val, labels_val),
        )
        path_in = path_weights / str(val_fold) / "gender_fc_model.h5"

        model.load_weights(str(path_in))


if args.ethnicity:
    for val_fold in range(1, N_FOLDS + 1):
        ref_tr, features_tr, labels_tr, ref_val, features_val, labels_val \
            = split_bfw_features(f_meta, dir_features, 'ethnicity')

        opt = Adam(lr=1e-4) if args.optimizer == 'adam' \
            else RMSprop(0.0001, decay=1e-6)
        model = get_finetuned_mlp(features_tr.shape[1:], optimizer=opt,
                                  loss='sparse_categorical_crossentropy',
                                  output_activation='softmax', output_size=4)

        model.fit(
            features_tr,
            labels_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(features_val, labels_val),
            callbacks=[tb_callback]
        )
        path_in = path_weights / str(val_fold) / "ethnicity_fc_model.h5"
        model.save_weights(str(path_in))
