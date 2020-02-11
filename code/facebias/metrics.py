import numpy as np
from numpy import greater_equal
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


# @author Joseph P. Robinson
# @date 18 January 2020


class Metrics:
    """
    Calculate pair-wise metrics.

    UPDATE: Now based on sklearn.metrics.

    Confusion stats:
         TP: true positive, TN: true negative,
         FP: false positive, FN: false negative

                Predicted Classes
                      p'    n'
                ___|_____|_____|
        Actual   p |     |     |
        Classes  n |     |     |

     precision = TP / (TP + FP)            per class label
     recall = TP / (TP + FN)               per class label
     specificity = TN / (FP + TN)          per class label
     fscore = 2*TP /(2*TP + FP + FN)       per class label



     True positives (TP) are documents in the same cluster; True negatives (TN)
     are two dissimilar documents in two different clusters. There are two error
     types: A (FP) decision is when two dissimilar documents are assumed the
     same. A (FN) decision is when two similar documents are in different
     classes or not considered the same.
    """

    n_samples = None
    n_predicted = None
    true_labels = None
    n_classes = None
    predicted_labels = None
    confusion_stats = {}

    def __init__(self):
        self.data_is_loaded = False

    def fit(self, true_labels, predicted_labels, set_confusion_stats=True):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.n_samples = len(true_labels)
        self.n_classes = len(np.unique(self.true_labels))
        self.data_is_loaded = True

        if set_confusion_stats:
            self._set_confusion_stats()

    def __repr__(self):
        if self.confusion_stats:
            stats = self.confusion_stats
            tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
        else:
            tp = tn = fp = fn = "Not Set"
        return (
            "CONFUSION METRICS:\n"
            "===============\n"
            "TP:\t{}\n"
            "TN:\t{}\n"
            "FP:\t{}\n"
            "FN:\t{}\n"
            "N_CLASSES:\t{}\n"
            "N_SAMPLES:\t{}".format(tp, tn, fp, fn, self.n_classes,
                                    self.n_samples)
        )

    def _check_state(self, check_stats=False):
        if self.n_samples is None or not self.data_is_loaded:
            print(
                "Data must to set. Return NONE. See self.fit() in {}()".format(
                    self.__class__
                )
            )

            return False
        if check_stats and not self.confusion_stats:
            self._set_confusion_stats()

        return True

    def _set_confusion_stats(self):
        """
        Calculate TP, FP, TN, and FN and store in dictionary container.
        :return: Confusion stats {TP, FP, TN, FN} (dictionary)
        """
        tn, fp, fn, tp = confusion_matrix(
            self.true_labels, self.predicted_labels
        ).ravel()

        (
            self.confusion_stats["tn"],
            self.confusion_stats["fp"],
            self.confusion_stats["fn"],
            self.confusion_stats["tp"],
        ) = (tn, fp, fn, tp)

        self.confusion_stats["n_neg"] = tn + fn
        self.confusion_stats["n_pos"] = tp + fp

    def precision(self):
        """
        Precision (P): How accurate are the positive predictions.

        Precision = TP / (TP + FP) (per class)
        :return: Precision value (float)
        """
        if not self._check_state():
            return None
        return precision_score(self.true_labels, self.predicted_labels)

    def recall(self):
        """
        Recall (R): Coverage of actual positive sample.

        R = TP / (TP + FN)
        :return: Recall value (float)
        """
        if not self._check_state():
            return None
        return recall_score(self.true_labels, self.predicted_labels)

    def accuracy(self):
        """
        Accuracy (Acc): Overall performance of model

        Acc = (TP + TN) / (TP + FP + FN + TN)
        """
        if not self._check_state():
            return None
        return accuracy_score(self.true_labels, self.predicted_labels)

    def specificity(self):
        """
        TODO - implement (stats["tn"] / (stats["tn"] + stats["fp"]))
        Recall = TN / (TN + FP)
        """
        if not self._check_state():
            return None
        pass

    def f1score(self):
        """
        Recall = 2TP / (2TP + FP + FN)
        """

        if not self._check_state():
            return None
        return f1_score(self.true_labels, self.predicted_labels)

    def calculate_negative_rates(self):
        """
        Calculate FMR and FNMR.
        :return:
        """
        if not self._check_state(check_stats=True):
            return None

        tn, fn, total_negative = (
            self.confusion_stats["tn"],
            self.confusion_stats["fn"],
            self.confusion_stats["n_neg"],
        )

        fm_rate = tn / total_negative
        fnm_rate = fn / total_negative

        return fnm_rate, fm_rate


def calculate_tar_and_far_values(y_true, scores):
    """
    Get TAR (TPR) and FAR (FNR) across various thresholds (via roc_curve)
    :param y_true:   ground truth label, boolean (1 if match; else, 0)
    :param scores:   scores for each pair.
    :return:    list of tuples (FAR, TAR, thresholds)
    """
    fpr, tar, thresholds = roc_curve(y_true, scores, pos_label=1)
    far = 1 - tar
    return far, tar, thresholds


def calculate_det_curves(y_true, scores):
    """
    Calculate false match rates, both for non-matches and matches
    :param y_true:   ground truth label, boolean (1 if match; else, 0)
    :param scores:   scores for each pair.
    :return:    list of tuples (false-match and false-non-match rates.
    """

    # y_pred = threshold_scores(scores, threshold)
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr, thresholds


def sum_tp(threshold, scores, op=greater_equal):
    return sum([1 if op(score, threshold) else 0 for score in list(scores)])


def sum_fn(threshold, scores, op=greater_equal):
    return sum([0 if op(score, threshold) else 1 for score in list(scores)])


def sum_tn(threshold, scores, op=greater_equal):
    return sum([0 if op(score, threshold) else 1 for score in list(scores)])


def sum_fp(threshold, scores, op=greater_equal):
    return sum([1 if op(score, threshold) else 0 for score in list(scores)])
