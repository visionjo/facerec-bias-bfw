from itertools import combinations
from math import factorial

import numpy as np
from numpy import greater_equal
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder


# @author Joseph P. Robinson
# @date 6 January 2020


def label_encoder(labels):
    return LabelEncoder().fit_transform(labels)


def align_pseudo_labels(*labels):
    """
    Utility function that aligns variable number of label lists, i.e.,
        true_ids,cluster_ids=align_pseudo_labels(true_ids, cluster_ids)
    :param labels:
    :return:
    """
    return (label_encoder(l) for l in labels)


def nchoosek(n, k=2):
    """
    Determines number of combinations from expressions of form n choose k.
    n choose k = [n ; k] = n!/k!(n-k)! for 0 <= k <= n, where ! is factorial.
    :param n:  The total number of items.
    :param k:  The number to choose each time.
    :return:   n choose k (see description above)
    """
    if k > n or k == 0:
        # if elements to choose is less than elements to choose from then No. of combinations is 0
        return 0
    elif n == 1 and k == 1:
        # If there is 1 element to choose from (i.e., n=1), and more than 1 element to choose (k>1), return 1
        return 1
    return factorial(n) / (factorial(k) * factorial(n - k))


class Metrics:
    """
    Calculate pair-wise metrics.

        Note, all metrics handle pairwise relationships (i.e., counting pairs)

                Predicted Classes
                      p'    n'
                ___|_____|_____|
        Actual   p |     |     |
        Classes  n |     |     |

     precision = TP / (TP + FP)            per class label
     recall = TP / (TP + FN)               per class label
     specificity = TN / (FP + TN)          per class label
     fscore = 2*TP /(2*TP + FP + FN)       per class label


     TP: true positive, TN: true negative,
     FP: false positive, FN: false negative

     True positives (TP) are documents in the same cluster; True negatives (TN) are two dissimilar documents in two
     different clusters. There are two error types: A (FP) decision is when two dissimilar documents get clustered
     together. A (FN) decision is when two similar documents are in different clusters.
    """

    feats_a = None
    n_clusters = None
    n_pairs = None
    n_samples = None
    n_predicted = None
    true_labels = None
    n_classes = None
    predicted_labels = None
    tp = None
    fp = None
    tn = None
    fn = None

    def __init__(self):
        self.data_is_loaded = False

    def fit(self, true_labels, predicted_labels, calculate_stats=True):
        true_labels, predicted_labels = align_pseudo_labels(
            true_labels, predicted_labels
        )
        self.feats_a = true_labels
        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        self.n_samples = len(true_labels)
        self.n_classes = len(np.unique(self.true_labels))
        self.n_clusters = len(np.unique(self.predicted_labels))
        if calculate_stats:
            stats = self.confusion_matrix_values(true_labels, predicted_labels)
            self.tp = stats["TP"]
            self.tn = stats["TN"]
            self.fp = stats["FP"]
            self.fn = stats["FN"]
        self.n_pairs = nchoosek(self.n_samples, 2)
        self.data_is_loaded = True

    def __repr__(self):
        return (
            "Pairwise stats:\n"
            "===============\n"
            "TP:\t{}\n"
            "TN:\t{}\n"
            "FP:\t{}\n"
            "FN:\t{}\n"
            "NCLASSES:\t{}\n"
            "NSAMPLES:\t{}".format(
                self.tp, self.tn, self.fp, self.fn, self.n_classes, self.n_samples
            )
        )

    def func_help(self):
        if self.n_samples is None:
            print("Data must be fit(): Return NONE")
            return False
        elif self.n_pairs is None:
            print(
                "Pairs need to be calculated. See self.fit() in {}()".format(
                    self.__class__
                )
            )
            return False
        return True

    def precision(self):
        """
        Precision (P): How accurate are the positive predictions.

        Precision = TP / (TP + FP) (per class)
        :return: Precision value (float)
        """
        if not self.func_help():
            return None

        if self.n_samples == 1:
            # special case: no way of any FP
            # Note, should not get reached, since helper checks npairs>0, thus, >=2 samples.
            return 1
        if self.fp == 0:
            if not self.tp + self.tn > 0:
                print("WARNING: precision of 1 returned but no correct prediction.")
                print(self)
            # special case: when labels were all marked positive
            return 1

        return self.tp / (self.tp + self.fp)

    def recall(self):
        """
        Recall (R): Coverage of actual positive sample.

        R = TP / (TP + FN)
        :return: Recall value (float)
        """
        if not self.func_help():
            return None
        if self.n_predicted == 1:
            # all clustered in single cluster. Therefore, no FN
            return 1
        if self.fn == 0:
            if not self.tp + self.fp > 0:
                print("WARNING: recall of 1 returned but no true predictions.")
                print(self)
            # special case: when labels were all marked positive
            return 1

        return self.tp / (self.tp + self.fn)

    def accuracy(self):
        """
        Accuracy (Acc): Overall performance of model

        Acc = (TP + TN) / (TP + FP + FN + TN)
        """
        if not self.func_help():
            return None
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    def specificity(self):
        """
        Recall = TN / (TN + FP)
        """
        if not self.func_help():
            return None
        return self.tn / (self.tn + self.fp)

    def f1score(self):
        """
        Recall = 2TP / (2TP + FP + FN)
        """

        if not self.func_help():
            return None
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    @staticmethod
    def calculate_tp(true_ids, cluster_ids):
        """
        Calculate the number of TP for a set of cluster assignments.
        Parameters
        ----------
        true_ids:     Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.
        cluster_ids:

        Returns
        ----------
        Number of true positives.
        """
        # calibrate labels such to start from 0,.., M, where M is # of unique labels
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        tp = 0  # TP (running sum)
        for i, cluster_id in enumerate(np.unique(cluster_ids)):
            # for each cluster, determine contents wrt to true labels
            cluster = true_ids[cluster_ids == cluster_id]
            # how many of each label type
            unique, counts = np.unique(cluster, return_counts=True)
            # count pairs for bins with more than 1 sample (i.e., 1 sample = 0 pairs, 0!)
            tp += sum(nchoosek(c, 2) for c in counts if c > 1)
        return int(tp)

    @staticmethod
    def calculate_fp(true_ids, cluster_ids):
        """
        Calculate the number of FP for a set of cluster assignments.
        Parameters
        ----------
        true_ids:     Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.
        cluster_ids:

        Returns
        ----------
        Number of false positives.
        """

        # calibrate labels such to start from 0,.., M, where M is # of unique labels
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        fp = 0  # FP (running sum)
        for i, cluster_id in enumerate(np.unique(cluster_ids)):
            # for each cluster, determine contents wrt to true labels
            cluster = true_ids[cluster_ids == cluster_id]
            # how many of each label type
            unique, counts = np.unique(cluster, return_counts=True)
            pairs = list(combinations(unique, 2))
            lut = dict(zip(unique, counts))
            # sum of products from each count for each class
            fp += sum(lut[pair[0]] * lut[pair[1]] for pair in pairs)
        return int(fp)

    @staticmethod
    def calculate_fn(true_ids, cluster_ids):
        """
        Calculate the number of FN for a set of cluster assignments.

        Parameters
        ----------
        true_ids:     Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.
        cluster_ids:

        Returns
        ----------
        Number of false negatives.
        """

        # calibrate labels such to start from 0,.., M, where M is # of unique labels
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        fn = 0  # FN (running sum)
        for i, cluster_id in enumerate(np.sort(np.unique(cluster_ids)[:-1])):
            # for each cluster, determine contents wrt to true labels
            cluster = true_ids[cluster_ids == cluster_id]
            # only look at larger values to not count same pair 2x (i.e., not look back, as prior already was calculated.
            another = true_ids[cluster_ids > cluster_id]
            # how many of each label in current cluster
            unique, counts = np.unique(cluster, return_counts=True)
            # how many of each in other clusters
            other_unique, other_counts = np.unique(another, return_counts=True)

            # make dictionaries and determine common keys to count
            lut = dict(zip(unique, counts))
            other_lut = dict(zip(other_unique, other_counts))

            common = list(set(lut.keys()).intersection(set(other_lut.keys())))

            for key in common:
                # number of elements in current * number outside
                fn += lut[key] * other_lut[key]

        return int(fn)

    def calculate_negative_rates(self):
        """
        Calculate FMR and FNMR. It is assumed true and predicted labels have already been fit.
        :return:
        """

        total_negative = self.tn + self.fn
        fm_rate = self.tn / total_negative
        fnm_rate = self.fn / total_negative

        return fnm_rate, fm_rate

    def confusion_matrix_values(self, true_ids, cluster_ids):
        """
        Calculate TP, FP, TN, and FN and store in dictionary container.
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids:     Cluster assignment [ Nx1 ].
        :return: Confusion stats {TP, FP, TN, FN} (dictionary)
        """
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        stats = {}
        stats["TP"] = self.calculate_tp(true_ids, cluster_ids)
        stats["FP"] = self.calculate_fp(true_ids, cluster_ids)
        stats["FN"] = self.calculate_fn(true_ids, cluster_ids)
        npairs = nchoosek(len(true_ids), 2)  # total number of pairs
        npositive = stats["FP"] + stats["TP"]  # total number of positive pairs
        nnegative = npairs - npositive  # total number of negative pairs
        stats["TN"] = int(nnegative - stats["FN"])

        return stats

    def calculate_precision(self, true_ids, cluster_ids):
        """
        Calculate precision of the ith cluster w.r.t. assigned clusterins. True labels are used to determine those from same
        class and, hence, should be clustered together. It is assumed all N samples are clustered.

        Precision (P): How accurate are the positive predictions.

        Precision = TP / (TP + FP) (per class)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids:     Cluster assignment [ Nx1 ].
        :return: Precision value (float)
        """

        stats = self.confusion_matrix_values(true_ids, cluster_ids)
        return stats["TP"] / (stats["TP"] + stats["FP"])

    def calculate_recall(self, true_ids, cluster_ids):
        """
        Calculate recall of the ith cluster w.r.t. clabels. Ground-truth is used to determine the observations from the same
        class (identity) and, hence, should be clustered together.

        Recall (R): Coverage of actual positive sample.

        R = TP / (TP + FN)

        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return: Recall value (float)
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return stats["TP"] / (stats["TP"] + stats["FN"])

    def calculate_accuracy(self, true_ids, cluster_ids):
        """
        Calculate accuracy.

        Accuracy (Acc): Overall performance of model

        Acc = (TP + TN) / (TP + FP + FN + TN)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return:
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return (stats["TP"] + stats["TN"]) / (
            stats["TP"] + stats["FP"] + stats["FN"] + stats["TN"]
        )

    def calculate_specificity(self, true_ids, cluster_ids):
        """
        Calculate specificity: Coverage of actual negative sample.

        Recall = TN / (TN + FP)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return:
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return stats["TN"] / (stats["TN"] + stats["FP"])

    def calculate_f1score(self, true_ids, cluster_ids):
        """
        Calculate F1-score: Hybrid metric useful for unbalanced classes.

        Recall = 2TP / (2TP + FP + FN)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return:
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return 2 * stats["TP"] / (2 * stats["TP"] + stats["FP"] + stats["FN"])


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
