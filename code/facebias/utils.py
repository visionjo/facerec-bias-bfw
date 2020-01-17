import sys


def add_package_path(path_package=f"../"):
    if path_package not in sys.path:
        sys.path.append(path_package)


def eval_acc(threshold, predicts):
    """    """

    y_predict = np.array([1 if float(p[2]) > threshold else 0 for p in predicts])
    y_true = np.array([int(p[3]) for p in predicts])

    return (y_true == y_predict).mean()


def find_best_threshold(thresholds, predicts):
    """
    Determine best threshold as the largest threshold that yields top accuracy. Note, tie goes to larger threshold.
    :param thresholds:  threshold values to calculate accuracy with respect to.
    :param predicts:    predictions [p1, p2, score, label], where score and label (index 2 and 3) are used.
    :return:            Threshold value that yielded best accuracy (same type as threshold[threshold.argmax()]).
    """
    best_threshold = best_acc = 0
    for threshold in thresholds:

        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold
