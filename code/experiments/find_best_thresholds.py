import pandas as pd
import numpy as np
from bias.Feature import BiasFeatures
from itertools import combinations
from utils.io import sys_home
from bias.configs import CONFIGS

# imdir = '../../data/bias-set-aligned/'
# imdir=sys_home() + '/bias-set-current/'
# f_feats = sys_home() + '/bias-set-current/features/features.pkl'


df_pairs = pd.read_pickle(CONFIGS.path.data_table)
# bias_features = BiasFeatures(CONFIGS.path.features)


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


thresholds = np.linspace(0.22, 0.6, 1000)
# attributes = bias_features.attributes
paired_atts = list(combinations(CONFIGS.settings.abbr_attribute, 2))
for att in CONFIGS.settings.abbr_attribute:
    # append pairs of same type
    paired_atts.append((att, att))
paired_atts = np.unique([sorted(p) for p in paired_atts], axis=1)
att_lut = {}

for i in range(len(paired_atts)):
    pair = paired_atts[i]
    att_lut[tuple(pair)] = i

# keys = list(att_lut.keys())
# for k in keys:
#     att_lut[tuple(np.flip(k))] = att_lut[k]

# df_pairs['type1'] = df_pairs.att1.apply(lambda x: CONFIGS.settings.attributes[x]).astype(str)
# df_pairs['type2'] = df_pairs.att2.apply(lambda x: CONFIGS.settings.attributes[x]).astype(str)

# df_pairs['type_pair'] = df_pairs[['type1', 'type2']].apply(lambda x: x, axis=1)


best_thresholds = {}
baseline_scores = {}

best_scores = {}
npairs = len(df_pairs)

# for att1 in attributes:
#     for att2 in attributes:
print("## All Subgroups ##")
thr = 0.40
df_pairs["optimal_threshold"] = None
for atts in paired_atts:
    ids = ((df_pairs.type1 == atts[0]) & (df_pairs.type2 == atts[1])) | (
        (df_pairs.type1 == atts[1]) & (df_pairs.type2 == atts[0])
    )
    df = df_pairs.loc[ids]

    # bs, bt = f oimn
    avg_scores = [((df.scores > th) == df.label).mean() for th in thresholds]

    max_ids = np.argmax(avg_scores)
    best_scores[tuple(atts)] = avg_scores[max_ids]
    best_thresholds[tuple(atts)] = thresholds[max_ids]
    df_pairs.loc[ids, "optimal_threshold"] = thresholds[max_ids]

    print(
        "{0}-{1}: {2:.4f} with threshold of {3:.4f}".format(
            atts[0], atts[1], avg_scores[max_ids], thresholds[max_ids]
        )
    )
    y_predict = df.scores > thr
    avg = (y_predict == df.label).mean()
    print("{0:.4f} with threshold of {1:.4f}".format(avg, thr))
    print()
    baseline_scores[tuple(atts)] = avg

df_pairs["gender_threshold"] = None
print("")
print("")
print("## Gender-Based ##")
for atts in [
    [CONFIGS.settings.abbr_gender[0]] * 2,
    [CONFIGS.settings.abbr_gender[1]] * 2,
    CONFIGS.settings.abbr_gender,
]:
    ids = ((df_pairs.gender1 == atts[0]) & (df_pairs.gender2 == atts[1])) | (
        (df_pairs.gender1 == atts[1]) & (df_pairs.gender2 == atts[0])
    )

    df = df_pairs.loc[ids]
    avg_scores = [((df.scores > th) == df.label).mean() for th in thresholds]
    max_ids = np.argmax(avg_scores)
    best_scores[tuple(atts)] = avg_scores[max_ids]
    best_thresholds[tuple(atts)] = thresholds[max_ids]
    df_pairs.loc[ids, "gender_threshold"] = thresholds[max_ids]
    print(
        "{0}-{1}: {2:.4f} with threshold of {3:.4f}".format(
            atts[0], atts[1], avg_scores[max_ids], thresholds[max_ids]
        )
    )
    y_predict = df.scores > thr
    avg = (y_predict == df.label).mean()
    print("{0:.4f} with threshold of {1:.4f}".format(avg, thr))
    print()
    baseline_scores[tuple(atts)] = avg
    # print(atts, thresholds[max_ids], avg_scores[max_ids])

print("## Ethnicity-Based ##")
df_pairs["ethnicity_threshold"] = None
for atts in np.unique(
    [sorted(s) for s in list(combinations(CONFIGS.settings.abbr_ethnicity * 2, 2))],
    axis=1,
):
    ids = ((df_pairs.ethnicity1 == atts[0]) & (df_pairs.ethnicity2 == atts[1])) | (
        (df_pairs.ethnicity1 == atts[1]) & (df_pairs.ethnicity2 == atts[0])
    )

    df = df_pairs.loc[ids]
    avg_scores = [((df.scores > th) == df.label).mean() for th in thresholds]
    max_ids = np.argmax(avg_scores)
    best_scores[tuple(atts)] = avg_scores[max_ids]
    best_thresholds[tuple(atts)] = thresholds[max_ids]
    df_pairs.loc[ids, "ethnicity_threshold"] = thresholds[max_ids]
    print(
        "{0}-{1}: {2:.4f} with threshold of {3:.4f}".format(
            atts[0], atts[1], avg_scores[max_ids], thresholds[max_ids]
        )
    )
    y_predict = df.scores > thr
    avg = (y_predict == df.label).mean()
    print("{0:.4f} with threshold of {1:.4f}".format(avg, thr))
    print()
    baseline_scores[tuple(atts)] = avg

meta = np.array(
    [
        (k, th, s, sc)
        for th, (k, s), sc in zip(
            best_thresholds.values(), best_scores.items(), baseline_scores.values()
        )
    ]
).reshape(-1, 4)
print(meta)
pd.to_pickle(meta, CONFIGS.path.dlists + "meta.pkl")
df_pairs.to_pickle(CONFIGS.path.data_table)

for (k, th, s, sc) in meta:
    print("{0}: {2:.4f} (th={1:.5f})\t {3:.5f} (baseline)".format(k, th, s, sc))

# for m in meta:
#     print("{0}:{1}\t{2:.4f}\t{3:.4f}".format(m))


# print(best_thresholds)
