import numpy as np
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score

def build_stims_Kim2020neuro(params):
    # initiate all unique conditions
    conditions = [(cat, rep_cat, oper) for cat, rep_cat in permutations(params["categories"], 2)
                                     for oper in params["operations"]]
    df = pd.DataFrame(conditions, columns = ["category", "replace_category", "operation"])
    num_main_trials = params["num_main_trials"]
    df = pd.concat([df] * (num_main_trials // len(conditions)))
    valid_seq = False
    while not valid_seq:
        df = df.sample(frac=1, ignore_index=True)
        valid_seq = not any((df.operation == df.operation.shift(1)) &
                            (df.operation == df.operation.shift(2)) &
                            (df.operation == df.operation.shift(3)))
    # initiate the stims dict
    assert params["num_loc_items"] % params["num_categories"] == 0, "num loc items are not fully divisible by num categories"
    num_items_per_cat = params["num_loc_items"] // params["num_categories"]
    num_cat_repeats = num_main_trials // params["num_loc_items"] * 2
    stims_dict = {cat: [i for n in range(num_cat_repeats) for i in np.random.permutation(np.arange(num_items_per_cat))] for cat in params["categories"] }
    stims_list = []
    replace_list = []
    for _, r in df.iterrows():
        stims_list.append(stims_dict[r.category].pop())
        replace_list.append(stims_dict[r.replace_category].pop())
    df["stim"] = stims_list
    df["replace_stim"] = replace_list
    return df


def decode_category(exp, data=None):
    """
    Decodes the category information for each trial for each timepoint using innate clf in exp
    :param exp: Experiment object
    :param data: default=None. Data in shape (n_trials, n_timepoints, n_layers, n_features)
    :return:
    """
    if data is None:
        data = np.asarray(exp.trials_data)
    n_trials = data.shape[0]
    n_timepoints = data.shape[1]
    n_classes = exp.params["num_categories"]

    clf = exp.clf
    probas = np.zeros((n_trials,
                       n_timepoints,
                       n_classes,
                       ))
    decfunc = np.zeros((n_trials,
                        n_timepoints,
                        n_classes))
    for trial in range(n_trials):
        probas[trial] = clf.predict_proba(data[trial, :, exp.get_clf_layer_inds()].reshape((n_timepoints, -1)))
        decfunc[trial] = clf.decision_function(data[trial, :, exp.get_clf_layer_inds()].reshape((n_timepoints, -1)))
    return {"proba": probas, "decfunc": decfunc}



def summarize_cat_decoding(exp,
                           results_dict,
                           df,
                           summary_value="evidence",
                           graph=False):
    """

    :param exp:
    :param results_dict:
    :param df:
    :param graph_value:
    :return:
    """
    correct_label = exp.label_encoder.transform(df.category)
    if summary_value == "evidence":
        values = 1 / (1 + np.exp(-results_dict["decfunc"]))
    elif summary_value == "proba" or summary_value == "rocauc":
        values = results_dict["proba"]
    else:
        raise ValueError(f"Graph value must be in ['proba', 'decfunc', 'rocauc']")
    # initiate the operations
    operations = exp.params["operations"].copy()
    if "replace" in operations:
        operations.remove("replace")
        operations += ["replace_old", "replace_new"]
    # initiate results array
    results_arr = np.zeros((values.shape[1],
                            len(operations),
                            ))
    # also initiate the figure
    if graph:
        fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"maintain": "forestgreen",
              "suppress": "firebrick",
              "replace_old": "darkblue",
              "replace_new": "cornflowerblue"}
    # graph each operation
    for o, oper in enumerate(operations):
        oper_inds = df.operation == (oper if oper not in ["replace_old", "replace_new"] else "replace")
        corr_label_oper = correct_label[oper_inds] if oper != "replace_new" else exp.label_encoder.transform(df.replace_category)[oper_inds]
        if summary_value == "rocauc":
            for tp in range(values.shape[1]):
                results_arr[tp, o] = roc_auc_score(y_true=corr_label_oper,
                                                   y_score=values[oper_inds, tp],
                                                   multi_class="ovr")
        else:
            results_arr[:, o] = values[oper_inds, :, corr_label_oper].mean(axis=0)
        # plot
        if graph:
            ax.plot(np.arange(values.shape[1]), results_arr[:, o], label=oper, color=colors[oper])
    #
    if graph:
        # plt.title(f"{graph_value} plot ")
        plt.xlabel(f"timepoint", fontsize=12)
        plt.ylabel(f"{summary_value}", fontsize=12)
        plt.axvline(exp.params["timesteps_per_phase"] - 1, linestyle="--", color="k")
        plt.legend()
        plt.show()
    return results_arr

def item_RSA():
    pass



