import numpy as np
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns

from scipy.stats import sem
from mne.stats import permutation_cluster_1samp_test

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score

from modrem_utils import Modrem_Exp, layers_dict


def build_stims_Kim2020neuro(params):
    # initiate all unique conditions
    conditions = [(cat, rep_cat, oper) for cat, rep_cat in permutations(params["categories"], 2)
                  for oper in params["operations"]]
    df = pd.DataFrame(conditions, columns=["category", "replace_category", "operation"])
    num_main_trials = params["num_main_trials"]
    df = pd.concat([df] * (num_main_trials // len(conditions)))
    valid_seq = False
    while not valid_seq:
        df = df.sample(frac=1, ignore_index=True)
        valid_seq = not any((df.operation == df.operation.shift(1)) &
                            (df.operation == df.operation.shift(2)) &
                            (df.operation == df.operation.shift(3)))
    # initiate the stims dict
    assert params["num_loc_items"] % params[
        "num_categories"] == 0, "num loc items are not fully divisible by num categories"
    num_items_per_cat = params["num_loc_items"] // params["num_categories"]
    num_cat_repeats = num_main_trials // params["num_loc_items"] * 2
    stims_dict = {cat: [i for n in range(num_cat_repeats) for i in np.random.permutation(np.arange(num_items_per_cat))]
                  for cat in params["categories"]}
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
    oper_list = []
    # graph each operation
    for o, oper in enumerate(operations):
        oper_list.append(oper)
        oper_inds = df.operation == (oper if oper not in ["replace_old", "replace_new"] else "replace")
        corr_label_oper = correct_label[oper_inds] if oper != "replace_new" else \
            exp.label_encoder.transform(df.replace_category)[oper_inds]
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
    return {"results": results_arr, "operations": oper_list}


def item_RSA(exp, ):
    pass


def simulate_participant(params,
                         diagnostic=False,
                         **kwargs):
    """

    :param params:
    :param diagnostic:
    :param kwargs:
    :return: Exp object
             trials_list: list, with dimensions (n_trials, n_timepoints, n_layers, n_features)
                        --The neural activity for each trial

    """
    # Initiate the experiment object
    Exp = Modrem_Exp(params)
    #  Initiate localizer memories
    loc_memories = Exp.create_loc_memories()
    # train classifier
    clf = Exp.classifier_train()
    # Set up stim list
    Exp.stim_df = build_stims_Kim2020neuro(params)
    # Simulate the entire experiment
    if not diagnostic:
        trials_list = Exp.simulate_experiment()
        return Exp
    else:
        trials_list = []
        for _, row in Exp.stim_df.iterrows():
            encode_item = "_".join([row.category, str(row.stim)])
            replace_item = "_".join([row.replace_category, str(row.replace_stim)])
            # Initialize the trial
            Exp.initialize_trial(item=encode_item)

            # run the trial
            trials_list.append(Exp.simulate_trial(operation=row.operation,
                                                  encode_item=encode_item,
                                                  replace_item=replace_item,
                                                  diagnostic=diagnostic))
        return trials_list


def simulate_full_experiment(params,
                             n_participants=30,
                             n_jobs=1,
                             **kwargs):
    # exp_list = []
    # for n in tqdm(range(n_participants)):
    #     Exp = simulate_participant(params=params,
    #                                summary_value="evidence",
    #                                **kwargs
    #                                )
    #     exp_list.append(Exp)
    exp_list = Parallel(n_jobs=n_jobs)(
        delayed(simulate_participant)(params=params,
                                      summary_value="evidence",
                                      **kwargs) for n in tqdm(range(n_participants))
    )
    return exp_list


def timecourse_cat_decoding(exp_list,
                            params):
    """
    Kim et al. (2020) Graph 4a: Timecourse for neural decoding of a WM item
    --Also outputs
    :param exp_list:
    :return: results_arr (array)
    """
    results = []
    opers_list = []
    for e, exp in enumerate(exp_list):
        results_dict = decode_category(exp=exp,
                                       data=None)

        res_dict = summarize_cat_decoding(exp=exp,
                                          results_dict=results_dict,
                                          df=exp.stim_df,
                                          summary_value="evidence",
                                          graph=False)

        results.append(res_dict["results"])
        opers_list.append(res_dict["operations"])

    ####
    # convert results list to array
    results_arr = np.asarray(results)
    # Take operations order from opers_list
    operations = np.unique(opers_list, axis=0).squeeze()
    if operations.ndim > 1:
        raise ValueError("There are more than one unique operations order:\n", operations)
    # Now graph it
    colors = {"maintain": "forestgreen",
              "suppress": "firebrick",
              "replace_new": "cornflowerblue",
              "replace_old": "darkblue",
              "enconly": "darkgray",
              "noise": "black",
              }
    fig, ax = plt.subplots()
    x = np.arange(1, results_arr.shape[1] + 1)
    y_mean = results_arr.mean(axis=0)
    y_se = sem(results_arr, axis=0)
    for o, oper in enumerate(operations):
        plt.plot(x, y_mean[:, o], label=oper, color=colors[oper])
        plt.fill_between(x, y_mean[:, o] + y_se[:, o], y_mean[:, o] - y_se[:, o], color=colors[oper], alpha=0.2)
    plt.legend(loc="upper right")
    plt.axvline(params["timesteps_per_phase"] + 2, color="k", linestyle="--")
    plt.axvline(int(params["timesteps_per_phase"] * 2) + 2, color="k", linestyle="--", label="fixation_start")
    pattern = "|".join(map(re.escape, ["ual", "bal", "{", "}", ":", "'", " "]))
    echweights = re.sub(pattern, "", str(params["echo_weights"]))
    #  echoWeights={echweights}
    plt.title(
        f"t={params['tau']}, tStyle={params["tau_style"]}, pt={params["post_tau"]}, ptStyle={params["post_tau_style"]}, ic={params['ic_ratio']},em={params['em_ratio']},b={params['beta']}")
    plt.show()
    return results_arr


def permtest_1d(X,
                tail=0,
                p_thresh=0.05):
    tobs, clusters, clusters_pv, _ = permutation_cluster_1samp_test(X,
                                                                    tail=tail, )
    # recreate the original array
    pvals = np.ones(X.shape[1])
    if len(clusters) == 0:
        return pvals, []
    for c, clus in enumerate(clusters):
        pvals[clus] = clusters_pv[c]
    sigs = np.where(pvals < p_thresh)[0]
    return pvals, sigs


def graph_operDiff_catDecode(exp_list,
                             params,
                             ylim=None,
                             ):
    """
    Kim et al. (2020) Graph 4b[i]: Trajectory for removal of an item [category]
    :param exp_list:
    :param params:
    :return: None
    """
    results = []
    opers_list = []
    for e, exp in enumerate(exp_list):
        results_dict = decode_category(exp=exp,
                                       data=None)

        res_dict = summarize_cat_decoding(exp=exp,
                                          results_dict=results_dict,
                                          df=exp.stim_df,
                                          summary_value="evidence",
                                          graph=False)

        results.append(res_dict["results"])
        opers_list.append(res_dict["operations"])

    # Save the ind numbers
    main_ind = np.unique(opers_list, axis=0).squeeze().tolist().index("maintain")
    supp_ind = np.unique(opers_list, axis=0).squeeze().tolist().index("suppress")
    rep_ind = np.unique(opers_list, axis=0).squeeze().tolist().index("replace_old")
    # Subtract each trace from maintain
    results_arr = np.asarray(results)
    # Perform subtractions
    diff_dict = {"replace": results_arr[..., rep_ind] - results_arr[..., main_ind],
                 "suppress": results_arr[..., supp_ind] - results_arr[..., main_ind],
                 }
    colors_dict = {"replace": "darkblue",
                   "suppress": "firebrick", }
    # plot these traces
    fig, ax = plt.subplots(figsize=(10, 4))
    ylim = ylim or (-0.5, 0.13)
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.axhspan(ylim[0], 0, alpha=0.15, color="gray")
    for i, (oper, res) in enumerate(diff_dict.items()):
        # Run permutation testing for these traces (test against 0)
        pvals, sigs = permtest_1d(res, tail=-1)
        # Get mean and sem
        means = res.mean(axis=0)
        sems = sem(res, axis=0)
        ax.fill_between(np.arange(res.shape[1]), means + sems, means - sems, color=colors_dict[oper], label=oper)
        # plot the sigs
        ax.plot(sigs, [0.07 + 0.02 * i] * len(sigs), color=colors_dict[oper], linewidth=4)
        # for s in sigs:
        #     plt.text(s, 0.09 + 0.01 * i , "*", )
    plt.axvline(params["timesteps_per_phase"] + 1, color="k", alpha=0.5, linestyle="--")
    plt.axvline(params["timesteps_per_phase"] * 2 + 1, color="k", alpha=0.5, linestyle="--")
    plt.title("Trajectory for removal of an item [category]")
    plt.ylabel("classifier evidence\n(removal - maintain)")
    plt.show()
    return None


def calculate_item_RSA(exp,
                       similarity_coefficient="pearson",
                       fisher=True,
                       layers=None,
                       **kwargs):
    # Obtain the encoded item for each trial
    item_reps_list = []
    for _, r in exp.stim_df.iterrows():
        item_reps_list.append(
            exp.representations.query_item_representation(f"{r.category}_{r.stim}")
        )
    item_reps_arr = np.array(item_reps_list)
    # Obtain trials data
    trials_data_arr = np.asarray(exp.trials_data)
    # Convert layers to list if it is a str
    if type(layers) is str:
        layers = [layers]
    # Subselect layers if necessary
    if layers:
        # find the indices
        layer_inds = [layers_dict[l] for l in layers]
        # then subselect
        item_reps_arr = item_reps_arr[:, layer_inds]
        trials_data_arr = trials_data_arr[:, :, layer_inds]
    if similarity_coefficient == "pearson":
        # reshape to flatten across layers
        item_reps_arr = item_reps_arr.reshape(item_reps_arr.shape[0], -1)
        trials_data_arr = trials_data_arr.reshape(trials_data_arr.shape[0], trials_data_arr.shape[1], -1)
        # mean-center
        item_reps_arr = item_reps_arr - item_reps_arr.mean(axis=-1, keepdims=True)
        trials_data_arr = trials_data_arr - trials_data_arr.mean(axis=-1, keepdims=True)
        # Normalize each vector
        item_reps_arr = item_reps_arr / np.linalg.norm(item_reps_arr, axis=-1, keepdims=True)
        trials_data_arr = trials_data_arr / np.linalg.norm(trials_data_arr, axis=-1, keepdims=True)
        # perform correlation
        corr = np.einsum("it,ijt->ij", item_reps_arr, trials_data_arr)
    elif similarity_coefficient == "cosine":
        # Dot the two arrays to obtain representational similarity for each layer
        layered_similarity = np.einsum("ijab,iab->ija", trials_data_arr, item_reps_arr)
        # Then average across layers
        corr = layered_similarity.mean(axis=-1)
    if fisher:
        corr = np.arctanh(corr)
    return corr


def graph_operDiff_itemRSA(exp_list,
                           params,
                           ylim=None,
                           **kwargs):
    """
    Graph 4b[i]: Trajectory for removal of an item [category]
    :param ylim:
    :param exp_list:
    :param params:
    :param kwargs: **Passed to calculate_item_RSA: fisher default=True (perform fisher Z transform on output),
                                                   layers default: list or str = None (layers to calculate RSA)

    :return:
    """

    results_dict = {op: [] for op in params["operations"]}

    # Loop through each list
    for e, exp in enumerate(exp_list):
        # calculate representational similarity
        corr = calculate_item_RSA(exp,
                                  similarity_coefficient="pearson",
                                  **kwargs)
        # Index and average across trials within operation
        for oper in params["operations"]:
            # subselect the trial indices
            oper_inds = np.where(exp.stim_df.operation == oper)[0]
            results_dict[oper].append(corr[oper_inds].mean(axis=0))
    ## Now graph each individual line and run stats
    # Declare the colors for graphing
    colors_dict = {"maintain": "forestgreen",
                   "replace": "darkblue",
                   "suppress": "firebrick", }
    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ylim = ylim or (-0.065, 0.035)
    ax.set_ylim(ylim)
    ax.axhspan(ylim[0], 0, alpha=0.15, color="gray")
    for i, (oper, results) in enumerate(results_dict.items()):
        if oper == "maintain":
            continue

        results = np.array(results)
        # subtract maintain from vals
        results -= np.asarray(results_dict["maintain"])
        # Test significance
        pvals, sigs = permtest_1d(results, tail=-1)
        # find the mean and standard error
        res_mean = results.mean(axis=0)
        res_sem = sem(results, axis=0)
        # plot
        ax.fill_between(np.arange(results.shape[-1]),
                        res_mean - res_sem,
                        res_mean + res_sem,
                        label=oper,
                        color=colors_dict[oper])
        # plot the significance
        ax.plot(sigs, [0.024 + 0.003 * i] * len(sigs), color=colors_dict[oper], linewidth=4)
    plt.axvline(params["timesteps_per_phase"] + 1, color="k", alpha=0.5, linestyle="--")
    plt.axvline(params["timesteps_per_phase"] * 2 + 1, color="k", alpha=0.5, linestyle="--")
    plt.ylabel("RSA\n(removal - maintain)")
    plt.title("Trajectory for removal of an item [item]")
    plt.show()
    return None


def graph_proactive_interference(exp_list,
                                 params,
                                 plot_delta=True,
                                 **kwargs):
    """
    Graph 4b[i]: Trajectory for removal of an item [category]
    :param plot_delta:
    :param exp_list:
    :param params:
    :param kwargs: **Passed to calculate_item_RSA: fisher default=True (perform fisher Z transform on output),
                                                   layers default: list or str = None (layers to calculate RSA)
    :return:
    """

    results_df = pd.DataFrame()

    # Loop through each list
    for e, exp in enumerate(exp_list):
        # calculate representational similarity
        corr = calculate_item_RSA(exp,
                                  similarity_coefficient="pearson",
                                  **kwargs)
        # Find the trials where the next trial is same/diff category
        cats = exp.stim_df["category"]
        curr_cats = cats.iloc[:-1].to_numpy()
        next_cats = cats.iloc[1:].to_numpy()
        same_inds = np.where(curr_cats == next_cats)[0]
        diff_inds = np.where(curr_cats != next_cats)[0]
        # Index and average across trials within operation
        for oper in params["operations"]:
            # subselect the trial indices
            oper_inds = np.where(exp.stim_df.operation == oper)[0]
            same_oper_inds = np.intersect1d(same_inds, oper_inds) + 1
            diff_oper_inds = np.intersect1d(diff_inds, oper_inds) + 1
            # calculate RSA value for last timepoint in the encoding period
            t_df = pd.DataFrame(
                {
                    "participant": e,
                    "operation": oper,
                    "same": corr[same_oper_inds, params["timesteps_per_phase"] + 1].mean(),
                    "diff": corr[diff_oper_inds, params["timesteps_per_phase"] + 1].mean(),
                },
                index=[0]
            )
            results_df = pd.concat([results_df, t_df], ignore_index=True)
    # calculate the delta values
    results_df["delta"] = results_df["same"] - results_df["diff"]
    # Declare the colors for graphing
    colors_dict = {"maintain": "forestgreen",
                   "replace": "darkblue",
                   "suppress": "firebrick", }

    results_long = results_df.melt(
        id_vars=["participant", "operation"],
        value_vars=["same", "diff", "delta"],
        var_name="n+1_condition",
        value_name="value"
    )

    if plot_delta:
        sns.barplot(data=results_df,
                    x="operation",
                    y="delta",
                    hue="operation",
                    errorbar="se",
                    palette=colors_dict,
                    )
        plt.ylabel("Encoding fidelity (same - different)")
        # plt.ylim(-0.25, 0.25)
    else:
        sns.barplot(data=results_long,
                    x="operation",
                    y="value",
                    hue="n+1_condition",
                    errorbar="se",
                    )
    plt.show()
    return None

    #
    # exp_list = simulate_full_experiment(params=mem_params,
    #                                     n_participants=20,
    #                                     n_jobs=10)
    #
    # # Graph 4a: Timecourse for neural decoding of a WM item
    # results_arr = timecourse_cat_decoding(exp_list=exp_list,
    #                                       params=mem_params, )
    # # Graph 4b[i]: Trajectory for removal of an item from WM (Category)
    # graph_operDiff_catDecode(exp_list=exp_list,
    #                          params=mem_params, )
    # # Graph 4b[ii]: Trajectory for removal of an item from WM (Item)
    # graph_operDiff_itemRSA(exp_list=exp_list,
    #                        params=mem_params,
    #                        fisher=True,
    #                        layers=["visual","verbal"])
