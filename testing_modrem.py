import numpy as np
from modrem_utils import *
import matplotlib.pyplot as plt
import KimEtal2020_sim as KE
from scipy.stats import sem
import re


mem_params = {  # Experiment design
    "num_loc_items": 54,
    "num_categories": 3,
    "categories": ["face", "scene", "fruit"],
    "operations": ["maintain", "replace", "suppress", ], # "noise"
    "num_loc_repeats": 5,
    "num_main_trials": 270,
    "timesteps_per_phase": 10,
    "trial_reset": True,
    # Model design
    "vec_len": 10,
    "loc_layers": ["visual", "verbal"],
    # "main_layers": ["visual", "verbal"],
    "clf_layers": ["visual"],
    "ic_ratio": 1,  # item vs category ratio
    "em_ratio": 1,  # external vs memory ratio
    "beta": 0.65,
    "tau_style": "linear",
    "tau": 16,
    "post_tau_style": "linear",  # ["exp", "power", "linear"]
    "post_tau": 1,
    "mem_source": "combined",
    "snr": 5,  # signal to noise ratio (not implemented)
    "echo_weights": {
        "visual": 1,
        "verbal": 1,
    },
    "update_rules": {
        # "encode": {
        # "external": {"visual": "representation",
        #              "verbal": "noise"},
        # "memory": {"echo_layers": ["visual", "verbal"],
        #            "noise_layers": [],
        #            "tau_dilation": 1},
        # },
        "replace": {
            "external": {"visual": "noise",
                         "verbal": "representation"},
            "memory": {"echo_layers": ["visual", "verbal"],
                       "noise_layers": [],
                       "tau_dilation": 1},
        },
        "maintain": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual",],
                       "noise_layers": [],
                       "tau_dilation": 1},
        },
        "suppress": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual", "verbal"],
                       "noise_layers": [],
                       "tau_dilation": 0.5}
        }
    },
    "init_state": "noise",
    "activation_intensity": False,
}



def simulate_participant(mem_params, summary_value="evidence", diagnostic=False, **kwargs):
    # Initiate the experiment object
    Exp = Modrem_Exp(mem_params)
    #  Initiate localizer memories
    loc_memories = Exp.create_loc_memories()
    # train classifier
    clf = Exp.classifier_train()

    # Set up experiment
    df = KE.build_stims_Kim2020neuro(mem_params)
    _, row = next(df.iterrows())
    trials_list = []
    for _, row in df.iterrows():
        encode_item = "_".join([row.category, str(row.stim)])
        replace_item = "_".join([row.replace_category, str(row.replace_stim)])
        # Initialize the trial
        Exp.initialize_trial(item=encode_item)

        # run the trial
        trials_list.append(Exp.simulate_trial(operation=row.operation,
                                              encode_item=encode_item,
                                              replace_item=replace_item,
                                              diagnostic=diagnostic))

    results_dict = KE.decode_category(exp=Exp,
                                      data=None)

    results_arr = KE.summarize_cat_decoding(exp=Exp,
                                            results_dict=results_dict,
                                            df=df,
                                            summary_value=summary_value,
                                            graph=False)
    return results_arr


def simulate_full_experiment(mem_params, n_participants=30, **kwargs):
    results = []
    for n in range(n_participants):
        results.append(simulate_participant(mem_params=mem_params,
                                            summary_value="evidence",
                                            **kwargs))
    # convert results list to array
    results_arr = np.asarray(results)
    # Convert replace
    operations = mem_params["operations"].copy()
    if "replace" in operations:
        operations.remove("replace")
        operations += ["replace_old", "replace_new"]
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
    plt.axvline(mem_params["timesteps_per_phase"] + 2, color="k", linestyle="--")
    pattern = "|".join(map(re.escape, ["ual", "bal", "{", "}", ":", "'", " "]))
    echweights = re.sub(pattern, "", str(mem_params["echo_weights"]))
    plt.title(f"t={mem_params['post_tau']}, tStyle={mem_params["post_tau_style"]}, echoWeights={echweights}, ic={mem_params['ic_ratio']},em={mem_params['em_ratio']},b={mem_params['beta']}")
    plt.show()
    return {"results": results_arr, "params": mem_params}




####################################################################################################################
#%%
# Initiate the experiment object
Exp = Modrem_Exp(mem_params)
#  Initiate localizer memories
loc_memories = Exp.create_loc_memories()
# train classifier
clf = Exp.classifier_train()

# Simulate 10 steps of encoding
Exp.reset_task_memories()
Exp.reset_current_trial()
Exp.initialize_trial(category="scene")
Exp.initialize_replacement(category="fruit")
current_trial = []
vis_sim = []
ver_sim = []
comb_sim = []
# Start off with some noise timesteps
for n in range(2):
    Exp.simulate_step(phase="noise",)
num_enc = 10
encode_phase = "encode"
for n in range(num_enc):
    new_state = Exp.simulate_step(phase=encode_phase,
                                  diagnostic=True,)
    vis_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["visual"],
                                                  diagnostic=True)[0])
    ver_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["verbal"],
                                                  diagnostic=True)[0])
    comb_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["visual", "verbal"],
                                                  diagnostic=True)[0])
    current_step = Exp.simulate_step(phase=encode_phase,)


    # print(Exp.incoming_state)
# # print(Exp.current_trial[1:] == current_trial)
# x = np.arange(len(Exp.current_trial))
# y = np.dot(np.asarray(Exp.current_trial).reshape(len(Exp.current_trial), -1), Exp.encoding_representation.reshape(-1))
# plt.ylim(-1, 1.3)
# plt.plot(x, y)
# plt.title("Similarity to encoding state")
# plt.show()
# Exp.plot_current_trial()
#
# # Simulate 10 steps of maintain
# maintain_portion = []
# for n in range(10):
#     current_trial.append(Exp.simulate_step(phase="maintain",
#                                            encode_category="scene"))
#     # print(Exp.incoming_state)
# x = np.arange(len(Exp.current_trial))
# y = np.dot(np.asarray(Exp.current_trial).reshape(len(Exp.current_trial), -1), Exp.encoding_representation.reshape(-1))
# plt.ylim(-1, 1.3)
# plt.plot(x, y)
# plt.title("Similarity to encoding state")
# plt.show()
# Exp.plot_current_trial()

# Simulate 10 steps of replace
oper = "replace"
maintain_portion = []
# Artificially push replacement representation into verbal layer
Exp.current_state[1] = Exp.replacement_representation[1]
for n in range(10):
    print(Exp.replacement_representation)
    new_state = Exp.simulate_step(phase=oper,
                                  diagnostic=True,)
    vis_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["visual"],
                                                  diagnostic=True)[0])
    ver_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["verbal"],
                                                  diagnostic=True)[0])
    comb_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["visual", "verbal"],
                                                  diagnostic=True)[0])

    current_step = Exp.simulate_step(phase=oper,
                                     diagnostic=False,)


# Plot all three traces
for s in range(len(vis_sim)):
    fig, ax = plt.subplots()
    x = np.arange(len(vis_sim[s]))
    # ax.plot(x, np.cumsum(vis_sim[s]), label="visual", color="blue", alpha=0.5)
    # ax.plot(x, np.cumsum(ver_sim[s]), label="verbal", color="green", alpha=0.5)
    # ax.plot(x, np.cumsum(comb_sim[s]), label="combined", color="black")
    ax.plot(x, vis_sim[s], label="visual", color="blue", alpha=0.5)
    ax.plot(x, ver_sim[s], label="verbal", color="green", alpha=0.5)
    ax.plot(x, comb_sim[s], label="combined", color="black")
    ax.axvline(269, color="r", linestyle="--", label="end of localizer")
    phase = "encode" if s < 10 else oper
    secax_y = ax.secondary_yaxis('right', transform=ax.transData)
    secax_y.set_ylabel('cumulative sum')
    ax.legend()
    plt.title(f"Step {s} tau:{mem_params['post_tau']} phase:{phase}")
    plt.ylim(0, 1)
    plt.ylabel("similarity")
    plt.show()

# plot similarity to encoding representation
arr = np.asarray(Exp.current_trial)
x = np.arange(arr.shape[0])
fig, ax = plt.subplots()
for l, layer in enumerate(["visual", "verbal"]):
    y = np.dot(arr[:, l], Exp.encoding_representation[l])
    ax.plot(x, y, label=layer)
plt.axvline(num_enc + 2, color="k", linestyle="--")
plt.ylim(-1, 1.3)
plt.title(f"Similarity to encoding state tau{mem_params['post_tau']} oper:{oper}")
plt.legend()
plt.show()

arr = np.asarray(Exp.current_trial)
x = np.arange(arr.shape[0])
fig, ax = plt.subplots()
for l, layer in enumerate(["visual", "verbal"]):
    y = np.dot(arr[:, l], Exp.replacement_representation[l])
    ax.plot(x, y, label=layer)
plt.axvline(num_enc + 2, color="k", linestyle="--")
plt.ylim(-1, 1.3)
plt.title(f"Similarity to replacement state tau{mem_params['post_tau']}")
plt.legend()
plt.show()


rocauc_arr = np.zeros((len(Exp.current_trial),
                           mem_params["num_categories"]))
for t, tp in enumerate(Exp.current_trial):
    rocauc_arr[t] = Exp.clf.predict_proba(tp[0].reshape(1, -1))
# plot
fig, ax = plt.subplots()
x = np.arange(rocauc_arr.shape[0])
for c, cls in enumerate(mem_params["categories"]):
    ax.plot(x, rocauc_arr[:, c], label=cls)
plt.legend()
plt.axvline(num_enc + 2, color="k", linestyle="--")
plt.title(f"Probas tau:{mem_params['post_tau']}")
plt.show()

#%%
diagnostic = False
# Build a stim list
df = KE.build_stims_Kim2020neuro(mem_params)

# results = simulate_participant(mem_params=mem_params,
#                      diagnostic=diagnostic,
#                      )
results = simulate_full_experiment(mem_params=mem_params,
                                   n_participants=20,
                                   diagnostic=diagnostic)


####################################################################################################################
#%%

results_dict = {}
params_dict = {"post_tau": [2, 4, 8],
               "ic_ratio": [0.5, 1, 2],
               "em_ratio": [0.5, 1, 2],
               "beta": [0.3, 0.65, 0.9]}

for k, v in params_dict.items():
    for val in v:
        params = mem_params.copy() | {"post_tau": 4, "em_ratio": 2, "beta": 0.3} # {k: val} #
        results_dict[k] = simulate_full_experiment(mem_params=params,)

