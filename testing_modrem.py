import numpy as np
from modrem_utils import Modrem_Exp
import matplotlib.pyplot as plt
import KimEtal2020_sim as ke

# %%
mem_params = {  # Experiment design
    "num_loc_items": 540,
    "num_categories": 3,
    "categories": ["face", "scene", "fruit"],
    "operations": ["maintain", "replace", "suppress", "clear"],  # "noise"
    "num_loc_repeats": 5,
    "num_main_trials": 360,
    "timesteps_per_phase": 10,
    "iti": 2,
    "trial_reset": False,
    # Model design
    "vec_len": 10,
    "representation_layers": ["visual", "verbal"],
    "memory_layers": ["visual", "verbal", "temporal"],# ],
    "clf_layers": ["visual"],
    "ic_ratio": 1,  # item vs category ratio
    "em_ratio": 0.6,  # external vs memory ratio
    "em_temporal_ratio": 2,
    "beta": 0.75,
    "beta_temporal": 0.075,
    "tau_style": "exp",
    "tau": 8,
    "post_tau_style": "linear",  # ["exp", "power", "linear"]
    "post_tau": np.nan,
    "mem_source": "combined",
    "echo_weights": {
        "visual": 1,
        "verbal": 1,
        "temporal": 0.25,
    },  # allows for
    "update_rules": {
        "encode": {
            "external": {"visual": "representation",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual", "verbal", "temporal"],  # "verbal", "temporal"
                       "noise_layers": [],
                       "tau_dilation": 1},
        },
        "replace": {
            "external": {"visual": "noise",
                         "verbal": "representation"},
            "memory": {"echo_layers": ["verbal", "temporal"],  # "visual", "temporal"
                       "noise_layers": [],
                       "tau_dilation": 1},
        },
        "maintain": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual", "verbal", "temporal"],  # "verbal", "temporal"
                       "noise_layers": [],
                       "tau_dilation": 1},
        },
        "suppress": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual", "verbal", "temporal"],  # "verbal", "temporal"
                       "noise_layers": [],
                       "tau_dilation": 0.3}
        },
        "clear": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["temporal"],  # "temporal"
                       "noise_layers": ["visual", "verbal"],
                       "tau_dilation": 1},
        },
        "noise": {
            "external": {"visual": "noise",
                         "verbal": "noise", },
            "memory": {"echo_layers": [],
                       "noise_layers": ["visual", "verbal", "temporal"],
                       "tau_dilation": 1},
        },
    },
    "init_state": "noise",
    "activation_intensity": False,
    "hrf": True,
    "unique_items": True,
}
# %% ### Run a single participant
em_ratios = [0.5, 1, 2, 4, 8]
betas = [0.05, 0.1, 0.2, 0.6]
combs = [(b, e) for e in em_ratios for b in betas]
for b, r in combs:
    mem_params["beta_temporal"] = b
    mem_params["em_temporal_ratio"] = r
    exp = ke.simulate_participant(params=mem_params,
                                  diagnostic=False)

    exp.plot_all_context_rsa(plot_previous=True,
                             title=f"em_temp_ratio:{mem_params["em_temporal_ratio"]} beta_temp:{mem_params["beta_temporal"]}")


#
# #%%
# mem_params["em_temporal_ratio"] = 32
# for i in 2 ** np.arange(2, 6):
#     mem_params["em_temporal_ratio"] = i
    exp_list = ke.simulate_full_experiment(params=mem_params,
                                           n_participants=30,
                                           n_jobs=10)
    # # Graph 4a: Timecourse for neural decoding of a WM item
    results_arr = ke.timecourse_cat_decoding(exp_list=exp_list,
                                             params=mem_params,
                                             shifted=True,
                                             oversampling=20,
                                             tr=0.6,
                                             shift_delay=4.6,
                                             title=f"em_temp_ratio:{mem_params["em_temporal_ratio"]} beta_temp:{mem_params["beta_temporal"]}"
                                             )
    # Graph 5b: WM Operation impact on encoding fidelity
    ke.graph_proactive_interference(exp_list=exp_list,
                                    params=mem_params,
                                    fisher=True,
                                    balance=True,
                                    layers="visual",
                                    exclude_rep_sameCat=True,
                                    timepoints="all",
                                    n_jobs=10,
                                    plot_delta=True,
                                    )
    # ke.graph_operDiff_itemRSA(exp_list=exp_list,
    #                           params=mem_params,
    #                           fisher=True,
    #                           layers="visual",
    #                           shifted=True,
    #                           oversampling=20,
    #                           tr=0.6,
    #                           shift_delay=4.6,
    #                           ylim=(-0.36, 0.07)
    #                           )

####################################################################################################################
# %% ### Simulate a single trial
mem_params["beta_temporal"] = 0.4
mem_params["em_temporal_ratio"] = 0.5
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
previous_temporal_representation = Exp.current_state[2]
print("next:\n", Exp.next_temporal_representation)
print("prev:\n", Exp.current_state[2])
current_trial = []
vis_sim = []
ver_sim = []
comb_sim = []
# Start off with some noise timesteps
for n in range(2):
    Exp.simulate_step(phase="noise", )
##
# for n in range(30):
#     Exp.simulate_step(phase="noise", )
# print(Exp.current_state[2])
# # plot the temporal similarity
# sim = Exp.plot_trial_context_rsa(title=f"em_temp_ratio:{mem_params["em_temporal_ratio"]} beta_temp:{mem_params["beta_temporal"]}")
# # do it again
# for n in range(30):
#     Exp.simulate_step(phase="noise", )
# print(Exp.current_state[2])
# # plot the temporal similarity
# sim = Exp.plot_trial_context_rsa(title=f"em_temp_ratio:{mem_params["em_temporal_ratio"]} beta_temp:{mem_params["beta_temporal"]}")


num_enc = 10
encode_phase = "encode"
for n in range(num_enc):
    new_state = Exp.simulate_step(phase=encode_phase,
                                  diagnostic=True, )
    # New state is the combined ext+memory after stepping
    vis_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["visual"],
                                                  intensity=True,
                                                  diagnostic=True)[0])
    ver_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                  probe_layers=["verbal"],
                                                  intensity=True,
                                                  diagnostic=True)[0])
    comb_sim.append(Exp.update_mechanism.calc_echo(probe=new_state,
                                                   probe_layers=["visual", "verbal"],
                                                   intensity=True,
                                                   diagnostic=True,
                                                   )[0])
    current_step = Exp.simulate_step(phase=encode_phase, )

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
# Exp.current_state[1] = Exp.replacement_representation[1]
for n in range(10):
    # print(Exp.replacement_representation)
    new_state = Exp.simulate_step(phase=oper,
                                  diagnostic=True, )
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
                                     diagnostic=False, )
#
# # Plot all three similarity traces for each step
# for s in range(len(vis_sim)):
#     fig, ax = plt.subplots()
#     x = np.arange(len(vis_sim[s]))
#     # ax.plot(x, np.cumsum(vis_sim[s]), label="visual", color="blue", alpha=0.5)
#     # ax.plot(x, np.cumsum(ver_sim[s]), label="verbal", color="green", alpha=0.5)
#     # ax.plot(x, np.cumsum(comb_sim[s]), label="combined", color="black")
#     ax.plot(x, vis_sim[s], label="visual", color="blue", alpha=0.5)
#     ax.plot(x, ver_sim[s], label="verbal", color="green", alpha=0.5)
#     ax.plot(x, comb_sim[s], label="combined", color="black")
#     ax.axvline(269, color="r", linestyle="--", label="end of localizer")
#     # also plot a line for the encoded image and the replacement image in the localizer
#     for r_rep in np.unique(np.where(Exp.memories.loc_memories == Exp.replacement_representation)[0]):
#         ax.axvline(r_rep, color="darkblue", linestyle="--", label="replacement image in loc", alpha=0.2)
#     for e_rep in np.unique(np.where(Exp.memories.loc_memories == Exp.encoding_representation)[0]):
#         ax.axvline(e_rep, color="black", linestyle="--", label="encoding image in loc", alpha=0.2)
#
#     phase = "encode" if s < 10 else oper
#     secax_y = ax.secondary_yaxis('right', transform=ax.transData)
#     secax_y.set_ylabel('cumulative sum')
#     # ax.legend()
#     plt.title(f"Step {s} tau:{mem_params['post_tau']} phase:{phase}")
#     plt.ylim(0, 1)
#     plt.ylabel("similarity")
#     plt.show()
#
# # plot similarity to encoding representation
# arr = np.asarray(Exp.current_trial)
# x = np.arange(arr.shape[0])
# fig, ax = plt.subplots()
# for l, layer in enumerate(["visual", "verbal"]):
#     y = np.dot(arr[:, l], Exp.encoding_representation[l])
#     ax.plot(x, y, label=layer)
# plt.axvline(num_enc + 2, color="k", linestyle="--")
# plt.ylim(-1, 1.3)
# plt.title(f"Similarity to encoding state tau{mem_params['post_tau']} oper:{oper}")
# plt.legend()
# plt.show()
#
# arr = np.asarray(Exp.current_trial)
# x = np.arange(arr.shape[0])
# fig, ax = plt.subplots()
# for l, layer in enumerate(["visual", "verbal"]):
#     y = np.dot(arr[:, l], Exp.replacement_representation[l])
#     ax.plot(x, y, label=layer)
# plt.axvline(num_enc + 2, color="k", linestyle="--")
# plt.ylim(-1, 1.3)
# plt.title(f"Similarity to replacement state tau{mem_params['post_tau']}")
# plt.legend()
# plt.show()
#
# rocauc_arr = np.zeros((len(Exp.current_trial),
#                        mem_params["num_categories"]))
# for t, tp in enumerate(Exp.current_trial):
#     rocauc_arr[t] = Exp.clf.predict_proba(tp[0].reshape(1, -1))
# # plot
# fig, ax = plt.subplots()
# x = np.arange(rocauc_arr.shape[0])
# for c, cls in enumerate(mem_params["categories"]):
#     ax.plot(x, rocauc_arr[:, c], label=cls)
# plt.legend()
# plt.axvline(num_enc + 2, color="k", linestyle="--")
# plt.title(f"Probas tau:{mem_params['post_tau']}")
# plt.show()

# plot the temporal similarity
Exp.plot_trial_context_rsa(plot_previous=True,
                           title=f"em_temp_ratio:{mem_params["em_temporal_ratio"]} beta_temp:{mem_params["beta_temporal"]}")
# ext_input = Exp.update_mechanism.build_external_input(spec=mem_params["update_rules"]["encode"],
#                                                       item=Exp.encoding_item_name,
#                                                       context=Exp.next_temporal_representation)


# exp.plot_all_context_rsa()

# %% ### Run a single participant
exp = ke.simulate_participant(params=mem_params,
                              diagnostic=False)

exp.plot_all_context_rsa(plot_previous=True,)

# %% ### Run a single experiment
diagnostic = False
# # Build a stim list
# df = KE.build_stims_Kim2020neuro(mem_params)
# results = simulate_participant(mem_params=mem_params,
#                      diagnostic=diagnostic,
#                      )
# results = simulate_full_experiment(mem_params=mem_params,
#                                    n_participants=20,
#                                    diagnostic=diagnostic)
# try:
#     results = simulate_full_experiment(mem_params=mem_params,
#                                    n_participants=20,
#                                    diagnostic=diagnostic)
# except Exception as e:
#     print(f"tau={tau_style}, post_tau={post_tau_style} failed: {e}")
# mem_params["update_rules"]["encode"]["memory"]["echo_layers"] = ["visual"]
#
# mem_params["update_rules"]["maintain"]["memory"]["echo_layers"] = ["visual"]
# mem_params["update_rules"]["suppress"]["memory"]["echo_layers"] = ["visual"]
# mem_params["update_rules"]["replace"]["memory"]["echo_layers"] = ["verbal"]
# Use new simulation functions
exp_list = ke.simulate_full_experiment(params=mem_params,
                                       n_participants=30,
                                       n_jobs=10)

# Graph 4a: Timecourse for neural decoding of a WM item
results_arr = ke.timecourse_cat_decoding(exp_list=exp_list,
                                         params=mem_params,
                                         shifted=True,
                                         oversampling=20,
                                         tr=0.6,
                                         shift_delay=4.6,
                                         )
# Graph 4b[i]: Trajectory for removal of an item from WM (Category)
ke.graph_operDiff_catDecode(exp_list=exp_list,
                            params=mem_params,
                            shifted=True,
                            oversampling=20,
                            tr=0.6,
                            shift_delay=4.6,
                            ylim=(-0.6, 0.15)
                            )
# Graph 4b[ii]: Trajectory for removal of an item from WM (Item)
ke.graph_operDiff_itemRSA(exp_list=exp_list,
                          params=mem_params,
                          fisher=True,
                          layers="visual",
                          shifted=True,
                          oversampling=20,
                          tr=0.6,
                          shift_delay=4.6,
                          ylim=(-0.33, 0.04)
                          )
# Graph 5b: WM Operation impact on encoding fidelity
ke.graph_proactive_interference(exp_list=exp_list,
                                params=mem_params,
                                # similarity_coefficient="pearson",
                                fisher=True,
                                layers="visual",
                                exclude_rep_sameCat=True,
                                timepoints="all",
                                n_jobs=-1,
                                )
# # Memory test
# beh_results = ke.summarize_memory_test(exp_list=exp_list,
#                                        mem_type="ltm",
#                                        support_layer="visual",
#                                        tau=40)

for i in range(3):
    exp_list = ke.simulate_full_experiment(params=mem_params,
                                           n_participants=30,
                                           n_jobs=10)
    # Graph 5b: WM Operation impact on encoding fidelity
    ke.graph_proactive_interference(exp_list=exp_list,
                                    params=mem_params,
                                    fisher=True,
                                    balance=True,
                                    layers="visual",
                                    exclude_rep_sameCat=True,
                                    timepoints="all",
                                    n_jobs=-1,
                                    plot_delta=True,
                                    )
    #
    # for layer in ["visual", "all"]:
    #     for memtype in ["ltm", "disc_wm", "cum_wm"]:
    #         beh_results = ke.summarize_memory_test(exp_list=exp_list,
    #                                                mem_type=memtype,
    #                                                support_layer=layer,
    #                                                tau=12,
    #                                                savefig=f"figures/{memtype}_{layer}_{i}.png", )

####################################################################################################################
# %% ### Gridsearch through parameters
# styles = ["exp", "linear", "power", "sigmoid", "softmax"]
# from itertools import combinations, permutations, combinations_with_replacement
# # for tau_style, post_tau_style in combinations_with_replacement(styles, 2):
# # for t in np.linspace(-1, 1.3, 8):
# mem_params["tau"] = 8
# mem_params["tau_style"] = "exp"
# mem_params["post_tau"] = 1
# mem_params["post_tau_style"] = "linear"
# mem_params["beta"] = 0.99
# mem_params["em_ratio"] = 0.9
# mem_params["timesteps_per_phase"] = 15
# mem_params["update_rules"]["suppress"]["memory"]["tau_dilation"] = 0.65
# mem_params["operations"] = ["maintain", "replace", "suppress"] # , "noise"
# # print(f"testing: tau={tau_style}, post_tau={post_tau_style}")
#
#
# exp_list = simulate_full_experiment(params=mem_params,
#                                     n_participants=20)
# results_arr = timecourse_cat_decoding(exp_list=exp_list,
#                                       params=mem_params,)
# results_dict = {}
# params_dict = {"post_tau": [2, 4, 8],
#                "ic_ratio": [0.5, 1, 2],
#                "em_ratio": [0.5, 1, 2],
#                "beta": [0.3, 0.65, 0.9]}
#
# for k, v in params_dict.items():
#     for val in v:
#         params = mem_params.copy() | {"post_tau": 4, "em_ratio": 2, "beta": 0.3} # {k: val} #
#         results_dict[k] = simulate_full_experiment(mem_params=params,)

# %% ### Create Kim et al. (2020) graphs


x = np.linspace(-6, 6, 1000)


# Normal PDF function
def normal_pdf(x, mean=0, sd=1):
    return (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean) / sd) ** 2
    )


# Parameters
mean = 0
sds = [0.5, 2]

# fig, ax = plt.subplots(figsize=(4, 4))
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["darkgoldenrod", "sandybrown"]
for s, sd in enumerate(sds):
    y = normal_pdf(x, mean=mean, sd=sd)
    ax.plot(x, y, label=f"{["large", "small"][s]} tau", color=colors[s])

ax.axis("off")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_title("Normal distributions with different standard deviations")
ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.42, 1.1)
)
plt.savefig("example_tau.png")
# plt.tight_layout()
plt.show()
