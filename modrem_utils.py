import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import LabelEncoder

layers_dict = {"visual": 0,
               "verbal": 1,
              # "location": 2,
              # "temporal": 3,
               }


def unit_length(vector):
    return vector / np.linalg.norm(vector, axis=-1, ord=2, keepdims=True)


def simulate_normalized_noise(rng, size):
    rand_init_state = rng.normal(size=size)
    return unit_length(rand_init_state)





class Modrem_Exp(object):
    default_params = {# Experiment design
        "num_loc_items": 54,
        "num_categories": 3,
        "categories": ["face", "scene", "fruit"],
        "operations": ["maintain", "replace", "suppress"],
        "num_loc_repeats": 5,
        "num_main_trials": 270,
        "timesteps_per_phase": 10,
        "trial_reset": True,
        # Model design
        "vec_len": 10,
        "loc_layers": ["visual", "verbal"],
        # "main_layers": ["visual", "verbal"],
        "clf_layers": ["visual"],
        "ic_ratio": 1,    # item vs category ratio
        "em_ratio": 1,    # external vs memory ratio
        "beta": 0.65,
        "tau_style": "linear",
        "tau": 8,
        "post_tau_style": "exp",  # ["exp", "power", "linear"]
        "post_tau": 8,
        "mem_source": "combined",
        "snr": 5,    # signal to noise ratio (not implemented)
        "echo_weights": {
            "visual": 1,
            "verbal": 1,
        },
        "update_rules": {},
        "init_state": "noise",
        "activation_intensity": False,
    }

    plot_colors = ["orange", "blue", "purple"]

    def __init__(self, params=None, seed=None):
        self.params = self.default_params | (params or {})
        # initiate a Memories and a Representations object for storage
        self.memories = Memories()
        self.representations = Representations()
        # create an RNG object to standardize random number generation
        self.rng = np.random.default_rng(seed=seed)
        # Initiate attributes to store current experiment states
        self.current_state = self.initialize_state(init_state=self.params["init_state"])
        self.encoding_representation = None
        self.encoding_item_name = None
        self.replacement_representation = None
        self.replacement_item_name = None
        # Store current trial
        self.current_trial = []
        #
        self.clf = None
        self.label_encoder = LabelEncoder().fit(self.params["categories"])
        self.update_mechanism = UpdateMechanism(rng=self.rng,
                                                representations=self.representations,
                                                memories=self.memories,
                                                params=self.params,)
        #
        self.trials_data = []

    def __repr__(self):
        return f"Modrem_Exp(params={self.params})"

    def classifier_train(self):
        """
        Trains a classifier on the localizer memories
        :return: classifier object
        """
        # First initiate a CV object
        test_fold = [n for n in range(self.params["num_loc_repeats"]) for i in range(self.params["num_loc_items"])]
        cv = PredefinedSplit(test_fold)
        # save some helper values
        total_loc_items = self.params["num_loc_items"] * self.params["num_loc_repeats"]
        # Pull the data
        X = self.memories.loc_memories[:, self.get_clf_layer_inds()].reshape(total_loc_items, -1)
        y = self.label_encoder.transform(self.memories.loc_labels)
        accuracies = np.zeros(cv.get_n_splits())
        for i, (train_idx, test_idx) in enumerate(cv.split()):
            clf = OneVsRestClassifier(LogisticRegression(penalty="l2", solver="liblinear", C=1))
            accuracies[i] = clf.fit(X[train_idx], y[train_idx]).score(X[test_idx], y[test_idx])
        print(f"Accuracy of classifier was {np.mean(accuracies)}. Training classifier on all loc")
        self.clf = OneVsRestClassifier(LogisticRegression(penalty="l2", solver="liblinear", C=1)).fit(X, y)
        return self.clf

    def classify_timepoints(self):
        clf = self.clf
        probas = np.zeros((len(self.params["categories"]),
                           len(self.current_trial)))
        for t, t_step in enumerate(self.current_trial):
            probas[:, t] = clf.predict_proba(t_step[self.get_clf_layer_inds()].reshape(1, -1))
        return probas

    def _create_item_codes(self,
                          vec_len,
                          num_items,
                          num_categories,
                          ic_ratio, ):
        assert num_items % num_categories == 0, "num_items must be divisible by num_categories"
        # first create an array of item codes
        item_codes = unit_length(np.random.randn(num_items, vec_len))
        # then create an array of category codes
        cat_codes = unit_length(np.random.randn(num_categories, vec_len))
        # tile the cat codes across
        cat_codes_tiled = np.tile(cat_codes, (num_items // num_categories, 1))
        # sum the two codes based on a weighting (Item:Category ratio)
        combined_codes = item_codes * ic_ratio + cat_codes_tiled
        # normalize the combined codes to unit length
        combined_codes = unit_length(combined_codes)
        return item_codes, cat_codes, combined_codes

    def create_loc_memories(self, params:dict = None):
        if params is None:
            params = self.params
        item_arr = np.zeros((params["num_loc_items"],
                              len(layers_dict),
                              params["vec_len"]))
        cat_arr = np.zeros((params["num_categories"],
                            len(layers_dict),
                            params["vec_len"]))
        combined_arr = np.zeros((params["num_loc_items"],
                              len(layers_dict),
                              params["vec_len"]))
        for layer in params["loc_layers"]:
            # first create visual codes
            item_layer, cat_layer, combined_layer = self._create_item_codes(params["vec_len"],
                                                                            params["num_loc_items"],
                                                                            len(params["categories"]),
                                                                            params["ic_ratio"],)
            item_arr[:, layers_dict[layer]] = item_layer
            cat_arr[:, layers_dict[layer]] = cat_layer
            combined_arr[:, layers_dict[layer]] = combined_layer
        # Save the generated codes into representations object
        representations = self.representations.save_representations(item_codes=item_arr,
                                                                    cat_codes=cat_arr,
                                                                    combined_codes=combined_arr,
                                                                    params=params)
        loc_memories = []
        labels = []
        for n in range(params["num_loc_repeats"]):
            for name, code in representations["combined"].items():
                loc_memories.append(code)
                labels.append(name.rsplit("_", 1)[0])
        # save to memories object
        self.memories.save_loc_memories(np.asarray(loc_memories), np.asarray(labels))
        return loc_memories

    def initialize_state(self, init_state):
        if init_state == "noise":
            return simulate_normalized_noise(self.rng, (len(layers_dict), self.params["vec_len"]))
        elif init_state == "zeros":
            return np.zeros((len(layers_dict), self.params["vec_len"]))
        else:
            print(f"Unknown init state: {init_state}")

    def initialize_trial(self, category=None, item=None):
        if self.params["trial_reset"]:
            self.reset_current_trial()
        # Add the current state to the list of time steps in current trial
        self.current_trial.append(self.current_state)
        # Also choose an image to be encoded
        self.encoding_representation, self.encoding_item_name = self.representations.query_representations(category=category,
                                                                                                           item=item,)
        return None

    def initialize_replacement(self, category=None, item=None):
        # find the current encoded category and choose an image not from this category
        current_encoded_category = self.encoding_item_name.rsplit("_", 1)[0]
        if item is not None:
            assert item.rsplit("_", 1)[0] != current_encoded_category, f"{item} belongs to current encoded category: {current_encoded_category}"
        elif category is not None:
            assert category != current_encoded_category, f"{category} is the same as current encoded category: {current_encoded_category}"
        else:
            category = np.random.choice([c for c in self.params["categories"] if c != current_encoded_category])
        self.replacement_representation, self.replacement_item_name = self.representations.query_representations(category=category, item=item)
        return self.replacement_representation, self.replacement_item_name


    def get_clf_layer_inds(self):
        return [layers_dict[l] for l in self.params["clf_layers"]]

    def plot_current_trial(self, **kwargs):
        probas = self.classify_timepoints()
        fig, ax = plt.subplots()
        x = np.arange(probas.shape[1])
        for i, cls in enumerate(probas):
            ax.plot(x, cls, color=self.plot_colors[i], label=self.label_encoder.inverse_transform([i])[0])
        plt.ylim(0, 1)
        plt.title("Decoding of trial")
        plt.legend()
        plt.show()

    def reset_current_trial(self):
        self.current_trial = []
        self.current_state = self.initialize_state(init_state=self.params["init_state"])
        self.encoding_representation = None
        self.encoding_item_name = None
        self.replacement_item_name = None
        self.replacement_representation = None

    def reset_task_memories(self):
        self.memories.task_memories = None

    def simulate_experiment(self,
                            stim_df):
        trials_list = []
        for _, row in stim_df.iterrows():
            encode_item = "_".join([row.category, str(row.stim)])
            replace_item = "_".join([row.replace_category, str(row.replace_stim)])
            # Initialize the trial
            self.initialize_trial(item=encode_item)
            # run the trial
            trials_list.append(self.simulate_trial(operation=row.operation,
                                                   encode_item=encode_item,
                                                   replace_item=replace_item))
        return trials_list

    def simulate_step(self, phase,
                      diagnostic=False):
        if phase == "encode":
            item_shown = self.encoding_item_name
        elif phase == "replace":
            item_shown = self.replacement_item_name
        else:
            item_shown = None

        new_state = self.update_mechanism.step(phase=phase,
                                               current_state=self.current_state,
                                               item=item_shown,
                                               )
        if not diagnostic:
            # Now save the info into current trial
            self.current_state = new_state
            self.current_trial.append(new_state)
            self.memories.save_task_memory(new_state)
        return new_state

    def simulate_trial(self, operation,
                       encode_item=None,
                       replace_item=None,
                       diagnostic=False,
                       **kwargs):
        if self.params["trial_reset"]:
            self.reset_current_trial()
        # Initialize trial and replacement
        self.initialize_trial(item=encode_item)
        self.initialize_replacement(item=replace_item)
        # Initiate trial data list
        trial_data = []
        for n in range(2):
            trial_data.append(self.simulate_step(phase="noise",))
        for phase in ["encode", operation]:
            if diagnostic:
                if operation == "replace":
                    # print("pushing replacement representation to current state")
                    # print("pre state", self.current_state)
                    # print("replacement item", self.replacement_representation)
                    self.current_state[layers_dict["verbal"]] = self.replacement_representation[layers_dict["verbal"]]
                    # print("post state", self.current_state)
            for n in range(self.params["timesteps_per_phase"]):
                trial_data.append(self.simulate_step(phase=phase,))
        self.trials_data.append(trial_data)
        return trial_data




class Memories(object):

    def __init__(self):
        self.loc_memories = None
        self.loc_labels = None
        self.task_memories = None

    def __repr__(self):
        return f"Memories object with loc_memories.shape={self.loc_memories.shape} and task_memories.shape={self.task_memories.shape}"

    def retrieve_visual_memory(self, verbal_query):
        """
        :param verbal_query: single vector of len = vec_len
        :return:
        """
        item_memories = self.loc_memories[verbal_query]
        # compute the similarity between the verbal code and all the memories
        similarity = np.dot(item_memories[:, 1, :], verbal_query)
        # find the visual codes that correspond to the largest cosine similarities
        max_similarity = similarity[np.abs(similarity).argmax()]
        # Find all values if multiple argmaxes
        visual_codes = item_memories[np.where(similarity == max_similarity), 0].squeeze()
        return visual_codes[np.random.randint(len(visual_codes))]

    def reset_memories(self):
        self.loc_memories = None
        self.task_memories = None
        print("Memories reset")

    def reset_task_memories(self):
        self.task_memories = np.array([])

    def save_loc_memories(self, loc_memories, labels):
        self.loc_memories = loc_memories
        # Also generate the labels for the loc memories
        self.loc_labels = labels
        return None

    def save_task_memory(self, memory):
        # Add a time dimension to single timepoint memory
        memory = np.expand_dims(memory, axis=0)
        if self.task_memories is None:
            self.task_memories = memory
        else:
            self.task_memories = np.concatenate((self.task_memories, memory), axis=0)


    def get_current_memories(self, source):
        if source == "combined":
            if self.task_memories is None:
                self.task_memories = np.empty((0,
                                               self.loc_memories.shape[1],
                                               self.loc_memories.shape[2],))
            memories = np.concatenate((self.loc_memories,
                                       self.task_memories), axis=0)
        elif source == "loc":
            memories = self.loc_memories
        elif source == "task":
            memories = self.task_memories
        else:
            raise SyntaxError('source not recognized. Must be "combined" or "loc" or "task"')
        return memories





class Representations(object):
    def __init__(self,):
        self.representations = {"category": {},
                                "item": {},
                                "combined": {}}
        self.categories = None
        self.vec_len = None


    def __repr__(self):
        return f"Representations object with {len(self.representations)} items from {self.categories} categories"


    def save_representations(self, item_codes, cat_codes, combined_codes,
                             params):
        # add some helper
        item_per_cat = params["num_loc_items"] // len(params["categories"])
        num_categories = len(params["categories"])
        code_nums = [c for c in range(item_per_cat) for n in range(num_categories)]
        # # Check that the number of items are divisible by number of categories
        # assert codes_arr.shape[0] % num_categories == 0, "number of unique items must be divisible by num_categories"
        # Save category codes in representations
        self.representations["category"] = {k: v for k, v in zip(params["categories"], cat_codes)}
        # Save the item and combined representations by name
        for i in range(params["num_loc_items"]):
            item_name = f"{params["categories"][i % num_categories]}_{code_nums[i]}"
            self.representations["item"][item_name] = item_codes[i]
            self.representations["combined"][item_name] = combined_codes[i]
        self.categories = params["categories"]
        self.vec_len = combined_codes.shape[-1]
        return self.representations


    def query_category_representation(self, category):
        return self.representations["category"][category]


    def query_item_representation(self, item):
        return self.representations["item"][item]


    def query_representations(self, category=None, item=None, **kwargs):
        if item is not None:
            assert item in self.representations["combined"].keys(), f"{item} does not exist"
            incoming_img = item
        elif category is not None:
            assert category in self.categories, f"{category} is not in current categories"
            cat_img_list = [i for i in self.representations["combined"].keys() if category in i]
            incoming_img = np.random.choice(cat_img_list)
        else:
            incoming_img = np.random.choice(list(self.representations["combined"].keys()))
        logging.info(f"Incoming image is {incoming_img}")
        return self.representations["combined"][incoming_img], incoming_img





class UpdateMechanism(object):
    default_rules = {
        "noise": {
            "external": {"visual": "noise",
                         "verbal": "noise",},
            "memory": {"echo_layers": [],
                       "noise_layers": ["visual", "verbal"],
                       "tau_dilation": 1},
        },
        "encode": {
            "external": {"visual": "representation",
                         "verbal": "noise"},
            "memory": {"echo_layers": [],
                       "noise_layers": ["visual", "verbal"],
                       "tau_dilation": 1},

        },
        "maintain": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual", "verbal"],
                       "noise_layers": [],
                       "tau_dilation": 1}
        },
        "replace": {
            "external": {"visual": "noise",
                         "verbal": "representation"},
            "memory": {"echo_layers": ["visual", "verbal"],
                       "noise_layers": [],
                       "tau_dilation": 1}
        },
        "suppress": {
            "external": {"visual": "noise",
                         "verbal": "noise"},
            "memory": {"echo_layers": ["visual", "verbal"],
                       "noise_layers": [],
                       "tau_dilation": 0.5}
        }
    }
    def __init__(self,
                 rng,
                 representations: Representations,
                 memories: Memories,
                 params: dict):
        self.update_rules = self.default_rules | params["update_rules"]
        self.rng = rng
        self.representations = representations
        self.memories = memories
        self.params = params

    def build_external_input(self, spec, item):
        external_spec = spec["external"]
        ext_state = np.zeros((len(layers_dict),
                              self.representations.vec_len))
        for layer, source in external_spec.items():
            if source == "noise":
                ext_state[layers_dict[layer]] = simulate_normalized_noise(self.rng,
                                                                          size=self.representations.vec_len)
            elif source == "representation":
                ext_state[layers_dict[layer]] = self.representations.query_representations(item=item)[0][layers_dict[layer]]
            else:
                raise NotImplementedError(f"unknown source for external input: {source}")
        return ext_state

    def build_memory_input(self, spec, current_state):
        memory_spec = spec["memory"]
        pre_tau = self.params["tau"] * memory_spec["tau_dilation"] if self.params[
                                                                          "tau_style"] != "linear" else "linear"

        post_tau = self.params["post_tau"] * memory_spec["tau_dilation"] if self.params[
                                                                                "post_tau_style"] != "linear" else "linear"
        # Calculate the echo
        mem_state = self.calc_echo(probe=current_state,
                                   probe_layers=memory_spec["echo_layers"],
                                   tau=pre_tau,
                                   post_tau=post_tau,
                                   intensity=self.params["activation_intensity"])
        # Replace layers with noise as defined by noise layers
        for layer in memory_spec["noise_layers"]:
            mem_state[layers_dict[layer]] = simulate_normalized_noise(self.rng,
                                                                      size=self.representations.vec_len)
        return mem_state

    def calc_echo(self, probe, probe_layers, tau=None, post_tau=None, intensity=True, diagnostic=False):
        """
        Calls the calc_similarity function to compute the similarity weighted echo
        :param probe:
        :param probe_layers:
        :param tau: parameter controlling 'sharpness' of similarity
        :return:
        """
        # pull the memstack
        memstack = self.get_memstack()
        if not probe_layers:
            return np.zeros((probe.shape[0], probe.shape[1]))
        # pull the scaled similarity
        similarity = self.calc_similarity(probe=probe,
                                          probe_layers=probe_layers,
                                          memstack=memstack,
                                          tau=tau)["scaled"]
        # Take the summed similarity across layers
        if len(probe_layers) > 1:
            similarity = self.calc_summed_similarity(similarity=similarity,
                                                     tau=post_tau,
                                                     probe_layers=probe_layers,)
        if not intensity:
            # normalize similarity
            similarity /= similarity.sum()
        # weighted sum of memstack across memories
        weighted_sum = (memstack * similarity[:, None, None]).sum(axis=0)
        # if diagnoising, return the similarity
        if diagnostic:
            return similarity, weighted_sum
        # normalize by each layer
        return weighted_sum / np.linalg.norm(weighted_sum, ord=2, axis=1, keepdims=True)

    def calc_similarity(self,
                        probe,
                        probe_layers,
                        memstack,
                        tau=None):
        # Declare tau
        tau = tau or self.params["tau"]
        similarity = np.zeros((len(layers_dict), len(memstack)))
        # Convert probe layers to list if type is not list
        probe_layers = [probe_layers] if type(probe_layers) is not list else probe_layers
        # Make sure probe is not 1 dimensional
        probe = probe[np.newaxis, :] if probe.ndim == 1 else probe
        # Calculate the dot product similarity
        for i, probe_vector in enumerate(probe):
            similarity[i, :] = np.dot(memstack[:, i], probe_vector)
        # remove the non-computed layers
        keep_layers = np.array([True if l in probe_layers else False for l in layers_dict.keys()])
        similarity = np.delete(similarity, ~keep_layers, axis=0).squeeze()
        # nonlinear scaling of the similarity values  (pulled from cmrwm-Polyn)
        if self.params["tau_style"] == 'power':
            scaled_similarity = similarity ** tau
        elif self.params["tau_style"] == 'exp':
            scaled_similarity = 1 / np.exp((1 - similarity) * tau)
        elif self.params["tau_style"] == 'linear':
            scaled_similarity = similarity
        else:
            # if the tau_style string is not in the list above
            raise SyntaxError('tau_style not recognized')
        return {"raw": similarity, "scaled": scaled_similarity,}

    def calc_summed_similarity(self, similarity, tau=None, probe_layers=None):
        """
        Placeholder function to produce summed similarity across layers
        :param similarity_dict:
        :return:
        """
        # Convert probe layers to list if type is not list
        probe_layers = [probe_layers] if type(probe_layers) is not list else probe_layers
        # Then order it by the layers dict
        probe_layers = [l for l in sorted(layers_dict, key=lambda x: layers_dict[x]) if l in probe_layers]
        # declare tau
        tau = tau or self.params["post_tau"]
        mult_vals = np.ones((similarity.shape[-1],))
        for n_layer in range(similarity.shape[0]):
            mult_vals *= (similarity[n_layer, :] * self.params["echo_weights"][probe_layers[n_layer]])
        # nonlinear scaling of the similarity values
        if self.params["post_tau_style"] == 'power':
            mult_vals = mult_vals ** tau
        elif self.params["post_tau_style"] == 'exp':
            mult_vals = 1 / np.exp((1 - mult_vals) * tau)
        elif self.params["post_tau_style"] == 'linear':
            pass
        else:
            # if the tau_style string is not in the list above
            raise SyntaxError('tau_style not recognized')
        return mult_vals

    def calc_new_state(self,
                       current_state,
                       incoming_state):
        """
        Calculates new state by updating state across layer
        :return:
        """
        beta = self.params["beta"]
        # Initiate new information with zeros
        new_state = np.full((len(layers_dict),
                             self.params["vec_len"]),
                            np.nan)
        for layer_ind in layers_dict.values():
            # calculate new current state
            curr_in_similarity = np.dot(current_state[layer_ind], incoming_state[layer_ind])
            # Calculate Rho
            rho = np.sqrt(1 + (beta ** 2) * (curr_in_similarity ** 2 - 1)) - beta * curr_in_similarity
            new_state[layer_ind] = rho * current_state[layer_ind] + beta * incoming_state[layer_ind]
        return new_state

    def get_memstack(self):
        return self.memories.get_current_memories(self.params["mem_source"])

    def step(self,
             phase: str,
             current_state: np.ndarray,
             item: str,
             ):
        """
        Takes a time step by updating the current state to a new state using phase rules
        :param phase:
        :param current_state:
        :param item:
        :return:
        """
        # Find the specifications for combining inputs for this phase
        spec = self.update_rules[phase]
        # Get the external input
        # print(f"item for step is {item}")
        ext_input = self.build_external_input(spec=spec, item=item)
        # As well as the memory input
        mem_input = self.build_memory_input(spec=spec, current_state=current_state)

        # Then combine external input and memory input based on mixing ratio
        incoming_state = unit_length(ext_input * self.params["em_ratio"] + mem_input)
        return self.calc_new_state(current_state=current_state,
                                   incoming_state=incoming_state)












