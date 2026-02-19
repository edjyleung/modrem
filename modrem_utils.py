import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import LabelEncoder



class Modrem_Exp(object):
    default_params = {"vec_len": 10,
                      "num_loc_items": 54,
                      "num_categories": 3,
                      "categories": ["face", "scene", "fruit"],
                      "layers_dict": {"visual": 0,
                                      "verbal": 1,
                                      # "location": 2,
                                      # "temporal": 3,
                                      },
                      "loc_layers": ["visual", "verbal"],
                      "main_layers": ["visual"],
                      "ic_ratio": 1,
                      "num_loc_repeats": 5,
                      "beta": 0.65,
                      "trial_reset": True,
                      "post_tau_style": "exp",    # ["exp", "power", "linear"]
                      "post_tau": 8,
                      "mem_source": "combined",
                      "snr": 5,
                      }

    plot_colors = ["orange", "blue", "purple"]

    def __init__(self, params=None):
        params = params or {}
        self.params = self.default_params | params

        self.memories = Memories()
        self.representations = Representations()
        self.current_state = None
        self.encoding_state = None
        self.encoded_item = None
        self.replacement_state = None
        self.clf = None
        self.current_trial = []
        self.current_similarity = {}
        self.label_encoder = LabelEncoder().fit(self.params["categories"])

    def __repr__(self):
        return f"Modrem_Exp(params={self.params})"

    def calc_new_state(self, current_state, incoming_state, beta=None, **kwargs):
        """
        Calculates new state by updating state across layer
        :param current_state:
        :param incoming_state:
        :return:
        """
        beta = beta or self.params["beta"]
        # Initiate new information with zeros
        new_state = np.full((len(self.params["layers_dict"]),
                              self.params["vec_len"]),
                            np.nan)

        for layer_ind in self.params["layers_dict"].values():
            # calculate new current state
            curr_in_similarity = np.dot(current_state[layer_ind], incoming_state[layer_ind])
            # Calculate Rho
            rho = np.sqrt(1 + (beta ** 2) * (curr_in_similarity ** 2 - 1)) - beta * curr_in_similarity
            new_state[layer_ind] = rho * current_state[layer_ind] + beta * incoming_state[layer_ind]
        return new_state

    def calc_similarity(self, probe, probe_layers, memstack):
        similarity = np.full((len(memstack), len(self.params["layers_dict"])),
                             np.nan)
        # Convert probe layers to list if type is not list
        probe_layers = [probe_layers] if type(probe_layers) is not list else probe_layers
        # Calculate the dot product similarity
        for i, probe_vector in enumerate(probe):
            similarity[:, i] = np.dot(memstack[:, i], probe_vector)
        # remove the non-computed layers
        keep_layers = np.array([True if l in probe_layers else False for l in self.params["layers_dict"].keys()])
        similarity = np.delete(similarity, ~keep_layers, axis=1).squeeze()
        # nonlinear scaling of the similarity values  (pulled from cmrwm-Polyn)
        if self.params['post_tau_style'] == 'power':
            scaled_similarity = similarity ** self.params['post_tau']
        elif self.params['post_tau_style'] == 'exp':
            scaled_similarity = 1 / np.exp((1 - similarity) * self.params['post_tau'])
        elif self.params['post_tau_style'] == 'linear':
            scaled_similarity = similarity
        else:
            # if the tau_style string is not in the list above
            raise SyntaxError('tau_style not recognized')
        # Save it into a dict
        self.current_similarity = {"raw": similarity,
                                   "scaled": scaled_similarity,}
        return scaled_similarity

    def _calc_similarity_weighted_rep(self, similarity, memstack):
        # normalize similarity
        similarity /= similarity.sum()
        # weighted sum of memstack across memories
        weighted_sum = (memstack * similarity[:, None, None]).sum(axis=0)
        # normalize by each layer
        return weighted_sum / np.linalg.norm(weighted_sum, ord=2, axis=1, keepdims=True)

    def _calc_summed_similarity(self):
        similarity = self.current_similarity["scaled"]
        mult_vals = np.ones((len(similarity),))
        for n_layer in range(similarity.shape[1]):
            mult_vals *= similarity[:, n_layer]
        # nonlinear scaling of the similarity values
        if self.params['post_tau_style'] == 'power':
            mult_vals = mult_vals ** self.params['post_tau']
        elif self.params['post_tau_style'] == 'exp':
            mult_vals = 1 / np.exp((1 - mult_vals) * self.params['post_tau'])
        elif self.params['post_tau_style'] == 'linear':
            pass
        else:
            # if the tau_style string is not in the list above
            raise SyntaxError('tau_style not recognized')
        self.current_similarity["cross_layer"] = mult_vals
        return self.current_similarity

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
            probas[:, t] = clf.predict_proba(t_step[self.get_clf_layer_inds()])
        return probas

    def _create_item_codes(self,
                          vec_len,
                          num_items,
                          num_categories,
                          ic_ratio, ):
        assert num_items % num_categories == 0, "num_items must be divisible by num_categories"
        # first create an array of item codes
        item_codes = np.random.randn(num_items, vec_len)
        # then create an array of category codes
        cat_codes = np.random.randn(num_categories, vec_len)
        cat_codes = np.tile(cat_codes, (num_items // num_categories, 1))
        # sum the two codes based on a weighting (Item:Category ratio)
        combined_codes = item_codes * ic_ratio + cat_codes
        # normalize the codes to unit length
        return combined_codes / np.linalg.norm(combined_codes, axis=1, ord=2, keepdims=True)

    def create_loc_memories(self, params:dict = None):
        if params is None:
            params = self.params
        codes_arr = np.zeros((params["num_loc_items"],
                              len(self.params["layers_dict"]),
                              params["vec_len"]))
        for layer in params["loc_layers"]:
            # first create visual codes
            codes_arr[:, self.params["layers_dict"][layer]] = self._create_item_codes(params["vec_len"],
                                                                            params["num_loc_items"],
                                                                            len(params["categories"]),
                                                                            params["ic_ratio"],)
        # Save the generated codes into representations object
        representations = self.representations.save_representations(codes_arr, params)
        loc_memories = []
        labels = []
        for n in range(params["num_loc_repeats"]):
            for name, code in representations.items():
                loc_memories.append(code)
                labels.append(name.rsplit("_", 1)[0])
        # save to memories object
        self.memories.save_loc_memories(np.asarray(loc_memories), np.asarray(labels))
        return loc_memories

    def initialize_encoding_phase(self, **kwargs):
        if self.params["trial_reset"]:
            self.reset_current_trial()
        rand_init_state = np.random.randn(len(self.params["layers_dict"]),
                                          self.params["vec_len"])
        rand_init_state = rand_init_state / np.linalg.norm(rand_init_state, ord=2)
        self.current_state = np.any(self.current_state) or rand_init_state
        # Add the current state to the list of time steps in current trial
        self.current_trial.append(self.current_state)
        self.encoding_state, self.encoded_item = self._random_image(**kwargs)

        return None

    def get_clf_layer_inds(self):
        return [self.params["layers_dict"][l] for l in self.params["main_layers"]]

    def _get_current_memories(self, source):
        if source == "combined":
            memories = np.concatenate((self.memories.loc_memories,
                                       self.memories.task_memories), axis=0)
        elif source == "loc":
            memories = self.memories.loc_memories
        elif source == "task":
            memories = self.memories.task_memories
        else:
            raise SyntaxError('source not recognized. Must be "combined" or "loc" or "task"')
        return memories

    def plot_current_similarity(self, simtype="scaled"):
        assert simtype in ["raw", "scaled", "cross_layer"], f"simtype {simtype} not recognized"
        fig, ax = plt.subplots()
        x = np.arange(len(self.current_similarity[simtype]))
        ax.bar(x,
               self.current_similarity[simtype],
               )
        plt.axvline(len(self.memories.loc_memories), color='r')
        plt.xlabel("Memory index")
        plt.ylabel("Similarity")
        plt.title("Similarity values across memories")
        plt.show()

    def plot_current_trial(self, **kwargs):
        probas = self.classify_timepoints()
        fig, ax = plt.subplots()
        x = np.arange(probas.shape[1])
        for i, cls in enumerate(probas):
            ax.plot(x, cls, color=self.plot_colors[i], label=self.label_encoder.inverse_transform([i])[0])
        plt.title("Decoding of trial")
        plt.legend()
        plt.show()

    def _random_image(self, category=None, item=None, **kwargs):
        if item is not None:
            assert item in self.representations.representations.keys(), f"{item} does not exist"
            incoming_img = item
        elif category is not None:
            assert category in self.params["categories"], f"{category} is not in current categories"
            cat_img_list = [i for i in self.representations.representations.keys() if category in i]
            incoming_img = np.random.choice(cat_img_list)
        else:
            incoming_img = np.random.choice(list(self.representations.representations.keys()))
        print(f"Incoming image is {incoming_img}")
        return self.representations.representations[incoming_img], incoming_img

    def reset_current_trial(self):
        self.current_trial = []
        self.encoding_state = None
        self.current_state = None
        self.encoded_item = None
        self.replacement_state = None

    def reset_task_memories(self):
        self.memories.task_memories = None

    def simulate_encoding_timestep(self,
                                   incoming_information:np.ndarray=None,
                                   **kwargs):
        if not self.current_trial:
            self.initialize_encoding_phase(**kwargs)
        # Find the incoming visual information
        incoming_information = np.any(incoming_information) or self.encoding_state
        new_state = self.calc_new_state(current_state=self.current_state,
                                        incoming_state=incoming_information)
        # Add the current state to the list of time steps in current trial and save to memories object
        self.current_state = new_state
        self.memories.save_task_memory(self.current_state)
        self.current_trial.append(self.current_state)
        return self.current_state

    def simulate_maintain_timestep(self, variant="retrieval", **kwargs):
        if variant == "retrieval":
            new_state = self._simulate_maintain_timestep_retrieval(**kwargs)
        elif variant == "decay":
            raise NotImplementedError("decay variant not implemented")
        # Now save the info into current trial
        self.current_state = new_state
        self.current_trial.append(new_state)
        self.memories.save_task_memory(new_state)
        return

    def _simulate_maintain_timestep_retrieval(self, source=None, **kwargs):
        """
        Simulates a maintain timestep using the retrieved memory variant
        :return:
        """
        memstack = self._get_current_memories(source or self.params["mem_source"])
        # Calculate the similarity-weighted average vector
        similarity = self.calc_similarity(self.current_state,
                                           self.params["main_layers"],
                                           memstack=memstack)
        incoming_state = self._calc_similarity_weighted_rep(similarity=similarity, memstack=memstack)
        new_state = self.calc_new_state(self.current_state, incoming_state, **kwargs)
        return new_state


    def simulate_replace_timestep(self, category=None, item=None,):
        if self.replacement_state is None:
            # find the current encoded category and choose an image not from this category
            current_encoded_category = self.encoded_item.rsplit("_", 1)[0]
            if item is not None:
                assert item.rsplit("_", 1)[0] != current_encoded_category, f"{item} belongs to current encoded category: {current_encoded_category}"
            elif category is not None:
                assert category != current_encoded_category, f"{category} is the same as current encoded category: {current_encoded_category}"
            else:
                category = np.random.choice([c for c in self.params["categories"] if c != current_encoded_category])
            replace_img, item = self._random_image(category=category, item=item)
            self.replacement_state = replace_img
        else:
            replace_img = self.replacement_state

        # # Zero out all layers except verbal layer
        # verbal_layer_bool = [False if n == self.params["layers_dict"]["verbal"] else True for n in range(len(self.params["layers_dict"])) ]
        # replace_verbal_rep = replace_img.copy()
        # replace_verbal_rep[verbal_layer_bool] = 0
        # use similarity to verbal memory to obtain weighted visual information
        memstack = self._get_current_memories(source="loc")
        similarity = self.calc_similarity(probe=replace_img,
                                          probe_layers="verbal",
                                          memstack=memstack)
        replace_state = self._calc_similarity_weighted_rep(similarity=similarity,
                                                           memstack=memstack)
        # Calculate the new state
        new_state = self.calc_new_state(current_state=self.current_state,
                                        incoming_state=replace_state)
        # Save
        self.current_state = new_state
        self.current_trial.append(new_state)
        self.memories.save_task_memory(new_state)
        return self.current_state













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
        self.memories = None
        self.loc_memories = None
        self.task_memories = None
        print("Memories reset")

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









class Representations(object):
    def __init__(self,):
        self.representations = None
        self.labels = None

    def save_representations(self, codes_arr, params, save_to_self=True):
        # add some helper
        item_per_cat = params["num_loc_items"] // len(params["categories"])
        num_categories = len(params["categories"])
        code_nums = [c for c in range(item_per_cat) for n in range(3)]
        # Check that the number of items are divisible by number of categories
        assert codes_arr.shape[0] % num_categories == 0, "number of unique items must be divisible by num_categories"
        # Save the representations into a dictionary format
        representations = {}
        for i, code in enumerate(codes_arr):
            representations[f"{params["categories"][i % num_categories]}_{code_nums[i]}"] = code
        # save the representations to self
        if save_to_self:
            self.representations = representations
        return representations


