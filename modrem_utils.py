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
                      "ic_ratio": 1,
                      "num_loc_repeats": 5,
                      "beta": 0.65,
                      "trial_reset": True,
                      }
    plot_colors = ["orange", "blue", "purple"]

    def __init__(self, params):
        params = params or {}
        self.params = self.default_params | params

        self.memories = Memories()
        self.representations = Representations()
        self.current_state = None
        self.incoming_state = None
        self.clf = None
        self.current_trial = []

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

    def create_loc_memories(self, params: dict = None):
        if params is None:
            params = self.params
        # first create visual codes
        visual_codes = self._create_item_codes(params["vec_len"],
                                               params["num_loc_items"],
                                               len(params["categories"]),
                                               params["ic_ratio"], )
        # then create verbal codes
        verbal_codes = self._create_item_codes(params["vec_len"],
                                               params["num_loc_items"],
                                               len(params["categories"]),
                                               params["ic_ratio"], )
        # Save the generated codes into representations object
        representations = self.representations.save_representations(visual_codes, verbal_codes, params["categories"])
        loc_memories = []
        labels = []
        for n in range(params["num_loc_repeats"]):
            for name, code in representations.items():
                loc_memories.append(code)
                labels.append(name.rsplit("_", 1)[0])
        # save to memories object
        self.memories.save_loc_memories(np.asarray(loc_memories), np.asarray(labels))
        return loc_memories

    def initialize_encoding_phase(self):
        if self.params["trial_reset"]:
            rand_init_state = np.random.randn(self.params["vec_len"])
            self.current_state = rand_init_state / np.linalg.norm(rand_init_state, ord=2)
        else:
            self.current_state = self.current_state or np.random.randn(self.params["vec_len"])
        # Add the current state to the list of time steps in current trial
        self.current_trial.append(self.current_state)
        return None

    def simulate_encoding_timestep(self,
                                   incoming_visual_information:np.ndarray=None,
                                   **kwargs):
        if not self.current_trial:
            self.initialize_encoding_phase()
        # Find the incoming visual information
        incoming_visual_information = incoming_visual_information or self._find_incoming_visual_information(**kwargs)
        # Calculate Rho
        curr_in_similarity = np.dot(self.current_state, incoming_visual_information)
        beta = self.params["beta"]
        rho = np.sqrt(1 + (beta ** 2) * (curr_in_similarity ** 2 - 1)) - beta * curr_in_similarity
        # calculate new current state
        self.current_state = rho * self.current_state + beta * incoming_visual_information
        # Add the current state to the list of time steps in current trial
        self.current_trial.append(self.current_state)
        return self.current_state

    def _find_incoming_visual_information(self, category=None, item=None, **kwargs):
        if self.incoming_state is None:
            if item is not None:
                assert item in self.representations.representations.keys(), f"{item} does not exist"
                incoming_img = item
            elif category is not None:
                assert category in self.params["categories"], f"{category} is not in current categories"
                cat_img_list = [i for i in self.representations.representations.keys() if category in i]
                incoming_img = np.random.choice(cat_img_list)
            else:
                print("No incoming state has been initialized. Using random image")
                incoming_img = np.random.choice(list(self.representations.representations.keys()))
            print(f"Incoming visual representation is {incoming_img}")
            # return just the visual information
            self.incoming_state = self.representations.representations[incoming_img][0]  # pull just visual info
        return self.incoming_state

    def train_classifier(self):
        """
        Trains a classifier on the localizer memories
        :return: classifier object
        """
        # First initiate a CV object
        test_fold = [n for n in range(self.params["num_loc_repeats"]) for i in range(self.params["num_loc_items"])]
        cv = PredefinedSplit(test_fold)
        # Pull the data
        X = self.memories.loc_memories[:, 0]    # only extract visual info
        y = self.memories.loc_labels
        accuracies = np.zeros(cv.get_n_splits())
        for i, (train_idx, test_idx) in enumerate(cv.split()):
            clf = OneVsRestClassifier(LogisticRegression(penalty="l2", solver="liblinear", C=1))
            accuracies[i] = clf.fit(X[train_idx], y[train_idx]).score(X[test_idx], y[test_idx])
        print(f"Accuracy of classifier was {np.mean(accuracies)}. Training classifier on all loc")
        self.clf = OneVsRestClassifier(LogisticRegression(penalty="l2", solver="liblinear", C=1)).fit(X, y)
        return self.clf

    def classify_timepoints(self):
        clf = self.clf
        probas = np.zeros((len(self.params["categories"]), len(self.current_trial)))
        for t, t_step in enumerate(self.current_trial):
            probas[:, t] = clf.predict_proba(t_step.reshape(1, -1))
        return probas

    def graph_current_trial(self):
        probas = self.classify_timepoints()
        fig, ax = plt.subplots()
        x = np.arange(probas.shape[1])
        for i, cls in enumerate(probas):
            ax.plot(x, cls, color=self.plot_colors[i], label=self.params["categories"][i])
        plt.title("Decoding of trial")
        plt.legend()
        plt.show()

    def reset_current_trial(self):
        self.current_trial = []
        self.incoming_state = None




class Memories(object):

    def __init__(self):
        self.memories = None
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

    def save_loc_memories(self, loc_memories, labels):
        self.loc_memories = loc_memories
        # Also generate the labels for the loc memories
        self.loc_labels = labels
        if self.memories is None:
            self.memories = loc_memories
        return None

    def reset_memories(self):
        self.memories = None
        self.loc_memories = None
        self.task_memories = None
        print("Memories reset")




class Representations(object):
    def __init__(self,):
        self.representations = None
        self.labels = None

    def save_representations(self, visual_codes, verbal_codes, categories, save_to_self=True):
        # Check that visual and verbal codes contain the same number of items
        assert visual_codes.shape[0] == verbal_codes.shape[0], "visual_codes and verbal_codes must have same length"
        # add some helper
        item_per_cat = len(visual_codes) // len(categories)
        num_categories = len(categories)
        code_nums = [c for c in range(item_per_cat) for n in range(3)]
        # Check that the number of items are divisible by number of categories
        assert visual_codes.shape[0] % num_categories == 0, "number of unique items must be divisible by num_categories"
        # Save the representations into a dictionary format
        representations = {}
        for i, vis_code in enumerate(visual_codes):
            representations[f"{categories[i % num_categories]}_{code_nums[i]}"] = np.stack((vis_code, verbal_codes[i]), axis=0)
        # save the representations to self
        if save_to_self:
            self.representations = representations
        return representations


