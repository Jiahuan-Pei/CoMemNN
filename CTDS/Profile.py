from typing import List
import torch

_keys = ['gender', 'age', 'dietary', 'favorite']

_attr_to_index = {
    'gender': {'female': 0, 'male': 1},
    'age': {'elderly': 0, 'middle-aged': 1, 'young': 2},
    'dietary': {'non-veg': 0, 'veg': 1},
    'favorite':
        {'biryani': 0, 'curry': 1, 'english_breakfast': 2, 'fish_and_chips': 3, 'omlette': 4,
         'paella': 5, 'pasta': 6, 'pizza': 7, 'ratatouille': 8, 'risotto': 9, 'shepherds_pie': 10,
         'souffle': 11, 'tapas': 12, 'tart': 13, 'tikka': 14}
}
def vector2profile(vector):
    profile_list = []
    pointer = 0
    # print(vector)
    for i, key in enumerate(_keys):
        for value in _attr_to_index[key]:
            idx = pointer+_attr_to_index[key][value]
            if vector[idx] == 1:
                profile_list.append(value)
                # print(key, value, idx)
        pointer += len(_attr_to_index[key])
    return profile_list

def vector2probdict(vector):
    # profile_prob_dict = {}
    profile_prob_dict = []
    pointer = 0
    for i, key in enumerate(_keys):
        for k in _attr_to_index[key]:
            idx = pointer+_attr_to_index[key][k]
            profile_prob_dict.append((k, vector[idx].item()))
        pointer += len(_attr_to_index[key])
    return profile_prob_dict


def pred_profile_to_onehot_vec(pred_profile):
    profile_len = pred_profile.size(1)
    v = torch.zeros_like(pred_profile)
    pointer = 0
    keys = _keys[:2] if profile_len<=5 else _keys
    for i, key in enumerate(keys):
        steps = len(_attr_to_index[key])
        v[:, pointer:pointer+steps][:, pred_profile[:, pointer:pointer+steps].argmax(dim=1)] = 1
        pointer += steps
    return v

def _vectorize(profile: dict):
    n_attributes = len(profile)
    result = []

    for key in _keys[:n_attributes]:
        one_hot_emb = [0] * len(_attr_to_index[key])
        value = profile[key]
        one_hot_emb[_attr_to_index[key][value]] = 1
        result += one_hot_emb[:]

    return result

def _get_profile_id(profile: dict):
    gender = profile['gender']
    age = profile['age']
    gender_size = len(_attr_to_index['gender'])

    result = _attr_to_index['gender'][gender]
    result += _attr_to_index['age'][age] * gender_size

    return result

class Profile:
    # genders = ['male', 'female']
    # ages = ['young', 'middle-aged', 'elderly']

    def __init__(self, attribute_values):
        """
        The :class:`Profile` represents a user profile.

        Parameters
        ----------
        attribute_values : list of str
            The list of profile attribute values.
            i.e., ['male', 'middle-aged', 'non-veg', 'fish_and_chips']
        """
        n_attributes = len(attribute_values)
        self.profile = dict(zip(_keys[:n_attributes], attribute_values))
        # for k, v in profile.items():
        #     self.__setattr__(k, v)
        self.vector: List[int] = _vectorize(self.profile)
        self.id = _get_profile_id(self.profile)

    def profile2vector(self, attribute_values):
        n_attributes = len(attribute_values)
        profile = dict(zip(_keys[:n_attributes], attribute_values))
        for k, v in profile.items():
            self.__setattr__(k, v)
        self.vector: List[int] = _vectorize(profile)
        return self.vector
