import typing
from typing import Literal, Callable
import numpy as np
import itertools
import re
from effector import helpers, utils


BIG_M = helpers.BIG_M


class Regions:
    def __init__(
        self,
        feature: int,
        heter_func: Callable,
        data: np.ndarray,
        data_effect: typing.Union[None, np.ndarray],
        feature_types: typing.Union[list, None],
        feature_names: typing.List[str],
        target_name: str,
        categorical_limit: int = 10,
        candidate_conditioning_features: typing.Union[Literal["all"], list] = "all",
        min_points_per_subregion: int = 10,
        nof_candidate_splits_for_numerical=20,
        max_split_levels=2,
        heter_pcg_drop_thres=0.1,
        heter_small_enough=0.1,
        split_categorical_features=False,
        regions_check: int = -1,
        global_drop: bool = False,
        early_stop: bool = True,
    ):
        # setters
        self.feature = feature
        self.data = data
        self.dim = self.data.shape[1]
        self.cat_limit = categorical_limit
        self.data_effect = data_effect
        self.feature_names = feature_names
        self.target_name = target_name
        self.min_points = min_points_per_subregion
        self.heter_func = heter_func
        self.nof_candidate_splits_for_numerical = nof_candidate_splits_for_numerical
        self.max_split_levels = max_split_levels
        self.heter_pcg_drop_thres = heter_pcg_drop_thres
        self.heter_small_enough = heter_small_enough
        self.split_categorical_features = split_categorical_features
        self.regions_check = regions_check

        self.foi = self.feature
        self.foc = (
            [i for i in range(self.dim) if i != self.feature]
            if candidate_conditioning_features == "all"
            else candidate_conditioning_features
        )

        # on-init
        self.feature_types = (
            utils.get_feature_types(data, categorical_limit)
            if feature_types is None
            else feature_types
        )
        self.foc_types = [self.feature_types[i] for i in self.foc]

        # init method args
        self.method_args = {}

        # init splits
        self.splits: dict = {}
        self.important_splits: dict = {}

        self.splits_tree: typing.Union[Tree, None] = None
        self.important_splits_tree: typing.Union[Tree, None] = None

        # state variable
        self.split_found: bool = False
        self.important_splits_selected: bool = False

        self.global_drop = global_drop
        self.early_stop = early_stop

    def search_all_splits(self):
        """
        Iterate over all features of conditioning and choose the best split for each level in a greedy fashion.
        """
        if (
            self.feature_types[self.feature] == "cat"
            and not self.split_categorical_features
        ):
            self.splits = []
        else:
            assert self.max_split_levels <= len(
                self.foc
            ), "nof_levels must be smaller than len(foc)"

            # initialize heterogeneity
            heter_init = (
                self.heter_func(self.data, self.data_effect)
                if self.data_effect is not None
                else self.heter_func(self.data)
            )

            # initialize x_list, x_jac_list, splits
            x_list = [self.data]
            x_jac_list = [self.data_effect] if self.data_effect is not None else None
            splits = [
                {
                    "heterogeneity": [heter_init],
                    "weighted_heter": heter_init,
                    "nof_instances": [len(self.data)],
                    "split_i": -1,
                    "split_j": -1,
                    "foc": self.foc.copy(),
                }
            ]

            level_foc = self.foc
            level_foc_types = self.foc_types
            for lev in range(self.max_split_levels):
                # if any subregion has less than min_points, stop
                if any([len(x) < self.min_points for x in x_list]):
                    print("less than min_points, breaking")
                    break

                # find optimal split

                split = self.single_level_splits(
                    x_list,
                    x_jac_list,
                    (
                        [heter_init] * len(x_list)
                        if self.global_drop
                        else splits[-1]["heterogeneity"]
                    ),
                    regions_check=self.regions_check,
                    foc=level_foc.copy(),
                    foc_types=level_foc_types.copy(),
                )
                splits.append(split)

                # split data and data_effect based on the optimal split found above
                feat, pos, typ = split["feature"], split["position"], split["type"]

                # find and delete from foc, foc_types the selected feature
                feat_idx = level_foc.index(feat)
                level_foc.pop(feat_idx)
                level_foc_types.pop(feat_idx)

                if x_jac_list is not None:
                    if typ == "cat" or type(pos) != tuple:
                        x_jac_list = self.flatten_list(
                            [
                                self.split_dataset(x, x_jac, feat, pos, typ)
                                for x, x_jac in zip(x_list, x_jac_list)
                            ]
                        )
                    else:
                        x_jac_list = self.flatten_list(
                            [
                                self.split_dataset_2(x, x_jac, feat, pos[0], pos[1])
                                for x, x_jac in zip(x_list, x_jac_list)
                            ]
                        )
                if typ == "cat" or type(pos) != tuple:
                    x_list = self.flatten_list(
                        [self.split_dataset(x, None, feat, pos, typ) for x in x_list]
                    )
                else:
                    x_list = self.flatten_list(
                        [
                            self.split_dataset_2(x, None, feat, pos[0], pos[1])
                            for x in x_list
                        ]
                    )

                self.splits = splits
                if self.early_stop and self.valid_level_split(lev + 1) == 0:
                    break

        # update state
        self.split_found = True
        return self.splits

    def valid_level_split(self, level):
        # level is 0 - base where 0 is the initial dataset
        assert level > 0, "level must be greater than 0"

        if len(self.splits) == 0 or self.splits[0]["weighted_heter"] == BIG_M:
            return False
        else:
            if self.global_drop:
                prev_heter = self.splits[0]["weighted_heter"]
            else:
                prev_heter = self.splits[level - 1]["weighted_heter"]

            cur_heter = self.splits[level]["weighted_heter"]
            heter_drop = (prev_heter - cur_heter) / prev_heter
            if heter_drop > self.heter_pcg_drop_thres:
                return True
            else:
                return False

    def single_level_splits(
        self,
        x_list: list,
        x_jac_list: typing.Union[list, None],
        heter_before: list,
        regions_check: int = -1,
        foc: list = None,
        foc_types: list = None,
    ):
        """Find all splits for a single level.
        regions_check = -1 -> checks 2 regions
        regions_check = 0 -> checks 2 and 3 regions
        regions_check = 1 -> checks 3 regions
        """
        nof_splits = self.nof_candidate_splits_for_numerical
        heter_func = self.heter_func
        cat_limit = self.cat_limit

        data = self.data

        big_M = -BIG_M
        # weighted_heter_drop[i,j,k] (i index of foc, j index of 1st split position, k index of 2nd split position). it is
        # the accumulated heterogeneity drop if I split foc[i] at index j and at index k (j < k)

        weighted_heter_drop_2 = np.ones([len(foc), max(nof_splits, cat_limit)]) * big_M

        weighted_heter_drop_3 = (
            np.ones([len(foc), max(nof_splits, cat_limit), max(nof_splits, cat_limit)])
            * big_M
        )

        # weighted_heter[i,j] (i index of foc, j index of first split position, k index of second split position) is
        # the accumulated heterogeneity if I split foc[i] at index j and k
        weighted_heter_2 = np.ones([len(foc), max(nof_splits, cat_limit)]) * big_M

        weighted_heter_3 = (
            np.ones([len(foc), max(nof_splits, cat_limit), max(nof_splits, cat_limit)])
            * big_M
        )
        # list with len(foc) elements
        # each element is a list with the split positions for the corresponding feature of conditioning
        candidate_split_positions = [
            (
                self.find_positions_cat(data, foc_i)
                if foc_types[i] == "cat"
                else self.find_positions_cont(data, foc_i, nof_splits)
            )
            for i, foc_i in enumerate(foc)
        ]

        # exhaustive search on all split positions
        for i, foc_i in enumerate(foc):
            if foc_types[i] == "cat" or regions_check <= 0:
                for j, position in enumerate(candidate_split_positions[i]):
                    x_list_2 = self.flatten_list(
                        [
                            self.split_dataset(x, None, foc_i, position, foc_types[i])
                            for x in x_list
                        ]
                    )
                    if x_jac_list is not None:
                        x_jac_list_2 = self.flatten_list(
                            [
                                self.split_dataset(
                                    x, x_jac, foc_i, position, foc_types[i]
                                )
                                for x, x_jac in zip(x_list, x_jac_list)
                            ]
                        )

                    # sub_heter: list with the heterogeneity after split of foc_i at position j
                    if x_jac_list is None:
                        sub_heter = [heter_func(x) for x in x_list_2]
                    else:
                        sub_heter = [
                            heter_func(x, x_jac)
                            for x, x_jac in zip(x_list_2, x_jac_list_2)
                        ]

                    # heter_drop: list with the heterogeneity drop after split of foc_i at position j

                    heter_drop = np.array(
                        self.flatten_list(
                            [
                                [
                                    heter_bef - sub_heter[int(2 * i)],
                                    heter_bef - sub_heter[int(2 * i + 1)],
                                ]
                                for i, heter_bef in enumerate(heter_before)
                            ]
                        )
                    )
                    # populations: list with the number of instances in each dataset after split of foc_i at position j
                    populations = np.array([len(xx) for xx in x_list_2])
                    # weights analogous to the populations in each split
                    weights = populations / np.sum(populations)

                    weighted_heter_drop_2[i, j] = np.sum(heter_drop * weights)
                    weighted_heter_2[i, j] = np.sum(weights * np.array(sub_heter))

            if foc_types[i] != "cat" and regions_check >= 0:
                for j, position1 in enumerate(candidate_split_positions[i]):
                    for k, position2 in enumerate(
                        candidate_split_positions[i][j + 1 :]
                    ):

                        # split datasets
                        x_list_2 = self.flatten_list(
                            [
                                self.split_dataset_2(
                                    x, None, foc_i, position1, position2
                                )
                                for x in x_list
                            ]
                        )
                        if x_jac_list is not None:
                            x_jac_list_2 = self.flatten_list(
                                [
                                    self.split_dataset_2(
                                        x, x_jac, foc_i, position1, position2
                                    )
                                    for x, x_jac in zip(x_list, x_jac_list)
                                ]
                            )

                        # sub_heter: list with the heterogeneity after split of foc_i at position j and k
                        if x_jac_list is None:
                            sub_heter = [heter_func(x) for x in x_list_2]
                        else:
                            sub_heter = [
                                heter_func(x, x_jac)
                                for x, x_jac in zip(x_list_2, x_jac_list_2)
                            ]

                        # heter_drop: list with the heterogeneity drop after split of foc_i at position j and k
                        heter_drop = np.array(
                            self.flatten_list(
                                [
                                    [
                                        heter_bef - sub_heter[int(3 * i)],
                                        heter_bef - sub_heter[int(3 * i + 1)],
                                        heter_bef - sub_heter[int(3 * i + 2)],
                                    ]
                                    for i, heter_bef in enumerate(heter_before)
                                ]
                            )
                        )
                        # populations: list with the number of instances in each dataset after split of foc_i at position j
                        populations = np.array([len(xx) for xx in x_list_2])
                        # weights analogous to the populations in each split
                        weights = populations / np.sum(populations)
                        pos_2_idx = j + k + 1
                        weighted_heter_drop_3[i, j, pos_2_idx] = np.sum(
                            heter_drop * weights
                        )
                        weighted_heter_3[i, j, pos_2_idx] = np.sum(
                            weights * np.array(sub_heter)
                        )

        # find the split with the largest weighted heterogeneity drop
        if regions_check <= 0 or "cat" in foc_types:
            i, j = np.unravel_index(
                np.argmax(weighted_heter_drop_2),
                weighted_heter_drop_2.shape,
            )
            drop2 = weighted_heter_drop_2[i, j]

        if regions_check >= 0:
            ii, jj, kk = np.unravel_index(
                np.argmax(weighted_heter_drop_3),
                weighted_heter_drop_3.shape,
            )
            drop3 = weighted_heter_drop_3[ii, jj, kk]

        # if drop3 == big_M -> all foc features are categorical
        if (
            regions_check == -1
            or drop3 == big_M
            or (regions_check == 0 and drop2 >= drop3)
        ):
            feature = foc[i]
            position = candidate_split_positions[i][j]
            split_positions = candidate_split_positions[i]

            # how many instances in each dataset after the min split
            x_list_2 = self.flatten_list(
                [
                    self.split_dataset(x, None, foc[i], position, foc_types[i])
                    for x in x_list
                ]
            )

            nof_instances = [len(x) for x in x_list_2]
            if x_jac_list is None:
                sub_heter = [heter_func(x) for x in x_list_2]
            else:
                x_jac_list_2 = self.flatten_list(
                    [
                        self.split_dataset(x, x_jac, foc[i], position, foc_types[i])
                        for x, x_jac in zip(x_list, x_jac_list)
                    ]
                )

                sub_heter = [
                    heter_func(x, x_jac) for x, x_jac in zip(x_list_2, x_jac_list_2)
                ]

            init_weghted_heter_2 = weighted_heter_drop_2[i, j] + weighted_heter_2[i, j]
            weighted_heter_drop_ratio_2 = (
                weighted_heter_drop_2[i, j] / init_weghted_heter_2
                if init_weghted_heter_2 != 0
                else 0
            )

            split_2 = {
                "feature": feature,
                "position": position,
                "range": [np.min(data[:, feature]), np.max(data[:, feature])],
                "candidate_split_positions": list(split_positions),
                "nof_instances": nof_instances,
                "type": foc_types[i],
                "heterogeneity": sub_heter,
                "split_i": i,
                "split_j": j,
                "foc": foc,
                "weighted_heter_drop": weighted_heter_drop_2[i, j],
                "weighted_heter_drop_ratio": weighted_heter_drop_ratio_2,
                "weighted_heter": weighted_heter_2[i, j],
            }
            return split_2
        else:
            feature = foc[ii]
            position1 = candidate_split_positions[ii][jj]
            position2 = candidate_split_positions[ii][kk]

            split_positions = candidate_split_positions[ii]

            # how many instances in each dataset after the min split
            x_list_2 = self.flatten_list(
                [
                    self.split_dataset_2(x, None, foc[ii], position1, position2)
                    for x in x_list
                ]
            )

            nof_instances = [len(x) for x in x_list_2]
            if x_jac_list is None:
                sub_heter = [heter_func(x) for x in x_list_2]
            else:
                x_jac_list_2 = self.flatten_list(
                    [
                        self.split_dataset_2(x, x_jac, foc[ii], position1, position2)
                        for x, x_jac in zip(x_list, x_jac_list)
                    ]
                )

                sub_heter = [
                    heter_func(x, x_jac) for x, x_jac in zip(x_list_2, x_jac_list_2)
                ]

            init_weghted_heter_3 = (
                weighted_heter_drop_3[ii, jj, kk] + weighted_heter_3[ii, jj, kk]
            )
            weighted_heter_drop_ratio_3 = (
                weighted_heter_drop_3[ii, jj, kk] / init_weghted_heter_3
                if init_weghted_heter_3 != 0
                else 0
            )

            split_3 = {
                "feature": feature,
                "position": (position1, position2),
                "range": [np.min(data[:, feature]), np.max(data[:, feature])],
                "candidate_split_positions": list(split_positions),
                "nof_instances": nof_instances,
                "type": foc_types[ii],
                "heterogeneity": sub_heter,
                "split_i": ii,
                "split_j": (jj, kk),
                "foc": foc,
                "weighted_heter_drop": weighted_heter_drop_3[ii, jj, kk],
                "weighted_heter_drop_ratio": weighted_heter_drop_ratio_3,
                "weighted_heter": weighted_heter_3[ii, jj, kk],
            }
            return split_3

    def choose_important_splits(self):
        assert self.split_found, "No splits found for feature {}".format(self.feature)

        # if split is empty, skip
        if len(self.splits) == 0:
            optimal_splits = {}
        # if initial heterogeneity is BIG_M, skip
        elif self.splits[0]["weighted_heter"] == BIG_M:
            optimal_splits = {}
        # if initial heterogeneity is small right from the beginning, skip
        elif self.splits[0]["weighted_heter"] < self.heter_small_enough:
            optimal_splits = {}
        else:
            splits = self.splits

            # accept split if heterogeneity drops over 20%
            heter = np.array([splits[i]["weighted_heter"] for i in range(len(splits))])

            if self.global_drop:
                heter_drop = (heter[0] - heter[1:]) / heter[0]
            else:
                heter_drop = (heter[:-1] - heter[1:]) / heter[:-1]

            split_valid = heter_drop > self.heter_pcg_drop_thres

            # if all are negative, return nothing
            if np.sum(split_valid) == 0:
                optimal_splits = {}
            # if all are positive, return all
            elif np.sum(split_valid) == len(split_valid):
                optimal_splits = splits[1:]
            else:
                if self.early_stop:
                    # find first negative split
                    first_negative = np.where(split_valid == False)[0][0]

                    # if first negative is the first split, return nothing
                    if first_negative == 0:
                        optimal_splits = {}
                    else:
                        optimal_splits = splits[1 : first_negative + 1]
                else:
                    # find last negative split
                    max_valid_idx = np.where(split_valid)[0].max()
                    # return all splits up to the last negative
                    optimal_splits = splits[1 : max_valid_idx + 2]

        # update state variable
        self.important_splits_selected = True
        self.important_splits = optimal_splits
        return optimal_splits

    def split_dataset(self, x, x_jac, feature, position, feat_type):
        if feat_type == "cat":
            ind_1 = x[:, feature] == position
            ind_2 = x[:, feature] != position
        else:
            ind_1 = x[:, feature] < position
            ind_2 = x[:, feature] >= position
        if x_jac is None:
            X1 = x[ind_1, :]
            X2 = x[ind_2, :]
        else:
            X1 = x_jac[ind_1, :]
            X2 = x_jac[ind_2, :]

        return X1, X2

    def split_dataset_2(self, x, x_jac, feature, position1, position2):
        ind_1 = x[:, feature] < position1
        ind_2 = (x[:, feature] >= position1) & (x[:, feature] <= position2)
        ind_3 = x[:, feature] > position2

        if x_jac is None:
            X1 = x[ind_1, :]
            X2 = x[ind_2, :]
            X3 = x[ind_3, :]
        else:
            X1 = x_jac[ind_1, :]
            X2 = x_jac[ind_2, :]
            X3 = x_jac[ind_3, :]

        return X1, X2, X3

    def find_positions_cat(self, x, feature):
        return np.unique(x[:, feature])

    def find_positions_cont(self, x, feature, nof_splits):
        step = (np.max(x[:, feature]) - np.min(x[:, feature])) / nof_splits
        return np.min(x[:, feature]) + (np.arange(nof_splits) + 0.5) * step

    def flatten_list(self, l):
        return [item for sublist in l for item in sublist]

    def splits_to_tree(self, only_important=False, scale_x_list=None):
        if len(self.splits) == 0:
            return None

        nof_instances = self.splits[0]["nof_instances"][0]
        tree = Tree()
        # format with two decimals
        data = {
            "heterogeneity": self.splits[0]["heterogeneity"][0],
            "feature_name": self.feature,
            "nof_instances": self.splits[0]["nof_instances"][0],
            "data": self.data,
            "data_effect": self.data_effect,
            "weight": 1.0,
        }

        feature_name = self.feature_names[self.feature]
        tree.add_node(feature_name, None, data=data, level=0)
        parent_level_nodes = [feature_name]
        parent_level_data = [self.data]
        parent_level_data_effect = [self.data_effect]
        splits = self.important_splits if only_important else self.splits[1:]
        prev_nodes_added = 1
        prev_nodes_start_idx = 0

        for i, split in enumerate(splits):

            # nof nodes to add
            nodes_to_add = len(split["nof_instances"])

            # how many regions are created by the split
            cur_split_n_regions = nodes_to_add / prev_nodes_added

            # find parent
            for j in range(nodes_to_add):
                parent_idx = prev_nodes_start_idx + int(j / cur_split_n_regions)
                parent_name = parent_level_nodes[parent_idx]
                parent_data = parent_level_data[parent_idx]
                parent_data_effect = parent_level_data_effect[parent_idx]

                # prepare data

                foc_name = self.feature_names[split["feature"]]
                foc = split["feature"]

                if cur_split_n_regions == 2:
                    pos = split["position"]
                    pos_text = (
                        pos.round(2)
                        if scale_x_list is None
                        else (
                            scale_x_list[foc]["std"] * pos + scale_x_list[foc]["mean"]
                        ).round(2)
                    )
                    data_1, data_2 = self.split_dataset(
                        parent_data, None, foc, pos, split["type"]
                    )
                    if self.data_effect is not None:
                        data_effect_1, data_effect_2 = self.split_dataset(
                            parent_data, parent_data_effect, foc, pos, split["type"]
                        )
                    else:
                        data_effect_1, data_effect_2 = None, None

                    data_new = data_1 if j % 2 == 0 else data_2
                    data_effect_new = data_effect_1 if j % 2 == 0 else data_effect_2
                    comparison = "<=" if j % 2 == 0 else ">"
                    name = f"{foc_name} {comparison} {pos_text}"
                elif cur_split_n_regions == 3:
                    pos1, pos2 = split["position"]
                    pos1_scaled = (
                        pos1.round(2)
                        if scale_x_list is None
                        else (
                            scale_x_list[foc]["std"] * pos1 + scale_x_list[foc]["mean"]
                        ).round(2)
                    )
                    pos2_scaled = (
                        pos2.round(2)
                        if scale_x_list is None
                        else (
                            scale_x_list[foc]["std"] * pos2 + scale_x_list[foc]["mean"]
                        ).round(2)
                    )
                    data_1, data_2, data_3 = self.split_dataset_2(
                        parent_data, None, foc, pos1, pos2
                    )
                    if self.data_effect is not None:
                        data_effect_1, data_effect_2, data_effect_3 = (
                            self.split_dataset_2(
                                parent_data, parent_data_effect, foc, pos1, pos2
                            )
                        )
                    else:
                        data_effect_1, data_effect_2, data_effect_3 = None, None, None

                    if j % 3 == 0:
                        data_new, data_effect_new, comparison = (
                            data_1,
                            data_effect_1,
                            f"< {pos1_scaled}",
                        )
                    elif j % 3 == 1:
                        data_new, data_effect_new, comparison = (
                            data_2,
                            data_effect_2,
                            f">= {pos1_scaled} and <= {pos2_scaled}",
                        )
                    elif j % 3 == 2:
                        data_new, data_effect_new, comparison = (
                            data_3,
                            data_effect_3,
                            f"> {pos2_scaled}",
                        )
                    else:
                        raise ValueError("j % 3 must be 0, 1 or 2")
                    name = f"{foc_name} {comparison}"
                else:
                    raise ValueError("cur_split_n_regions must be 2 or 3")

                name = (
                    parent_name + " | " + name
                    if nodes_to_add in [2, 3]
                    else parent_name + " and " + name
                )

                data = {
                    "heterogeneity": split["heterogeneity"][j],
                    "weight": float(data_new.shape[0]) / nof_instances,
                    "position": split["position"],
                    "feature": split["feature"],
                    "feature_type": split["type"],
                    "range": split["range"],
                    "candidate_split_positions": split["candidate_split_positions"],
                    "nof_instances": split["nof_instances"][j],
                    "data": data_new,
                    "data_effect": data_effect_new,
                    "comparison": comparison,
                }

                tree.add_node(name, parent_name=parent_name, data=data, level=i + 1)

                parent_level_nodes.append(name)
                parent_level_data.append(data_new)
                parent_level_data_effect.append(data_effect_new)

            prev_nodes_added = nodes_to_add
            prev_nodes_start_idx = len(parent_level_nodes) - nodes_to_add
        return tree


class Node:
    def __init__(self, idx, name, parent_node, data, level):
        self.idx = idx
        self.name = name
        self.parent_node = parent_node
        self.data = data
        self.level = level

        self.heterogeneity = data["heterogeneity"]
        self.weight = data["weight"]
        self.nof_instances = data["nof_instances"]

        self.foc = data["feature"] if "feature" in data else None
        self.foc_type = data["feature_type"] if "feature_type" in data else None
        self.foc_position = data["position"] if "position" in data else None
        self.comparison = data["comparison"] if "comparison" in data else None
        self.candidate_split_positions = (
            data["candidate_split_positions"]
            if "candidate_split_positions" in data
            else None
        )
        self.range = data["range"] if "range" in data else None

    def show(self, show_data=False):
        print("Node id: ", self.idx)
        print("name: ", self.name)
        print(
            "parent name: ",
            self.parent_node.name if self.parent_node is not None else None,
        )
        print("level: ", self.level)

        print("heterogeneity: ", self.heterogeneity)
        print("weight: ", self.weight)
        print("nof_instances: ", self.nof_instances)

        print("foc: ", self.foc)
        print("foc_type: ", self.foc_type)
        print("foc_position: ", self.foc_position)
        print("comparison: ", self.comparison)

        print("data: ", self.data) if show_data else None


class Tree:
    def __init__(self):
        self.nodes = []
        self.idx = 0

    def rename_nodes(self, scale_x_per_feature):
        nodes = self.nodes

        for node in nodes:
            node.name = self._rename_node(node.name, scale_x_per_feature)

    def _rename_node(self, name, scale):
        pattern = r"(x_\d+) (==|!=|<=|<|>=|>) ([\d\.]+)"
        for match in re.finditer(pattern, name):
            var_name = match.group(1)
            symb = match.group(2)
            orig_value = match.group(3)
            feat_i = int(var_name[2:])
            std = scale["feature_" + str(feat_i)]["std"]
            mean = scale["feature_" + str(feat_i)]["mean"]
            scaled_value = std * float(orig_value) + mean
            new_string = name.replace(orig_value, str(scaled_value))
            name = new_string
        return name

    def add_node(self, name, parent_name, data, level):
        if parent_name is None:
            parent_node = None
        else:
            assert parent_name in [node.name for node in self.nodes]
            parent_node = self.get_node(parent_name)

        idx = self.idx
        self.idx += 1
        node = Node(idx, name, parent_node, data, level)
        self.nodes.append(node)

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_level_nodes(self, level):
        nodes = []
        for node_i in self.nodes:
            if node_i.level == level:
                nodes.append(node_i)
        return nodes

    def get_root(self):
        for node in self.nodes:
            if node.parent_node is None:
                return node
        raise ValueError("Root node not found.")

    def get_children(self, name):
        parent_node = self.get_node(name)
        return [node for node in self.nodes if node.parent_node == parent_node]

    def get_level_stats(self, level):
        level_nodes = self.get_level_nodes(level)
        nof_instances = self.get_root().data["nof_instances"]

        w_heter = 0
        for nod in level_nodes:
            w_heter += nod.data["heterogeneity"] * nod.data["weight"]

        return {"heterogeneity": w_heter}

    def show_full_tree(self, node=None):
        if node is None:
            node = self.get_root()

        indent = node.level * 2
        print(
            "    " * indent
            + "Node id: %d, name: %s, heter: %.2f || nof_instances: %5d || weight: %.2f"
            % (
                node.idx,
                node.name,
                node.data["heterogeneity"],
                node.data["nof_instances"],
                node.data["weight"],
            )
        )
        children = self.get_children(node.name)
        for child in children:
            self.show_full_tree(child)

    def show_level_stats(self, node=None):
        max_level = max([node.level for node in self.nodes])
        prev_heter = 0
        for lev in range(max_level + 1):
            level_stats = self.get_level_stats(lev)
            if lev == 0:
                print(
                    "    " * lev * 2
                    + "Level %.d, heter: %.2f" % (lev, level_stats["heterogeneity"])
                )
            else:
                print(
                    "    " * lev * 2
                    + "Level %.d, heter: %.2f || heter drop: %.2f (%.2f%%)"
                    % (
                        lev,
                        level_stats["heterogeneity"],
                        prev_heter - level_stats["heterogeneity"],
                        100 * (prev_heter - level_stats["heterogeneity"]) / prev_heter,
                    )
                )
            prev_heter = level_stats["heterogeneity"]


class DataTransformer:
    def __init__(self, splits: typing.Dict):
        self.splits = splits

    def transform(self, X):
        # feat_mapping <- to how many features each feature is mapped
        feat_mapping = []
        for split in self.splits.values():
            if len(split) == 0:
                feat_mapping.append(1)
            else:
                feat_mapping.append(2 ** len(split))

        # the enhanced data, without masking
        new_data = []
        for i in range(X.shape[1]):
            new_data.append(np.repeat(X[:, i, np.newaxis], feat_mapping[i], axis=-1))
        new_data = np.concatenate(new_data, axis=-1)

        # create mask, based on splits
        mask = np.ones(new_data.shape)
        new_columns = []
        for feat in range(X.shape[1]):
            # jj = j in the enhanced dataset
            pos = int(np.sum(feat_mapping[:feat]))

            if feat_mapping[feat] == 1:
                new_columns.append("x{}".format(feat))
                continue
            else:
                feat_splits = self.splits["feat_{}".format(feat)]
                lst = [
                    list(i) for i in itertools.product([0, 1], repeat=len(feat_splits))
                ]
                for ii, bin in enumerate(lst):
                    new_name = "x{} | ".format(feat)
                    init_col_mask = np.ones(new_data.shape[0]) * True
                    for jj, b in enumerate(bin):
                        if b == 0:
                            if feat_splits[jj]["type"] == "cat":
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    == feat_splits[jj]["position"],
                                )
                                # add with two decimals
                                new_name += "x{}={:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                            else:
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    <= feat_splits[jj]["position"],
                                )
                                new_name += "x{}<={:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                        else:
                            if feat_splits[jj]["type"] == "cat":
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    != feat_splits[jj]["position"],
                                )
                                new_name += "x{}!={:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                            else:
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    > feat_splits[jj]["position"],
                                )
                                new_name += "x{}>{:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                    # current position in mask
                    mask[:, pos + ii] = init_col_mask
                    new_columns.append(new_name[:-3])
        self.mask = mask
        self.new_data = new_data * mask
        self.new_names = new_columns
        return self.new_data


def rename_features():
    pass
