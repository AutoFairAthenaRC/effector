import numpy as np
import effector.binning_methods as binning_methods
import effector.helpers as helpers
import effector.utils as utils
from effector.partitioning import Regions
from effector.global_effect_ale import RHALE
from effector.global_effect_pdp import PDP
import typing
from tqdm import tqdm


BIG_M = helpers.BIG_M


class RegionalEffectBase:
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: callable,
        nof_instances: int | str = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ) -> None:
        """
        Constructor for the RegionalEffect class.
        """
        self.method_name = method_name.lower()

        # select nof_instances from the data
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        data = data[self.indices, :]
        self.data = data
        self.dim = self.data.shape[1]
        self.model = model

        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        feature_types = (
            utils.get_feature_types(data, cat_limit)
            if feature_types is None
            else feature_types
        )
        feature_names: list[str] = (
            helpers.get_feature_names(axis_limits.shape[1])
            if feature_names is None
            else feature_names
        )

        self.cat_limit = cat_limit

        self.axis_limits: np.ndarray = axis_limits
        self.feature_types: list = feature_types
        self.feature_names: list = feature_names
        self.target_name = target_name
        self.dim = self.axis_limits.shape[1]

        # state variables
        self.splits_full_depth_found: np.ndarray = np.ones([self.dim]) < 0
        self.splits_only_important_found: np.ndarray = np.ones([self.dim]) < 0

        # parameters used when fitting the regional effect
        self.method_args: typing.Dict = {}

        # dictionary with all the information required for plotting or evaluating the regional effects
        self.regions: typing.Dict[str, Regions] = {}
        # self.splits_full_depth: typing.Dict[str, list[dict]] = {}
        self.splits_full_depth_tree: typing.Dict = {}
        # self.splits_only_important: typing.Dict[str, list[dict]] = {}
        self.splits_only_important_tree: dict = {}

    def _fit_feature(
        self,
        feature: int,
        heter: callable,
        heter_pcg_drop_thres=0.1,
        heter_small_enough=0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a single feature.
        """

        # init Region Extractor
        regions = Regions(
            feature,
            heter,
            self.data,
            None,
            self.feature_types,
            self.feature_names,
            self.target_name,
            self.cat_limit,
            candidate_conditioning_features,
            min_points_per_subregion,
            nof_candidate_splits_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
            heter_small_enough,
            split_categorical_features,
        )

        self.regions["feature_{}".format(feature)] = regions
        splits = regions.search_all_splits()

        # self.splits_full_depth["feature_{}".format(feature)] = splits
        self.splits_full_depth_found[feature] = True

        important_splits = regions.choose_important_splits()
        # self.splits_only_important[
        #     "feature_{}".format(feature)
        # ] = important_splits
        self.splits_only_important_found[feature] = True

        self.splits_full_depth_tree["feature_{}".format(feature)] = regions.splits_to_tree()
        self.splits_only_important_tree["feature_{}".format(feature)] = regions.splits_to_tree(True)

    def refit(self, feature, centering):
        "Checks if refitting is needed"
        if not self.splits_full_depth_found[feature]:
            return True
        else:
            # TODO: change that, we want refiting in many cases
            if centering is not False:
                if self.method_args["feature_" + str(feature)]["centering"] != centering:
                    return True
        return False

    def _get_data_from_node(self, feature, node_idx):
        # find the region and get data (and data_effect)
        if not self.splits_full_depth_found[feature]:
            self._fit_feature(feature)

        # assert node id exists
        assert node_idx in [node.idx for node in self.splits_only_important_tree["feature_{}".format(feature)].nodes], "Node {} does not exist".format(node_idx)

        # find the node
        node = [node for node in self.splits_only_important_tree["feature_{}".format(feature)].nodes if node.idx == node_idx][0]

        # get data
        data = node.data["data"]
        data_effect = node.data["data_effect"]
        return data, data_effect

    def eval(self, feature, node_idx, xs, heterogeneity=False, centering=False):
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def plot_first_level(self, *args, **kwargs):
        raise NotImplementedError

    def print_tree(self, features, only_important=True):
        features = helpers.prep_features(features, self.dim)
        for feat in features:
            if not self.splits_full_depth_found[feat]:
                self._fit_feature(feat)

            print("Feature {}".format(feat))
            if only_important:
                if self.splits_only_important_tree["feature_{}".format(feat)] is None:
                    print("No important splits found for feature {}".format(feat))
                else:
                    self.splits_only_important_tree["feature_{}".format(feat)].show()
            else:
                if self.splits_full_depth_tree["feature_{}".format(feat)] is None:
                    print("No important splits found for feature {}".format(feat))
                else:
                    self.splits_full_depth_tree["feature_{}".format(feat)].show()

    def print_level_stats(self, features):
        features = helpers.prep_features(features, self.dim)
        for feat in features:
            if not self.splits_full_depth_found[feat]:
                self._fit_feature(feat)

            print("Feature {}".format(feat))
            if self.splits_full_depth_tree["feature_{}".format(feat)] is None:
                print("No important splits found for feature {}".format(feat))
            else:
                self.splits_full_depth_tree["feature_{}".format(feat)].show1()

    def describe_subregions(
        self,
        features,
        only_important=True,
        scale_x: typing.Union[None, list[dict]] = None,
    ):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            if not self.splits_full_depth_found[feature]:
                self._fit_feature(feature)

            # it means it a categorical feature
            if self.splits_full_depth_tree["feature_{}".format(feature)] is None:
                continue

            feature_name = self.feature_names[feature]
            if only_important:
                tree = self.splits_only_important_tree["feature_{}".format(feature)]
                if len(tree.nodes) == 1:
                    print("No important splits found for feature {}".format(feature))
                    continue
                else:
                    print("Important splits for feature {}".format(feature_name))
            else:
                print("All splits for feature {}".format(feature_name))
                tree = self.splits_full_depth_tree["feature_{}".format(feature)]

            max_level = max([node.level for node in tree.nodes])
            for level in range(1, max_level+1):
                previous_level_nodes = tree.get_level_nodes(level-1)
                level_nodes = tree.get_level_nodes(level)
                type_of_split_feature = level_nodes[0].data["feature_type"]
                foc_name = self.feature_names[level_nodes[0].data["feature"]]
                print("- On feature {} ({})".format(foc_name, type_of_split_feature))

                position_split_formatted = (
                    "{:.2f}".format(level_nodes[0].data["position"])
                    if scale_x is None
                    else "{:.2f}".format(
                        level_nodes[0].data["position"] * scale_x[level_nodes[0].data["feature"]]["std"]
                        + scale_x[level_nodes[0].data["feature"]]["mean"]
                    )
                )
                print("  - Position of split: {}".format(position_split_formatted))

                weight_heter_before = np.sum([node.data["weight"] * node.data["heterogeneity"] for node in previous_level_nodes])
                print("  - Heterogeneity before split: {:.2f}".format(weight_heter_before))

                weight_heter = np.sum([node.data["weight"] * node.data["heterogeneity"] for node in level_nodes])
                print("  - Heterogeneity after split: {:.2f}".format(weight_heter))
                weight_heter_drop = weight_heter_before - weight_heter
                print("  - Heterogeneity drop: {:.2f} ({:.2f} %)".format(
                    weight_heter_drop, weight_heter_drop / weight_heter_before * 100)
                )

                nof_instances_before = [nod.data["nof_instances"] for nod in previous_level_nodes]
                print("  - Number of instances before split: {}".format(nof_instances_before))
                nof_instances = [nod.data["nof_instances"] for nod in level_nodes]
                print("  - Number of instances after split: {}".format(nof_instances))


class RegionalRHALEBase(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        instance_effects: None | np.ndarray = None,
        nof_instances: int | str = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        """
        Regional RHALE constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            model_jac: the black-box model Jacobian (N,D) -> (N,D)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
        """
        self.model_jac = model_jac
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        instance_effects = self.model_jac(data)
        data = data[self.indices, :]

        instance_effects = instance_effects[self.indices, :]
        self.instance_effects = instance_effects
        super(RegionalRHALEBase, self).__init__(
            "rhale",
            data,
            model,
            "all",
            axis_limits,
            feature_types,
            cat_limit, feature_names, target_name
        )

    def _create_heterogeneity_function(self, foi, binning_method, min_points=10):
        binning_method = helpers.prep_binning_method(binning_method)

        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            rhale = RHALE(data, self.model, self.model_jac,  "all", None, instance_effects)
            try:
                rhale.fit(features=foi, binning_method=binning_method)
            except:
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            axis_limits = helpers.axis_limits_from_data(data)
            stop = np.array([axis_limits[:, foi][1]])
            _, z = rhale.eval(feature=foi, xs=stop, heterogeneity=True)
            return z.item()

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            binning_method: binning method to use
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(
                feat, binning_method, min_points_per_subregion
            )

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )

    def plot_first_level(
        self,
        feature: int = 0,
        heterogeneity: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        scale_x_per_feature: typing.Union[None, list[dict[str, float]]] = None,
        scale_y: typing.Union[None, dict] = None,
        show_avg_output: bool = False,
    ):

        if not self.splits_full_depth_found[feature]:
            self._fit_feature(feature)

        tree = self.splits_only_important_tree["feature_{}".format(feature)]
        level_nodes = tree.get_level_nodes(level=1)

        if len(tree.nodes) == 1:
            print("No important splits found for feature {}".format(feature))
            return

        # split to two datasets
        foc = level_nodes[0].data["feature"]
        position = level_nodes[0].data["position"]
        type_of_split_feature = self.feature_types[foc]
        data_1 = level_nodes[0].data["data"]
        data_2 = level_nodes[1].data["data"]
        data_effect_1 = level_nodes[0].data["data_effect"]
        data_effect_2 = level_nodes[1].data["data_effect"]
        axis_limits = helpers.axis_limits_from_data(self.data)

        # plot the two RHALE objects
        foc_name = self.feature_names[foc]
        foi_name = self.feature_names[feature]

        # feature_names
        position = (
            scale_x_per_feature[foc]["mean"]
            + scale_x_per_feature[foc]["std"] * position
            if scale_x_per_feature is not None
            else position
        )
        if type_of_split_feature != "cat":
            feature_name_1 = "{} given {}<={:.2f}".format(foi_name, foc_name, position)
            feature_name_2 = "{} given {}>{:.2f}".format(foi_name, foc_name, position)
        else:
            feature_name_1 = "{} given {}=={:.2f}".format(foi_name, foc_name, position)
            feature_name_2 = "{} given {}!={:.2f}".format(foi_name, foc_name, position)

        feature_names_1 = self.feature_names.copy()
        feature_names_1[feature] = feature_name_1
        feature_names_2 = self.feature_names.copy()
        feature_names_2[feature] = feature_name_2

        # create two RHALE objects and plot
        rhale_1 = RHALE(
            data_1,
            self.model,
            self.model_jac,
            "all",
            axis_limits,
            data_effect_1,
            feature_names=feature_names_1,
            target_name=self.target_name,
        )
        rhale_1.fit(features=feature, binning_method=binning_method)
        rhale_2 = RHALE(
            data_2,
            self.model,
            self.model_jac,
            "all",
            axis_limits,
            data_effect_2,
            feature_names=feature_names_2,
            target_name=self.target_name,
        )
        rhale_2.fit(features=feature, binning_method=binning_method)

        rhale_1.plot(
            feature,
            heterogeneity,
            centering,
            scale_x_per_feature[feature] if scale_x_per_feature is not None else None,
            scale_y if scale_y is not None else None,
            show_avg_output=show_avg_output
        )
        rhale_2.plot(
            feature,
            heterogeneity,
            centering,
            scale_x_per_feature[feature] if scale_x_per_feature is not None else None,
            scale_y if scale_y is not None else None,
            show_avg_output=show_avg_output
        )


class RegionalPDP(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        nof_instances: int | str = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
    ):
        """
        Regional PDP constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
        """
        super(RegionalPDP, self).__init__("pdp", data, model, nof_instances, axis_limits, feature_types, cat_limit, feature_names)

    def _create_heterogeneity_function(self, foi, min_points=10):
        def heter(data) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            pdp = PDP(data, self.model, None, nof_instances=100)
            try:
                pdp.fit(features=foi)
            except:
                return BIG_M

            # heterogeneity is the mean heterogeneity over the curve
            axis_limits = helpers.axis_limits_from_data(data)

            xx = np.linspace(axis_limits[:, foi][0], axis_limits[:, foi][1], 10)
            try:
                _, z = pdp.eval(feature=foi, xs=xx, heterogeneity=True)
            except:
                return BIG_M
            return np.mean(z)

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional PDP for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
            split_categorical_features
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(feat, min_points_per_subregion)

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )

    def plot_first_level(
        self,
        feature: int = 0,
        heterogeneity: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = True,
        scale_x_per_feature: typing.Union[None, list[dict[str, float]]] = None,
        scale_y: typing.Union[None, dict] = None,
        show_avg_output: bool = False,
    ):

        if not self.splits_full_depth_found[feature]:
            self._fit_feature(feature)

        tree = self.splits_only_important_tree["feature_{}".format(feature)]
        level_nodes = tree.get_level_nodes(level=1)

        if len(tree.nodes) == 1:
            print("No important splits found for feature {}".format(feature))
            return

        # split to two datasets
        foc = level_nodes[0].data["feature"]
        position = level_nodes[0].data["position"]
        type_of_split_feature = self.feature_types[foc]
        data_1 = level_nodes[0].data["data"]
        data_2 = level_nodes[1].data["data"]
        axis_limits = helpers.axis_limits_from_data(self.data)

        # plot the two RHALE objects
        foc_name = self.feature_names[foc]
        foi_name = self.feature_names[feature]

        # feature_names
        position = (
            scale_x_per_feature[foc]["mean"]
            + scale_x_per_feature[foc]["std"] * position
            if scale_x_per_feature is not None
            else position
        )
        if type_of_split_feature != "cat":
            feature_name_1 = "{} given {}<={:.2f}".format(foi_name, foc_name, position)
            feature_name_2 = "{} given {}>{:.2f}".format(foi_name, foc_name, position)
        else:
            feature_name_1 = "{} given {}=={:.2f}".format(foi_name, foc_name, position)
            feature_name_2 = "{} given {}!={:.2f}".format(foi_name, foc_name, position)

        feature_names_1 = self.feature_names.copy()
        feature_names_1[feature] = feature_name_1
        feature_names_2 = self.feature_names.copy()
        feature_names_2[feature] = feature_name_2

        # create two RHALE objects and plot
        pdp_1 = PDP(
            data_1,
            self.model,
            axis_limits,
            feature_names=feature_names_1,
            target_name=self.target_name,
        )
        pdp_2 = PDP(
            data_2,
            self.model,
            axis_limits,
            feature_names=feature_names_2,
            target_name=self.target_name,
        )

        pdp_1.plot(
            feature,
            heterogeneity="ice",
            nof_ice=100,
            scale_x=scale_x_per_feature[feature]
            if scale_x_per_feature is not None
            else None,
            scale_y=scale_y if scale_y is not None else None,
            centering=centering,
            show_avg_output=show_avg_output,
        )
        pdp_2.plot(
            feature,
            heterogeneity="ice",
            nof_ice=100,
            scale_x=scale_x_per_feature[feature]
            if scale_x_per_feature is not None
            else None,
            scale_y=scale_y if scale_y is not None else None,
            centering=centering,
            show_avg_output=show_avg_output
        )


def scale_names(scale_x_per_feature, feature_name):
    # Example feature name: "x_1 | x_2 != 0.0 and x_0  > 0.1"
    # Example scale_x_per_feature: {"feature_0": {"mean": 0.1, "std": 0.2}, "feature_1": {"mean": 0.1, "std": 0.2}, "feature_3": {"mean": 0.1, "std": 0.2}}
    # I want to make it "x_1 | x_2 != 0.0*0.2 + 0.1 and x_0  > 0.1*0.2 + 0.1"


    # split the feature name into the feature name and the condition
    feature_name, condition = feature_name.split("|")

        # for each feature in the condition, replace the feature name with the scaled feature name

