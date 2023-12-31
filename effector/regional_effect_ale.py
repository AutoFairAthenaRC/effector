import typing
from effector.regional_effect import RegionalEffectBase
from effector import helpers
import numpy as np
from effector.global_effect_ale import ALE, RHALE
from tqdm import tqdm
from effector import binning_methods


BIG_M = helpers.BIG_M


class RegionalRHALE(RegionalEffectBase):
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

        super(RegionalRHALE, self).__init__(
            "rhale",
            data,
            model,
            model_jac,
            instance_effects,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name
        )

    def _create_heterogeneity_function(self, foi, binning_method, min_points, centering, nof_instances):
        binning_method = prep_binning_method(binning_method)

        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            rhale = RHALE(data, self.model, self.model_jac, nof_instances, None, instance_effects)
            try:
                rhale.fit(features=foi, binning_method=binning_method, centering=centering)
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
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        binning_method: typing.Union[
                str,
                binning_methods.Fixed,
                binning_methods.DynamicProgramming,
                binning_methods.Greedy,
        ] = "greedy",
        centering: typing.Union[bool, str] = False,
        nof_instances: int = 100,
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
                feat, binning_method, min_points_per_subregion, centering, nof_instances
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

            # todo: add method args


class RegionalALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
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
        super(RegionalALE, self).__init__(
            "ale",
            data,
            model,
            None,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name
        )

    def _create_heterogeneity_function(self, foi, binning_method, min_points, centering, nof_instances):
        binning_method = prep_binning_method(binning_method)
        isinstance(binning_method, binning_methods.Fixed)
        
        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            ale = ALE(data, self.model, nof_instances, None, instance_effects)
            try:
                ale.fit(features=foi, binning_method=binning_method, centering=centering)
            except:
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            axis_limits = helpers.axis_limits_from_data(data)
            stop = np.array([axis_limits[:, foi][1]])
            _, z = ale.eval(feature=foi, xs=stop, heterogeneity=True)
            return z.item()

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
        binning_method: str | binning_methods.Fixed = binning_methods.Fixed(nof_bins=20, min_points_per_bin=0),
        centering: typing.Union[bool, str] = False,
        nof_instances: int = 100,
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
                feat, binning_method, min_points_per_subregion, centering, nof_instances
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

            # todo: add method args


def prep_binning_method(method):
    assert method in ["greedy", "dp", "fixed"] or isinstance(method, binning_methods.Fixed) or isinstance(method, binning_methods.DynamicProgramming) or isinstance(method, bm.Greedy)

    if method == "greedy":
        return binning_methods.Greedy()
    elif method == "dp":
        return binning_methods.DynamicProgramming()
    elif method == "fixed":
        return binning_methods.Fixed()
    else:
        return method
