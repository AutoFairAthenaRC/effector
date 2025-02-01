import numpy as np
from typing import Callable, List, Optional, Union, Tuple
from effector import helpers
from abc import ABC, abstractmethod


class GlobalEffectBase(ABC):
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        avg_output: Optional[float] = None,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ) -> None:
        """
        Constructor for the FeatureEffectBase class.
        """
        assert data.ndim == 2

        self.method_name = method_name.lower()
        self.model = model
        self.model_jac = model_jac

        self.dim = data.shape[1]

        # data preprocessing (i): if axis_limits passed manually,
        # keep only the points within,
        # otherwise, compute the axis limits from the data
        if axis_limits is not None:
            assert axis_limits.shape == (2, self.dim)
            assert np.all(axis_limits[0, :] <= axis_limits[1, :])

            # drop points outside of limits
            accept_indices = helpers.indices_within_limits(data, axis_limits)
            data = data[accept_indices, :]
            data_effect = data_effect[accept_indices, :] if data_effect is not None else None
        else:
            axis_limits = helpers.axis_limits_from_data(data)
        self.axis_limits: np.ndarray = axis_limits

        # data preprocessing (ii): select nof_instances from the remaining data
        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        data = data[self.indices, :]
        data_effect = data_effect[self.indices, :] if data_effect is not None else None

        # store the data
        self.data: np.ndarray = data
        self.data_effect: Optional[np.ndarray] = data_effect

        self.avg_output = None

        # set feature names
        feature_names: list[str] = (
            helpers.get_feature_names(axis_limits.shape[1])
            if feature_names is None
            else feature_names
        )
        self.feature_names: list = feature_names
        self.target_name = "y" if target_name is None else target_name

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 0

        # parameters used when fitting the feature effect
        self.fit_args: dict = {}

        # dict, like {"feature_i": {"quantity_1": value_1, "quantity_2": value_2, ...}} for the i-th
        self.feature_effect: dict = {}

    #     # .eval() method cache, each element is a dict like {"feature_i": {"xs": xs, "data": {"y": y, "std": std}}}
    #     self.cache: dict[str, list[dict]] = {}

    # def update_cache(self, feature, xs, data, parameters):
    #     """Cache stores up to 3 items per feature"""
    #     if "feature_" + str(feature) not in self.cache.keys():
    #         self.cache["feature_" + str(feature)] = []

    #     # choose the feature-specific cache
    #     cache = self.cache["feature_" + str(feature)]
    #     if len(cache) < 3:
    #         cache.append({"xs": xs, "data": data, "parameters": parameters})
    #     else:
    #         # drop the first element
    #         cache = cache[1:]
    #         cache.append({"feature_" + str(feature): {"xs": xs, "data": data, "parameters": parameters}})

    # def retrieve_from_cache(self, feature, xs, parameters):
    #     """Retrieve data from the cache"""
    #     if "feature_" + str(feature) not in self.cache.keys():
    #         return None
    #     else:
    #         cache = self.cache["feature_" + str(feature)]
    #         for i in range(len(cache)):
    #             if cache[i]["xs"].shape == xs.shape:
    #                 if np.allclose(cache[i]["xs"], xs) and self._parameters_equal(parameters, cache[i]["parameters"]):
    #                     return cache[i]["data"]
    #     return None

    # def _parameters_equal(self, param_1, param_2):
    #     return True

    @abstractmethod
    def fit(
            self,
            features: Union[int, str, list] = "all",
            centering: Union[bool, str] = False,
            **kwargs
    ) -> None:
        """Fit, i.e., compute the quantities that are necessary for evaluating and plotting the feature effect, for the given features.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.
            centering: whether to center the feature effect plot

                    - If `centering` is `False`, the plot is not centered
                    - If `centering` is `True` or `zero_integral`, the plot is centered around the `y` axis.
                    - If `centering` is `zero_start`, the plot starts from zero.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(
            self,
            feature: int,
            heterogeneity: Union[bool, str] = False,
            centering: Union[bool, str] = False,
            **kwargs
    ) -> None:
        """

        Parameters
        ----------
        feature: index of the feature to plot
        heterogeneity: whether to plot the heterogeneity measures

            - If `heterogeneity=False`, the plot shows only the mean effect
            - If `heterogeneity=True`, the plot additionally shows the heterogeneity with the default visualization, e.g., ICE plots for PDPs
            - If `heterogeneity=<str>`, the plot shows the heterogeneity using the specified method

        centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.
        **kwargs: all other plot-specific arguments
        """
        raise NotImplementedError

    def requires_refit(self, feature, centering):
        """Check if refitting is needed
        """
        if not self.is_fitted[feature]:
            return True
        else:
            if centering is not False and centering != self.fit_args["feature_" + str(feature)]["centering"]:
                return True
        return False

    @abstractmethod
    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        heterogeneity: bool = False,
        centering: Union[bool, str] = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Notes:
            This is a common method among all the FE classes.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot

              - `np.ndarray` of shape `(T, )`

            heterogeneity: whether to return the heterogeneity measures.

                  - if `heterogeneity=False`, the function returns the mean effect at the given `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, heterogeneity)` otherwise

        Notes:
            * If `centering` is `False`, the plot is not centered
            * If `centering` is `True` or `"zero_integral"`, the plot is centered by subtracting its mean.
            * If `centering` is `"zero_start"`, the plot starts from zero.

        Notes:
            * If `heterogeneity` is `False`, the plot returns only the mean effect `y` at the given `xs`.
            * If `heterogeneity` is `True`, the plot returns `(y, std)` where:
                * `y` is the mean effect
                * `std` is the standard deviation of the mean effect
        """
        raise NotImplementedError
