import typing

import pythia.binning_methods
import pythia.utils as utils
import pythia.visualization as vis
import pythia.bin_estimation as be
import pythia.helpers as helpers
from pythia.fe_base import FeatureEffectBase
import numpy as np


class DALE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """
        Initializes DALE.

        Parameters
        ----------
        data: [N, D] np.array, X matrix
        model: Callable [N, D] -> [N,], prediction function
        model_jac: Callable [N, D] -> [N,D], jacobian function
        axis_limits: [2, D] np.ndarray or None, if None they will be auto computed from the data
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.model_jac = model_jac
        self.data = data
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(DALE, self).__init__(axis_limits)

        # init as None, it will get gradients after compile
        self.data_effect = None

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        TODO add numerical approximation
        """
        if self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        else:
            # TODO add numerical approximation
            pass

    def _fit_feature(self, feat: int, params) -> typing.Dict:
        """Fit a specific feature, using DALE.

        Parameters
        ----------
        feat: index of the feature
        params: Dict, with fitting-specific parameters
            - "bin_method": in ["fixed", "greedy", "dp"], which method to use
            - "nof_bins": int (default 100), how many bins to create -> important only if "fixed" is used
            - "max_nof_bins: int (default 20), max number of bins -> important only if "greedy" or "dp" is used
            - "min_points_per_bin" int (default 10), min numbers per bin -> important in all cases

        Returns
        -------

        """
        # params = helpers.prep_dale_fit_params(params)

        if self.data_effect is None:
            self.compile()

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feat] >= self.axis_limits[0, feat],
            self.data[:, feat] <= self.axis_limits[1, feat],
        )
        data = self.data[ind, :]
        data_effect = self.data_effect[ind, :]

        # bin estimation
        # assert params["bin_method"] in ["fixed", "greedy", "dp"]
        if isinstance(params, pythia.binning_methods.Fixed):
            bin_est = be.Fixed(
                data, data_effect, feature=feat, axis_limits=self.axis_limits
            )
            bin_est.find(nof_bins=params.nof_bins, min_points=params.min_points_per_bin)
        elif isinstance(params, pythia.binning_methods.Greedy):
            bin_est = be.Greedy(
                data, data_effect, feature=feat, axis_limits=self.axis_limits
            )
            bin_est.find(
                min_points=params.min_points_per_bin, n_max=params.max_nof_bins, fact=params.fact
            )
        elif isinstance(params, pythia.binning_methods.DynamicProgramming):
            bin_est = be.DP(
                data, data_effect, feature=feat, axis_limits=self.axis_limits
            )
            bin_est.find(
                min_points=params.min_points_per_bin, k_max=params.max_nof_bins, discount=params.discount
            )

        # stats per bin
        assert bin_est.limits is not False, (
            "Impossible to compute bins with enough points for feature "
            + str(feat + 1)
            + " and binning strategy: "
            + params["bin_method"]
            + ". Change bin strategy or "
            "the parameters of the method"
        )
        dale_params = utils.compute_ale_params_from_data(
            data[:, feat], data_effect[:, feat], bin_est.limits
        )

        dale_params["alg_params"] = params
        return dale_params

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            binning_method="fixed",
            normalize: bool = True,
            ) -> None:
        """Fit feature effect plot for the asked features

        Parameters
        ----------
        features: features to compute the normalization constant
            - "all", all features
            - int, the index of the feature
            - list, list of indexes of the features
        binning_method: dictionary with method-specific parameters for fitting the FE plots
        normalize: bool, whether to compute the normalization constants
        """

        # if binning_method is a string -> make it a class
        if isinstance(binning_method, str):
            assert binning_method in ["fixed", "greedy", "dynamic_programming"]
            if binning_method == "fixed":
                tmp = pythia.binning_methods.Fixed()
            elif binning_method == "greedy":
                tmp = pythia.binning_methods.Greedy()
            else:
                tmp = pythia.binning_methods.DynamicProgramming()
            binning_method = tmp

        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s, binning_method)
            if normalize:
                self.norm_const[s] = self._compute_norm_const(s)
            self.is_fitted[s] = True

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False):
        params = self.feature_effect["feature_" + str(feature)]
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if uncertainty:
            std = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_variance"]),
                dx=params["dx"],
            )
            std_err = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_estimator_variance"]),
                dx=params["dx"],
            )

            return y, std, std_err
        else:
            return y

    def plot(
        self,
        feature: int = 0,
        error: typing.Union[None, str] = "std",
        scale_x=None,
        scale_y=None,
        savefig=False,
    ):
        """

        Parameters
        ----------
        s
        error:
        scale_x:
        scale_y:
        savefig:
        """
        vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            error=error,
            scale_x=scale_x,
            scale_y=scale_y,
            savefig=savefig,
        )


class DALEGroundTruth(FeatureEffectBase):
    def __init__(self, mean, mean_int, var, var_int, axis_limits):
        super(DALEGroundTruth, self).__init__(axis_limits)
        self.mean = mean
        self.mean_int = mean_int
        self.var = var
        self.var_int = var_int

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all"):
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.is_fitted[s] = True

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False):
        if not uncertainty:
            return self.mean_int(x)
        else:
            return self.mean_int(x), self.var_int(x), None

    def plot(self, feature: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        if normalized:
            y = self.eval(x, feature)
        else:
            y = self._eval_unnorm(x, feature)
        vis.plot_1d(x, y, title="Ground-truth ALE for feature %d" % (s + 1))


class DALEBinsGT(FeatureEffectBase):
    def __init__(
        self,
        mean: callable,
        var: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        super(DALEBinsGT, self).__init__(axis_limits)
        self.mean = mean
        self.var = var

    def _fit_feature(self, feat: int, params) -> typing.Dict:
        """Fit a specific feature, using DALE.

        Parameters
        ----------
        feat: index of the feature
        params: Dict, with fitting-specific parameters
            - "bin_method": in ["fixed", "greedy", "dp"], which method to use
            - "nof_bins": int (default 100), how many bins to create -> important only if "fixed" is used
            - "max_nof_bins: int (default 20), max number of bins -> important only if "greedy" or "dp" is used
            - "min_points_per_bin" int (default 10), min numbers per bin -> important in all cases

        Returns
        -------

        """
        if isinstance(params, pythia.binning_methods.Fixed):
            bin_est = be.FixedGT(self.mean, self.var, self.axis_limits, feature=feat)
            bin_est.find(nof_bins=params.nof_bins, min_points=params.min_points_per_bin)
        elif isinstance(params, pythia.binning_methods.Greedy):
            bin_est = be.GreedyGT(self.mean, self.var, self.axis_limits, feature=feat)
            bin_est.find(
                min_points=params.min_points_per_bin, n_max=params.max_nof_bins, fact=params.fact
            )
        elif isinstance(params, pythia.binning_methods.DynamicProgramming):
            bin_est = be.DPGT(self.mean, self.var, self.axis_limits, feature=feat)
            bin_est.find(
                min_points=params.min_points_per_bin, k_max=params.max_nof_bins, discount=params.discount
            )

        # stats per bin
        assert bin_est.limits is not False, (
            "Impossible to compute bins with enough points for feature "
            + str(feat + 1)
            + " and binning strategy: "
            + params["bin_method"]
            + ". Change bin strategy or "
            "the parameters of the method"
        )
        dale_params = utils.compute_ale_params_from_gt(self.mean, bin_est.limits)
        dale_params["limits"] = bin_est.limits
        return dale_params

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False):
        params = self.feature_effect["feature_" + str(feature)]
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if uncertainty:
            var = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=params["bin_variance"],
                dx=params["dx"],
                square=True,
            )
            return y, var
        else:
            return y

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            binning_method="fixed",
            normalize: bool = True,
            ) -> None:
        """Fit feature effect plot for the asked features

        Parameters
        ----------
        features: features to compute the normalization constant
            - "all", all features
            - int, the index of the feature
            - list, list of indexes of the features
        binning_method: dictionary with method-specific parameters for fitting the FE plots
        normalize: bool, whether to compute the normalization constants
        """

        # if binning_method is a string -> make it a class
        if isinstance(binning_method, str):
            assert binning_method in ["fixed", "greedy", "dynamic_programming"]
            if binning_method == "fixed":
                tmp = pythia.binning_methods.Fixed()
            elif binning_method == "greedy":
                tmp = pythia.binning_methods.Greedy()
            else:
                tmp = pythia.binning_methods.DynamicProgramming()
            binning_method = tmp

        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s, binning_method)
            if normalize:
                self.norm_const[s] = self._compute_norm_const(s)
            self.is_fitted[s] = True

    def plot(self, feature: int, normalized: bool = True, nof_points: int = 30) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        if normalized:
            y = self.eval(x, feature)
        else:
            y = self._eval_unnorm(x, feature)
        vis.plot_1d(x, y, title="ALE GT Bins for feature %d" % (feature + 1))
