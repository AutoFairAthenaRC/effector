import typing
import numpy as np
from effector import binning_methods as bm


BIG_M = 1e8
EPS = 1e-8
EMPTY_SYMBOL = 1e8


def prep_features(feat: typing.Union[str, list], D) -> list:
    assert type(feat) in [list, str, int]
    if feat == "all":
        feat = [i for i in range(D)]
    elif type(feat) == int:
        feat = [feat]
    return feat


def prep_centering(centering):
    assert type(centering) in [bool, str]
    if type(centering) is str:
        assert centering in ["zero_start", "zero_integral"]

    # set the default
    if centering is True:
        centering = "zero_integral"
    return centering


def prep_confidence_interval(confidence_interval):
    assert type(confidence_interval) in [bool, str]
    if type(confidence_interval) is str:
        assert confidence_interval in ["std", "std_err", "ice", "shap_values"]

    if confidence_interval is True:
        confidence_interval = "std"
    return confidence_interval


def axis_limits_from_data(data: np.ndarray) -> np.ndarray:
    """Compute axis limits from data."""
    D = data.shape[1]
    axis_limits = np.zeros([2, D])
    for d in range(D):
        axis_limits[0, d] = data[:, d].min()
        axis_limits[1, d] = data[:, d].max()
    return axis_limits


def prep_dale_fit_params(par: dict):
    if par is None:
        par = {}

    if "bin_method" in par.keys():
        assert par["bin_method"] in ["fixed", "greedy", "dp"]
    else:
        par["bin_method"] = "fixed"

    if "nof_bins" in par.keys():
        assert type(par["nof_bins"]) == int
    else:
        par["nof_bins"] = 100

    if "max_nof_bins" in par.keys():
        assert type(par["max_nof_bins"]) == int
    else:
        par["max_nof_bins"] = 20

    if "min_points_per_bin" in par.keys():
        assert type(par["max_nof_bins"]) == int
    else:
        par["min_points_per_bin"] = None

    return par


def prep_ale_fit_params(par: dict):
    if "nof_bins" in par.keys():
        assert type(par["nof_bins"]) == int
    else:
        par["nof_bins"] = 100
    return par


def prep_nof_instances(nof_instances: typing.Union[int, str], N: int) -> tuple[int, np.ndarray]:
    """Prepares the argument nof_instances

    Args
    ---
        nof_instances (int or str): The number of instances to use for the explanation
        N (int): The number of instances in the dataset

    Returns
    ---
        nof_instances (int): The number of instances to use for the explanation
        indices (np.ndarray): The indices of the instances to use for the explanation
    """
    # assertions
    assert type(nof_instances) in [int, str]
    if type(nof_instances) is not int:
        assert nof_instances == "all"

    # prepare nof_instances
    if nof_instances == "all":
        nof_instances = N
    indices = np.random.choice(N, nof_instances, replace=False) if nof_instances < N else np.arange(N)
    return nof_instances, indices


def prep_binning_method(method):
    assert method in ["greedy", "dp", "fixed"] or isinstance(method, bm.Fixed) or isinstance(method, bm.DynamicProgramming) or isinstance(method, bm.Greedy)

    if method == "greedy":
        return bm.Greedy()
    elif method == "dp":
        return bm.DynamicProgramming()
    elif method == "fixed":
        return bm.Fixed()
    else:
        return method


def get_feature_names(dim: int) -> list:
    """Returns the feature names for the given dimensionality"""
    return [i for i in range(dim)]


def prep_avg_output(data, model, avg_output, scale_y) -> float:
    breakpoint()
    avg_output = avg_output if avg_output is not None else np.mean(model(data))
    avg_output = avg_output * scale_y["std"] + scale_y["mean"] if scale_y is not None else avg_output
    return avg_output


def drop_points_outside_axis_limits(data: np.ndarray, axis_limits: np.ndarray, feature: int) -> np.ndarray:
    """Drop points outside the axis limits"""
    data = data[data[:, feature] >= axis_limits[0, feature]]
    data = data[data[:, feature] <= axis_limits[1, feature]]
    return data