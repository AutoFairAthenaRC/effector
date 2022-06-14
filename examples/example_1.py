import numpy as np
import feature_effect as fe
import examples.example_utils as utils

def gen_model_params():
    params = [{"b": 0, "from": 0., "to": 20.},
              {"b": 5., "from": 20, "to": 40},
              {"b": -5, "from": 40, "to": 60},
              {"b": 1., "from": 60, "to": 80},
              {"b": -1., "from": 80, "to": 100}]
    x_start = -5
    utils.find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x = np.random.uniform(0, 100-eps, size=int(N))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x, np.array([100. - eps]))), axis=-1)
    return x


example_dir = "./examples/example_1/"

# parameters
N = 100
noise_level = 1.
K_max_fixed = 200
min_points_per_bin = 10

# init functions
seed = 4834545
np.random.seed(seed)

# define functions
model_params = gen_model_params()
model_jac = utils.create_noisy_jacobian(model_params, noise_level, seed)

# generate data and data effect
data = np.sort(generate_samples(N=N), axis=0)
model = utils.create_model(model_params, data)
y = model(data)
data_effect = model_jac(data)

# dale without standard error
dale = fe.DALE(data, model, model_jac=model_jac)
dale.fit(alg_params={"nof_bins": 5, "min_points_per_bin": min_points_per_bin})
dale.plot(error="standard error",  savefig=example_dir + "im_1.png", gt=model, gt_bins=utils.create_gt_bins(model_params))

dale = fe.DALE(data, model, model_jac=model_jac)
dale.fit(alg_params={"nof_bins": 5, "min_points_per_bin": min_points_per_bin})
dale.plot(error="std",  savefig=example_dir + "im_2.png", gt=model, gt_bins=utils.create_gt_bins(model_params))


# parameters
N = 1000
noise_level = 1.
K_max_fixed = 200
min_points_per_bin = 10

# init functions
seed = 4834545
np.random.seed(seed)

# define functions
model_params = gen_model_params()
model_jac = utils.create_noisy_jacobian(model_params, noise_level, seed)

# generate data and data effect
data = np.sort(generate_samples(N=N), axis=0)
model = utils.create_model(model_params, data)
y = model(data)
data_effect = model_jac(data)

# dale without standard error
dale = fe.DALE(data, model, model_jac=model_jac)
dale.fit(alg_params={"nof_bins": 5, "min_points_per_bin": min_points_per_bin})
dale.plot(error="standard error",  savefig=example_dir + "im_3.png", gt=model, gt_bins=utils.create_gt_bins(model_params))

dale = fe.DALE(data, model, model_jac=model_jac)
dale.fit(alg_params={"nof_bins": 5, "min_points_per_bin": min_points_per_bin})
dale.plot(error="std",  savefig=example_dir + "im_4.png", gt=model, gt_bins=utils.create_gt_bins(model_params))


# parameters
N = 10000
noise_level = 1.
K_max_fixed = 200
min_points_per_bin = 10

# init functions
seed = 4834545
np.random.seed(seed)

# define functions
model_params = gen_model_params()
model_jac = utils.create_noisy_jacobian(model_params, noise_level, seed)

# generate data and data effect
data = np.sort(generate_samples(N=N), axis=0)
model = utils.create_model(model_params, data)
y = model(data)
data_effect = model_jac(data)

dale = fe.DALE(data, model, model_jac=model_jac)
dale.fit(alg_params={"nof_bins": 5, "min_points_per_bin": min_points_per_bin})
dale.plot(error="standard error",  savefig=example_dir + "im_5.png", gt=model, gt_bins=utils.create_gt_bins(model_params))

dale = fe.DALE(data, model, model_jac=model_jac)
dale.fit(alg_params={"nof_bins": 5, "min_points_per_bin": min_points_per_bin})
dale.plot(error="std",  savefig=example_dir + "im_6.png", gt=model, gt_bins=utils.create_gt_bins(model_params))

