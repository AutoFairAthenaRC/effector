import examples.utils as utils
path = utils.add_parent_path()
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import example_models.distributions as dist
import example_models.models as models


# gen dist
gen_dist = dist.Correlated1(D=2, x1_min=0, x1_max=1, x2_sigma=0)
X = gen_dist.generate(N=500)
axis_limits = gen_dist.axis_limits

# model
model = models.Example1()
# model.plot(axis_limits=axis_limits, nof_points=30)

# ALE with equal bin sizes
dale = fe.DALE(data=X,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "fixed", "nof_bins" : 20, "min_points_per_bin": 5}
dale.fit(alg_params=alg_params)
y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
                           s=0,
                           uncertainty=True)
dale.plot(s=0)


# ALE with variable bin sizes
dale = fe.DALE(data=X,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "dp", "max_nof_bins" : 20, "min_points_per_bin": 10}
dale.fit(alg_params=alg_params)
y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
                           s=0,
                           uncertainty=True)
dale.plot(s=0)


# # PDP
# pdp = fe.PDP(data=X,
#              model=model.predict,
#              axis_limits=axis_limits)
# y = pdp.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
#              s=0,
#              uncertainty=False)
# pdp.plot(s=0)


# PDP with ICE
pdp_ice = fe.PDPwithICE(data=X,
                        model=model.predict,
                        axis_limits=axis_limits)
pdp_ice.plot(s=0)
