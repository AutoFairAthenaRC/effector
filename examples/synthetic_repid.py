import sys, os
import timeit
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pythia
import pythia.regions as regions
import pythia.interaction as interaction
import matplotlib.pyplot as plt
# from nodegam.sklearn import NodeGAMClassifier, NodeGAMRegressor
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor

# hack to reload modules
import importlib
pythia = importlib.reload(pythia)


class RepidSimpleDist:
    """
    x1 ~ U(-1, 1)
    x2 ~ U(-1, 1)
    x3 ~ Bernoulli(0.5)
    """

    def __init__(self):
        self.D = 2
        self.axis_limits = np.array([[-1, 1], [-1, 1], [0, 1]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x2 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x3 = np.random.choice([0, 1], int(N), p=[0.5, 0.5])

        x = np.stack((x1, x2, x3), axis=-1)
        return x


class RepidSimpleModel:
    def __init__(self, a1=8, a2=-16):
        self.a1 = a1
        self.a2 = a2

    def predict(self, x):
        y = np.zeros_like(x[:,0])

        cond = x[:, 0] > 0
        y[cond] += self.a1*x[cond, 1]

        cond = x[:, 2] == 0
        y[cond] += self.a2*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        y = np.zeros_like(x)

        cond = x[:, 0] > 0
        y[cond, 1] += self.a1

        cond = x[:, 2] == 0
        y[cond, 1] += self.a2
        return y


def plot_effect_ebm(ebm_model, ii):
    explanation = ebm_model.explain_global()
    plt.figure(figsize=(10, 6))
    xx = explanation.data(ii)["names"][:-1]
    yy = explanation.data(ii)["scores"]
    plt.xlim(-1, 1)
    plt.ylim(-4, 4)
    plt.plot(xx, yy)
    plt.show()


np.random.seed(21)
feature_names = ["x1", "x2", "x3"]
dist = RepidSimpleDist()
model = RepidSimpleModel()

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)


# # check interactions
# h_index = interaction.HIndex(data=X, model=model.predict, nof_instances=950)
# # print(h_index.eval_pairwise(0, 1))
# print(h_index.eval_one_vs_all(0))
# h_index.plot(interaction_matrix=False, one_vs_all=True)
#
# # check interactions with REPID (dPDP based method)
# repid_index = interaction.REPID(data=X, model=model.predict, model_jac=model.jacobian, nof_instances=950)
# print(repid_index.eval_one_vs_all(0))
# repid_index.plot()

# find regions
reg = pythia.regions.Regions(data=X, model=model.predict, model_jac=model.jacobian, cat_limit=25)
reg.search_splits(nof_levels=2, nof_candidate_splits=20, criterion="rhale")
opt_splits = reg.choose_important_splits(0.2)

transf = pythia.regions.DataTransformer(splits=opt_splits)
new_X = transf.transform(X)

# split data
seed = 21
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_X, Y, test_size=0.20, random_state=seed)


# fit a GAM to the transformed data
gam_subspaces = ExplainableBoostingRegressor(interactions=0)
gam_subspaces.fit(X_train_new, y_train_new)
print(gam_subspaces.score(X_test_new, y_test_new))

# # fit a GAM to the initial data
# gam = ExplainableBoostingRegressor(feature_names, interactions=0)
# gam.fit(X_train, y_train)
# print(gam.score(X_test, y_test))

# fit a GAM with interactions to the initial data
gam_interactions = ExplainableBoostingRegressor(feature_names)
gam_interactions.fit(X_train, y_train)
print(gam_interactions.score(X_test, y_test))
