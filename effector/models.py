from effector import helpers
import re
import numpy as np

class Base:
    def __init__(self, name):
        # CamelCase to snake_case
        self.name = helpers.camel_to_snake(name)

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConditionalInteraction(Base):
    def __init__(self):
        """Define a simple model.

        $f(x_1, x_2, x_3) = -x_1^2\mathbb{1}_{x_2 < 0} + x_1^2\mathbb{1}_{x_2 \geq 0} + e^{x_3}$

        """
        super().__init__(name=self.__class__.__name__)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Output of the model, shape (N,)
        """
        y = np.exp(x[:, 2])
        ind = x[:, 1] < 0
        y[ind] += -x[ind, 0]**2
        y[~ind] += x[~ind, 0]**2
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of the model.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Jacobian of the model, shape (N, 3)
        """
        y = np.zeros_like(x)
        y[:, 2] = np.exp(x[:, 2])
        ind = x[:, 1] < 0
        y[ind, 0] = -2*x[ind, 0]
        y[~ind, 0] = 2*x[~ind, 0]
        return y
        

