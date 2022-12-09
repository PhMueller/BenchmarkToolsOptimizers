from typing import Dict, Union, Tuple, List

import ConfigSpace
import ConfigSpace as CS
import numpy as np

from BenchmarkToolsOptimizers.optimizers.base_optimizer import BenchmarkToolsOptimizer

from sklearn.gaussian_process import GaussianProcessRegressor
from BenchmarkToolsOptimizers import logger
from scipy.stats import norm


class _BaseBOOptimizer(BenchmarkToolsOptimizer):
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(_BaseBOOptimizer, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

        self.num_init_samples = self.optimizer_settings.get('num_init_samples', 20)
        self.num_random_samples = self.optimizer_settings.get('num_random_samples', 100)
        self.archive: Union[Tuple[np.ndarray, np.ndarray], None] = None
        self.model = None

    def _init_design(self):
        # Sample K random configurations
        random_configs = self.cs.sample_configuration(self.num_init_samples)
        for random_config in random_configs:
            self.experiment.evaluate_configuration(configuration=random_config, fidelity=None)
        data = self.experiment.study.trials_dataframe()
        # TODO: Extract x and y data
        logger.info('Init Design finished')
        return data

    def run(self, **kwargs):
        raise NotImplementedError()

    def _model_predict(self, x):
        raise NotImplementedError()
    
    def _model_fit(self, x, y):
        raise NotImplementedError()

    def _optimize_acquisition_function(self):
        raise NotImplementedError()

    def _acquisition_function(self, samples: np.ndarray):
        raise NotImplementedError()


class _SklearnGP_BO(_BaseBOOptimizer):
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(_SklearnGP_BO, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

        self.model = GaussianProcessRegressor()
    
    def run(self, **kwargs):
        self.archive = self._init_design()
        self._model_fit(self.archive[0], self.archive[1])
        while True:
            new_x = self._optimize_acquisition_function()
            random_config = self.experiment.search_space.sample_configuration(1)
            self.experiment.evaluate_configuration(configuration=random_config, fidelity=None)

    def _model_predict(self, x):
        predictions, std = self.model.predict(x, return_std=True)
        return predictions, std

    def _model_fit(self, x, y):
        self.model.fit(x, y)

    def _optimize_acquisition_function(self):
        # calculate the acquisition function
        # Generate random configs to sample from the search space and take the one with the highest acquisition value
        # here prob of improvement
        random_configs: List[CS.Configuration] = self.cs.sample_configuration(self.num_random_samples)
        random_configs: np.ndarray = np.array([config.get_array() for config in random_configs])

        probs = self._acquisition_function(samples=random_configs)
        id_highest_prob = np.argmax(probs)
        new_sample = random_configs[id_highest_prob]
        return new_sample

    def _acquisition_function(self, samples: np.ndarray):
        raise NotImplementedError()


class GPWithPIOptimizer(_SklearnGP_BO):
    def _acquisition_function(self, samples: np.ndarray):
        # probability of improvement

        # calculate best so far seen surrogate value
        _y_predictions_mu, _y_predictions_std = self.model.predict(self.archive[0], return_std=True)
        best_seen = np.max(_y_predictions_mu)

        y_predictions_mu, y_predictions_std = self.model.predict(samples, return_std=True)
        probs = norm.cdf((y_predictions_mu - best_seen) / (y_predictions_std + 1E-9))

        return probs


class GPWithUCBOptimizer(_SklearnGP_BO):
    def _acquisition_function(self, samples: np.ndarray):
        # probability of improvement

        # calculate best so far seen surrogate value
        _y_predictions_mu, _y_predictions_std = self.model.predict(self.archive[0], return_std=True)
        best_seen = np.max(_y_predictions_mu)

        y_predictions_mu, y_predictions_std = self.model.predict(samples, return_std=True)
        probs = y_predictions_mu + y_predictions_std
        # probs = norm.cdf((y_predictions_mu - best_seen) / (y_predictions_std + 1E-9))
        return probs


from scipy.optimize import minimize
import torch
from typing import Callable
from botorch.models import SingleTaskGP


class GPWithPIAutoGradOptimizer(_BaseBOOptimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(GPWithPIAutoGradOptimizer, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

        self.model = None

    def _model_predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        posterior = self.model.posterior(x)
        mean, variance = posterior.mean, posterior.variance
        return mean, torch.sqrt(variance)

    def _model_fit(self, x, y):
        self.model = SingleTaskGP(torch.from_numpy(x), torch.from_numpy(y))

    def run(self, **kwargs):
        pass

    def _acquisition_function(self, samples: np.ndarray):

        from botorch.acquisition import ProbabilityOfImprovement
        _y_predictions_mu, _y_predictions_std = self._model_predict(self.archive[0])
        best_seen = torch.max(_y_predictions_mu)

        pi = ProbabilityOfImprovement(model=self.model, best_f=best_seen)
        probs = torch.tensor([pi(sample.view((-1, 1))) for sample in torch.from_numpy(samples)]).view((-1, 1)).requires_grad_()
        return probs

        y_predictions_mu, y_predictions_std = self._model_predict(samples)

        best_seen = best_seen.expand_as(y_predictions_mu)

        normal = torch.distributions.normal.Normal(
            torch.zeros_like(y_predictions_mu), torch.ones_like(y_predictions_mu)
        )
        probs = normal.cdf((y_predictions_mu - best_seen) / (y_predictions_std + 1E-9))
        return probs

    def _optimize_acquisition_function(self):
        def f_np_wrapper(x: np.ndarray, func: Callable):
            shape = (len(x), 1)
            x_torch = torch.from_numpy(x).view(shape).contiguous().requires_grad_(True)
            loss = func(x.reshape((-1, 1))).sum()
            grad_f = torch.autograd.grad(loss, x_torch)

            # grad_f = grad_f[0].contiguous().view(-1))
            return loss.item(), grad_f

        def f(x):
            return -self._acquisition_function(samples=x)

        constraints = None
        callback = None
        x0 = self.archive[0]
        bounds = [(hp.lower, hp.upper) for hp in self.cs.get_hyperparameters()] * len(x0)
        new_sample = minimize(
            fun=f_np_wrapper,
            args=(f,),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            # options={k: v for k, v in options.items() if k not in ["method", "callback"]},
        )

        return new_sample


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    def plot(optimizers, title, new_sample):
        f, ax = plt.subplots(ncols=len(optimizers), figsize=(10*len(optimizers), 8))
        for i, optimizer in enumerate(optimizers):
            ax[i].scatter(optimizer.archive[0], optimizer.archive[1], label='Observed')
            if new_sample is not None:
                ax[i].scatter(new_sample[0], new_sample[1], label='new data_point')
            Xsamples = np.arange(0, 1, 0.001)
            Xsamples = Xsamples.reshape(len(Xsamples), 1)
            ysamples, _ = optimizer._model_predict(Xsamples)
            if isinstance(ysamples, torch.Tensor):
                ysamples = ysamples.detach().numpy()
            ax[i].plot(Xsamples, ysamples, label='predicted', c='blue')

            probs = optimizer._acquisition_function(samples=Xsamples)
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().numpy()
            ax[i].plot(Xsamples, probs, label='acquisition_function', ls='dashed', c='red')

            y_true_samples = [objective({'x1': x}, noise=0) for x in Xsamples]
            ax[i].plot(Xsamples, y_true_samples, label='true', c='black', ls='dashed')
            ax[i].set_title(title)
            ax[i].legend()

        plt.show()
        plt.close()

    def objective(x: Dict, noise=0.01):
        noise = np.random.normal(loc=0, scale=noise)
        return (x['x1'] ** 2 * np.sin(5 * np.pi * x['x1']) ** 6.0) + noise

    def init_design(optimizer):
        random_configs = optimizer.cs.sample_configuration(optimizer.num_init_samples)
        x, y = [], []
        for random_config in random_configs:
            _y = objective(x=random_config)
            x.append(random_config['x1'])
            y.append(_y)

        logger.info('Init Design finished')
        return np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1))

    simple_cs = CS.ConfigurationSpace()
    simple_cs.add_hyperparameters([
        CS.UniformFloatHyperparameter('x1', lower=0, upper=1),
        # CS.UniformFloatHyperparameter('x2', lower=0, upper=1)
    ])

    pi_optimizer = GPWithPIOptimizer(
        optimizer_settings={}, benchmark_settings={}, configuration_space=simple_cs
    )
    ucb_optimizer = GPWithUCBOptimizer(
        optimizer_settings={}, benchmark_settings={}, configuration_space=simple_cs
    )
    pi_torch_optimizer = GPWithPIAutoGradOptimizer(
        optimizer_settings={}, benchmark_settings={}, configuration_space=simple_cs
    )
    
    optimizers = [pi_optimizer, pi_torch_optimizer]
    for optimizer in optimizers:
        optimizer.archive = init_design(optimizer)
        optimizer._model_fit(optimizer.archive[0], optimizer.archive[1])
    plot(optimizers, title='After Init', new_sample=None,)

    for i_config in range(101):
        for optimizer in optimizers:
            if np.random.random() > 0.3:
                new_x = optimizer._optimize_acquisition_function()
            else:
                new_x = np.array(optimizer.cs.sample_configuration()['x1'])
            new_y = objective({'x1': new_x})
            logger.info(f'New X: {new_x}')
            logger.info(f'New Y: {new_y}')

            if (i_config % 20) == 0:
                plot(optimizers, title=f'Sample {i_config}', new_sample=[new_x, new_y])

            optimizer.archive = np.concatenate([optimizer.archive[0], new_x.reshape((-1, 1))], axis=0),\
                                np.concatenate([optimizer.archive[1], new_y.reshape((-1, 1))], axis=0)
            optimizer._model_fit(optimizer.archive[0], optimizer.archive[1])
