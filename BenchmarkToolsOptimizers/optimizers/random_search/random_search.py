from typing import Dict, Union

import ConfigSpace as CS

from BenchmarkToolsOptimizers.optimizers.base_optimizer import Optimizer


class BaseRandomOptimizer(Optimizer):
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(BaseRandomOptimizer, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

    def run(self, **kwargs):
        raise NotImplementedError()


class RandomSearchOptimizer(BaseRandomOptimizer):
    def run(self, **kwargs):
        while True:
            random_config = self.experiment.search_space.sample_configuration(1)
            self.experiment.evaluate_configuration(configuration=random_config, fidelity=None)
