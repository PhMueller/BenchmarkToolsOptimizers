from typing import Callable, Dict, Union
import ConfigSpace as CS

from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment


class BenchmarkToolsOptimizer:
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):

        self.optimizer_settings = optimizer_settings
        self.benchmark_settings = benchmark_settings
        self.cs = configuration_space
        self.experiment: Union[MultiObjectiveExperiment, None] = None
        self.seed = 0

    def link_experiment(self, experiment):
        self.experiment = experiment

    def init(self, seed: int = 0, **kwargs):
        self.seed = seed

    def run(self, objective_function: Callable, **kwargs):
        raise NotImplementedError()
