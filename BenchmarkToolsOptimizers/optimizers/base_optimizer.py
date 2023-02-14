from typing import Callable, Dict, Union, Tuple, List
import ConfigSpace as CS

from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment


class Optimizer:
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
        self.supports_ask_and_tell = False

    def link_experiment(self, experiment):
        self.experiment = experiment

    def init(self, seed: int = 0, **kwargs):
        self.seed = seed

    def run(self, **kwargs):
        raise NotImplementedError()


class AskAndTellOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super(AskAndTellOptimizer, self).__init__(**kwargs)
        self.supports_ask_and_tell = True

    def ask(self) -> Tuple[List[Dict], List[Dict]]:
        raise NotImplementedError()

    def tell(self, jobs: List[Job]) -> None:
        raise NotImplementedError()

    def run(self, **kwargs):
        raise ValueError(
            'Please do not use the run() function for optimizers which support the ask-and-tell-interface.'
        )
