from typing import List, Tuple, Dict, Union

from BenchmarkToolsOptimizers.optimizers.base_optimizer import AskAndTellOptimizer
from BenchmarkToolsOptimizers.optimizers.hebo.configspace_tools import \
    hebo_config_to_configspace_config, configspace_cs_to_hebo_cs

from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.general import GeneralBO

from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.constants import BenchmarkToolsTrackMetrics


class HEBOOptimizer(AskAndTellOptimizer):
    def __init__(self, **kwargs):
        super(HEBOOptimizer, self).__init__(**kwargs)

        self.hebo_cs: Union[DesignSpace, None] = None
        self.algorithm: Union[HEBO, None] = None

        # ############################ PARAMETERS TO SPECIFY IN YAML ###################################################
        self.n_suggestions = self.optimizer_settings['optimizer_parameters'].get('n_suggestions', 1)

        default_general_algorithm_parameters = {
            'kappa': 2.,
            'c_kappa': 0.,
            'use_noise': False,
        }
        _general_algorithm_parameters = self.optimizer_settings['optimizer_parameters'].get('general_algorithm_parameters', {})
        self.general_algorithm_parameters = default_general_algorithm_parameters
        self.general_algorithm_parameters.update(**_general_algorithm_parameters)

        self.model_parameters = self.optimizer_settings['optimizer_parameters']['model_parameters']
        if isinstance(self.model_parameters, DictConfig):
            self.model_parameters = OmegaConf.to_container(self.model_parameters)
        self.model_name = self.model_parameters.pop('model_name')

        # ############################ PARAMETERS TO SPECIFY IN YAML ###################################################

    def init(self, seed: int = 0, **kwargs):
        self.seed = seed
        self.hebo_cs = configspace_cs_to_hebo_cs(self.cs)

        self.algorithm = GeneralBO(
            space=self.hebo_cs,
            num_obj=len(self.experiment.objective_names),
            num_constr=0,
            model_name=self.model_name,
            model_conf=self.model_parameters,
            **self.general_algorithm_parameters
        )
        # self.algorithm = HEBO(self.hebo_cs)

    def ask(self) -> Tuple[List[Dict], List[Dict]]:
        suggestions_df = self.algorithm.suggest(n_suggestions=self.n_suggestions)

        suggestions_list = suggestions_df.to_dict('records')
        suggestions_list = [
            hebo_config_to_configspace_config(configuration, self.cs) for configuration in suggestions_list
        ]

        return suggestions_list, [{} for _ in range(len(suggestions_list))]

    def tell(self, jobs: List[Job]) -> None:

        configurations = []
        performances_list = []

        for job in jobs:
            metrics = [
                job.result_dict[BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD][metric]
                for metric in self.experiment.objective_names
            ]
            performances_list.append(metrics)

            configurations.append(job.configuration)

        configuration_df = pd.DataFrame(configurations)
        performances = np.array(performances_list).reshape((-1, len(self.experiment.objective_names)))

        # Stack a 0 column to the performance list (since we do not use constraints)
        c = np.zeros((performances.shape[0], 0))
        performances = np.hstack([performances, c])

        self.algorithm.observe(X=configuration_df, y=performances)
