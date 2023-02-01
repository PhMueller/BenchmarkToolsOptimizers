from copy import deepcopy
from typing import Dict, Union

import ConfigSpace as CS
from omegaconf import OmegaConf

from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.constants import MAXINT
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO
from smac.initial_design import SobolInitialDesign

from BenchmarkTools import logger
from BenchmarkToolsOptimizers.optimizers.base_optimizer import Optimizer
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment
from BenchmarkTools.core.constants import BenchmarkToolsTrackMetrics


MAP_MULTI_OBJECTIVE_ALGORITHMS = {
    'ParEGO': ParEGO,
    'MeanAggregationStrategy': MeanAggregationStrategy
}
MAP_SMAC_FACADE = {
    'BlackBoxFacade': BlackBoxFacade,  # GP
    'HyperparameterOptimizationFacade': HyperparameterOptimizationFacade  # RF
}


class SMACOptimizer(Optimizer):
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):

        super(SMACOptimizer, self).__init__(optimizer_settings, benchmark_settings, configuration_space, **kwargs)
        self.algorithm = None
        self.experiment: Union[MultiObjectiveExperiment, None] = None
        self._objective_function = None
        self.scenario = None

    def init(self, seed: int = 0, **kwargs):
        super(SMACOptimizer, self).init(seed=seed)

        if self.experiment.bookkeeper.tae_limit is None:
            logger.warning(f'TAE limit is not set, but SMAC does require one. Set it to 100k.')
            self.experiment.bookkeeper.tae_limit = 100000

        # Extract the highest fidelity, since the current MO version of SMAC only supports full fidelity evaluations
        fidelity_space: CS.ConfigurationSpace = self.experiment.benchmark.get_fidelity_space()
        if fidelity_space is None or len(fidelity_space) == 0:
            max_fidelity = None
        else:
            max_fidelity = {hp.name: hp.upper for hp in fidelity_space.get_hyperparameters()}

        # TODO: think about adding a wrapper for tracking the optimization limits.
        #       Currently optimizer, benchmark and experiment are too much connected.
        def smac_objective_function(cfg: Dict, seed: int = 0) -> Dict[str, float]:
            # Define the objective function as minimization problem.
            # If the benchmark is "maximization", negate the returned values, since SMAC does minimize.
            result_dict = self.experiment.evaluate_configuration(configuration=cfg, fidelity=max_fidelity)
            result_dict[BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD] = {
                o_name: (-1 if o_dir == 'maximize' else 1) * result_dict['function_value'][o_name]
                for o_name, o_dir in zip(self.experiment.objective_names, self.experiment.directions)
            }
            return result_dict[BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD]

        self._objective_function = smac_objective_function

        # --------------------- INIT SMAC OPTIMIZER --------------------------------------------------------------------
        # Load all smac dependent settings and instantiate the MO algorithm (e.g. ParEGO) and the correct
        # facade/surrogate model (e.g. RF or GP)
        algorithm_parameters = deepcopy(self.optimizer_settings['optimizer_parameters'])
        algorithm_parameters = OmegaConf.to_container(algorithm_parameters, resolve=True)
        algorithm_facade_name = algorithm_parameters['smac_facade']
        algorithm_facade_type = MAP_SMAC_FACADE[algorithm_facade_name]

        algorithm_aggregation_name = algorithm_parameters['mo_algorithm']['name']
        algorithm_aggregation_params = algorithm_parameters['mo_algorithm'].get('parameters', {})
        algorithm_aggregation_type = MAP_MULTI_OBJECTIVE_ALGORITHMS[algorithm_aggregation_name]

        # Shift the seed to have different initial designs
        if algorithm_facade_name == 'BlackBoxFacade':
            seed += 1000
        if algorithm_aggregation_name == 'ParEGO':
            seed += 10000
            algorithm_aggregation_params['seed'] = seed

        # Create the scenario object. We use it sequentially.
        self.scenario = Scenario(
            configspace=self.cs,
            output_directory=self.experiment.output_path,
            deterministic=algorithm_parameters['deterministic'],
            objectives=self.experiment.objective_names,
            crash_cost=[float(MAXINT), float(MAXINT)],
            n_trials=self.experiment.bookkeeper.tae_limit,
            walltime_limit=self.experiment.bookkeeper.wallclock_limit_in_s,
            n_workers=1,  # TODO: add support for multiple workers.
            seed=seed
        )

        algorithm_aggregation = algorithm_aggregation_type(
            scenario=self.scenario, **algorithm_aggregation_params
        )

        initial_design = SobolInitialDesign(
            scenario=self.scenario,
            n_configs=max(1, min(20, 0.1 * self.scenario.n_trials)),
            # n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_ratio=0.1,
            # additional_configs=additional_configs,
            seed=self.scenario.seed,
        )

        self.algorithm = algorithm_facade_type(
            scenario=self.scenario,
            target_function=self._objective_function,
            initial_design=initial_design,
            multi_objective_algorithm=algorithm_aggregation,
            overwrite=True,
        )
        # --------------------- INIT SMAC OPTIMIZER --------------------------------------------------------------------

    def run(self, **kwargs):
        logger.info('Start Optimization Run')
        incumbent = self.algorithm.optimize()
        logger.info(f'Finished Optimization Run with incumbent {incumbent}')
