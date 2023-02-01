"""
pip install ax-platform
"""

from collections.abc import MutableMapping
from typing import Any, Iterable, Set
from typing import Dict, Optional
from typing import List, Union

import ConfigSpace as CS
import ax
from ax import Models, MultiObjectiveOptimizationConfig, Experiment as AxExperiment, Runner, Trial, \
    ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace, core
from ax.core.base_trial import TrialStatus
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_NEHVI, get_MOO_PAREGO

from BenchmarkTools import logger
from BenchmarkTools.core.constants import BenchmarkToolsTrackMetrics
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment
from BenchmarkToolsOptimizers.optimizers.base_optimizer import Optimizer
from BenchmarkTools.core.exceptions import BudgetExhaustedException


class BoTorch_Optimizer(Optimizer):
    def __init__(self,
                 optimizer_settings: Dict,
                 benchmark_settings: Dict,
                 configuration_space: CS.ConfigurationSpace,
                 **kwargs,
                 ):

        super(BoTorch_Optimizer, self).__init__(optimizer_settings, benchmark_settings, configuration_space, **kwargs)

        self.algorithm_func = None
        self.seed = 0
        self.seed_offset = 0
        self.batch_size = self.optimizer_settings['optimizer_parameters'].get('batch_size', 1)

        # Ax related objects
        self.ax_searchspace: Union[SearchSpace, None] = None
        self.ax_experiment: Union[AxExperiment, None] = None
        self.ax_observation_data = None
        self.ax_runner = None

    def link_experiment(self, experiment: MultiObjectiveExperiment):
        self.experiment = experiment

    def init(self, seed: int = 0, **kwargs):
        self.seed = seed + self.seed_offset

        self.ax_searchspace = convert_config_space_to_ax_space(self.experiment.search_space)
        optimization_config = get_ax_optimization_config(self.benchmark_settings)
        tracking_metrics = get_ax_tracking_metrics(self.benchmark_settings)
        experiment = self.experiment

        class BenchmarkToolsRunner(Runner):

            def run(self, trial: Trial) -> Dict[str, Any]:

                # Query the experiment
                result_dict = experiment.evaluate_configuration(configuration=trial.arms[0].parameters, fidelity=None)

                # Filter out non-selected objectives
                result_dict[BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD] = {
                    k: v for k, v in result_dict[BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD].items()
                    if k in experiment.objective_names
                }

                # Filter out non-tracked metrics
                result_dict = {
                    k: v for k, v in result_dict.items()
                    if k in [BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD, BenchmarkToolsTrackMetrics.COST]
                }

                # to make it understandable by Ax, flatten the dictionary
                result_dict = _flatten(result_dict)
                return result_dict

            def stop(self, trial: core.base_trial.BaseTrial, reason: Optional[str] = None) -> Dict[str, Any]:
                return trial.run_metadata

            def poll_trial_status(self, trials: Iterable[ax.core.base_trial.BaseTrial]) -> Dict[TrialStatus, Set[int]]:
                return {TrialStatus.COMPLETED: {t.index for t in trials}}

        self.ax_runner = BenchmarkToolsRunner()
        self.ax_experiment = AxExperiment(
            name=self.experiment.study_name,
            search_space=self.ax_searchspace,
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
            runner=self.ax_runner
        )

    def _eval_one_trial(self, trial: ax.Trial):
        function_values = trial.run()
        data = ax.Data.from_evaluations(evaluations={trial.arm.name: trial.run_metadata}, trial_index=trial.index)
        trial = trial.mark_completed()
        self.ax_experiment.attach_data(data, combine_with_last_data=True)
        self.ax_observation_data = self.ax_experiment.fetch_data()

    def initial_design(self):
        sobol = Models.SOBOL(search_space=self.ax_searchspace, seed=self.seed)

        for _ in range(self.optimizer_settings['optimizer_parameters']['initial_samples']):
            trial = self.ax_experiment.new_trial(generator_run=sobol.gen(1))
            self._eval_one_trial(trial)
        logger.info('Init Design: DONE')

    def _new_config_already_evaluated(self, new_config: Dict) -> bool:
        """
        It happens that the models generate multiple times the same configuration.
        Unfortunately ax is not that robust and crashes then.

        This function goes through the already evaluated configurations and checks if we have seen that configuration
        already.
        """
        already_evaluated = False
        for trial in self.ax_experiment.trials.values():
            if new_config == trial.arm.parameters:
                return True
        return False

    def _run(self):
        self.initial_design()

        # This loop will be interrupted in the `experiment.evaluate_configuration` function by the bookkeeper
        while True:

            # Fit a BO model on the observed data
            model = self.algorithm_func(
                experiment=self.ax_experiment,
                data=self.ax_observation_data,
                search_space=self.ax_searchspace
            )

            # Allow for batch generations. To make it comparable to the other optimizers, we evaluate them sequentially.
            generated_runs = model.gen(self.batch_size)

            # Evaluate the generated configurations. Note that currently we do this sequentially here.
            for new_arm in generated_runs.arms:
                # Check if configuration is already evaluated:
                if self._new_config_already_evaluated(new_config=new_arm.parameters):
                    logger.info(f"Sampled Config: {new_arm.parameters} already seen: SKIP")
                    continue

                logger.info(f"Evaluate new config: {new_arm.parameters}")
                new_trial = self.ax_experiment.new_trial(None)
                new_trial.add_arm(new_arm)
                self._eval_one_trial(new_trial)

    def run(self, **kwargs):
        try:
            self._run()
        except BudgetExhaustedException:
            # If the optimization limits have been reached, we can gracefully shut down here.
            pass
        except Exception as e:
            # Check if at least some configurations, else than the random configs from the initial design,
            # have been evaluated (we add a small margin of 3 here)
            n_threshold = self.optimizer_settings['optimizer_parameters']['initial_samples'] + 3

            if self.experiment.bookkeeper.num_tae_calls < n_threshold:
                logger.warning(
                    f'CRASH + Number of evaluated configs lower than threshold {n_threshold}. '
                    f'At least 10 configs have to be sampled. '
                    f'But only {self.experiment.bookkeeper.num_tae_calls} have been selected.'
                )
                raise e

            else:
                # Botorch collapses sometimes (Gradients are all Nan). This happens when the optimizer is converged and
                # no new configurations are sampled.
                # Proposed solution from https://github.com/facebook/Ax/issues/576
                # However, if enough configurations have been evaluated, we allow that here.
                import traceback
                tb = traceback.format_exc()
                logger.info(
                    f'Stop optimizer gracefully due to Error in Botorch. '
                    f'A total of {self.experiment.bookkeeper.num_tae_calls} have been evaluated.'
                )
                logger.warning(f'{tb}')


class QEHVI(BoTorch_Optimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs
    ):
        super(QEHVI, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

        self.algorithm_func = get_MOO_EHVI
        self.seed_offset = 3100


class QNParEgo(BoTorch_Optimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs
    ):
        super(QNParEgo, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

        self.algorithm_func = get_MOO_PAREGO
        self.seed_offset = 3200


class QNEHVI(BoTorch_Optimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configuration_space: CS.ConfigurationSpace,
            **kwargs,
    ):
        super(QNEHVI, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configuration_space=configuration_space,
            **kwargs
        )

        self.algorithm_func = get_MOO_NEHVI
        self.seed_offset = 3300


def get_ax_parameter_type(item) -> Union[ParameterType, int]:
    value_type = type(item)
    if isinstance(item, bool):
        return ParameterType.BOOL
    elif type(item) is int:
        return ParameterType.INT
    elif type(item) is float:
        return ParameterType.FLOAT
    elif type(item) is str:
        return ParameterType.STRING
    else:
        raise ValueError(f'Unknown type: {type(item)}')


def convert_config_space_to_ax_space(cs_searchspace: CS.ConfigurationSpace) -> ax.SearchSpace:
    """
    Given an ConfigSpace.ConfigurationSpace create an Ax.SearchSpace from it.
    """
    ax_space_dict = {}

    for hp_name, hp in cs_searchspace.get_hyperparameters_dict().items():
        if isinstance(hp, CS.CategoricalHyperparameter):
            values = hp.choices
            if not isinstance(values, List):
                values = list(values)
            parameter_type = get_ax_parameter_type(values[0])

            # A categorical with a single value can be replaced by a constant!
            if len(values) == 1:
                ax_space_dict[hp_name] = FixedParameter(
                    name=hp_name, parameter_type=parameter_type, value=values[0]
                )
            else:
                ax_space_dict[hp_name] = ChoiceParameter(
                    name=hp_name, values=values, parameter_type=parameter_type, is_ordered=False, sort_values=False
                )

        elif isinstance(hp, CS.OrdinalHyperparameter):
            values = hp.sequence
            if not isinstance(values, List):
                values = list(values)
            parameter_type = get_ax_parameter_type(values[0])
            ax_space_dict[hp_name] = ChoiceParameter(    # are already sorted
                name=hp_name, values=values, parameter_type=parameter_type, is_ordered=True, sort_values=False
            )

        elif isinstance(hp, (CS.NormalIntegerHyperparameter, CS.UniformIntegerHyperparameter)):
            ax_space_dict[hp_name] = RangeParameter(
                name=hp_name, parameter_type=ParameterType.INT, lower=hp.lower, upper=hp.upper, log_scale=hp.log
            )

        elif isinstance(hp, (CS.NormalFloatHyperparameter, CS.UniformFloatHyperparameter)):
            ax_space_dict[hp_name] = RangeParameter(
                name=hp.name, parameter_type=ParameterType.FLOAT, lower=hp.lower, upper=hp.upper,log_scale=hp.log
            )
        elif isinstance(hp, CS.Constant):
            parameter_type = get_ax_parameter_type(hp.value)
            ax_space_dict[hp_name] = FixedParameter(
                name=hp.name, parameter_type=parameter_type, value=hp.value
            )

    ax_space = SearchSpace(parameters=list(ax_space_dict.values()))
    return ax_space


def get_ax_optimization_config(benchmark_settings: Dict) -> MultiObjectiveOptimizationConfig:
    """ Used to initialize an Ax.Experiment. """
    objective_settings: List[Dict] = benchmark_settings['objectives']

    objectives = []
    thresholds = []

    for entry in objective_settings:
        metric = ax.Metric(entry['name'], lower_is_better=entry['lower_is_better'])

        objective = ax.Objective(metric)
        objectives.append(objective)

        threshold_value = min(entry['limits']) if entry['lower_is_better'] else max(entry['limits'])
        threshold = ax.ObjectiveThreshold(metric, threshold_value, relative=False)
        thresholds.append(threshold)

    multi_objective = ax.MultiObjective(objectives)
    
    mo_opt_config = MultiObjectiveOptimizationConfig(
        objective=multi_objective,
        objective_thresholds=thresholds,
    )
    
    return mo_opt_config


def get_ax_tracking_metrics(benchmark_settings: Dict) -> List:
    """ used to initialize an Ax.Experiment """
    track_settings: List = benchmark_settings.get('track_metrics', [])

    track_metrics = []
    for entry in track_settings:
        metric = ax.Metric(name=entry['name'], lower_is_better=entry['lower_is_better'])
        track_metrics.append(metric)

    if BenchmarkToolsTrackMetrics.COST not in track_metrics:
        metric = ax.Metric(name=BenchmarkToolsTrackMetrics.COST, lower_is_better=True)
        track_metrics.append(metric)
    return track_metrics


def _flatten(d: Union[MutableMapping, Dict], parent_key: str = '', sep: str = '_') -> Dict:
    """ Flattens a nested dictionary """
    items = []
    for k, v in d.items():
        # new_key = parent_key + sep + k if parent_key else k
        new_key = k
        if isinstance(v, MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
