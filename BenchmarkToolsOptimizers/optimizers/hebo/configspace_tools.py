from typing import Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    NormalFloatHyperparameter, NormalIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
from hebo.design_space.design_space import DesignSpace


def hebo_config_to_configspace_config(hebo_config: Dict, configspace_cs: CS.ConfigurationSpace):

    configspace_config = {}
    for hp in configspace_cs.get_hyperparameters():
        value = hebo_config[hp.name]
        if isinstance(hp, OrdinalHyperparameter):
            value = hp.sequence[value]

        configspace_config[hp.name] = value

    config = CS.Configuration(configspace_cs, configspace_config, allow_inactive_with_values=True)
    config = deactivate_inactive_hyperparameters(config, configspace_cs)
    return config


def configspace_cs_to_hebo_cs(configspace_cs: CS.ConfigurationSpace) -> DesignSpace:
    hebo_parameters = []
    for hp in configspace_cs.get_hyperparameters():
        if isinstance(hp, (UniformIntegerHyperparameter, NormalIntegerHyperparameter)):
            if hp.log:
                _hp_dependent = {'type': 'pow_int', 'base': np.e, 'lb': hp.lower, 'ub': hp.upper}
            else:
                _hp_dependent = {'type': 'int', 'lb': hp.lower, 'ub': hp.upper}

        elif isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
            if hp.log:
                _hp_dependent = {'type': 'pow', 'base': np.e, 'lb': hp.lower, 'ub': hp.upper}
            else:
                _hp_dependent = {'type': 'num', 'lb': hp.lower, 'ub': hp.upper}

        elif isinstance(hp, OrdinalHyperparameter):
            _hp_dependent = {'type': 'int', 'lb': 0, 'ub': len(hp.sequence)}

        elif isinstance(hp, CategoricalHyperparameter):
            _hp_dependent = {'type': 'cat', 'categories': hp.choices}

        elif isinstance(hp, Constant):
            _hp_dependent = {'type': 'cat', 'categories': [hp.value]}
        else:
            raise ValueError(f'unknown hp type: {type(hp)}')

        _entry = {**{'name': hp.name}, **_hp_dependent}
        hebo_parameters.append(_entry)

    design_space = DesignSpace().parse(hebo_parameters)
    return design_space
