""" Authors: Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019
"""

import numpy as np


class Parameter:
    """Class containing all basic information for a model parameter. """

    def __init__(self, name, description='',
                 bounds_hard=[], range_plausible=[],
                 typical_value=[], parameterization=[]):
        """
        Input:
            name -- parameter name (string)
            description -- description for printing/plotting (string)
            bounds_hard -- [lower, upper] bounds, can be inf (np.array)
            range_plausible -- [lower, upper] plausible range (np.array)
            typical value -- example parameter value (float)
            parameterization -- type of parameterization (string)
        """

        self.name = name
        self.description = description

        if bounds_hard:
            self.bounds_hard = bounds_hard
        else:
            self.bounds_hard = np.array([-np.inf, np.inf])

        if range_plausible:
            self.range_plausible = range_plausible
        else:
            self.range_plausible = self.bounds_hard

        assert self.bounds_hard[0] <= self.bounds_hard[1], \
            "Lower bound cannot be higher than upper bound."

        assert self.range_plausible[0] <= self.range_plausible[1], \
            "Lower plausible range cannot be higher than upper plausible range."

        assert self.bounds_hard[0] <= self.range_plausible[0], \
            "Lower plausible range cannot be lower than hard lower bound."

        assert self.range_plausible[1] <= self.bounds_hard[1], \
            "Higher plausible range cannot be higher than hard upper bound."

        assert np.all(np.isfinite(self.range_plausible)), \
            "Parameter plausible range should be finite."

        if typical_value:
            self.typical_value = typical_value
        else:
            self.typical_value = (self.range_plausible[0] + self.range_plausible[1]) / 2.

        assert self.typical_value >= self.range_plausible[0] and \
            self.typical_value <= self.range_plausible[1], \
            "Typical value should be included in the plausible range."

        if parameterization:
            self.parameterization = parameterization
        else:
            self.parameterization = 'standard'

    def __str__(self):
        if self.description:
            _str = self.description
        else:
            _str = self.name
        return _str

    def __repr__(self):
        _str = "Parameter '" + self.name + "'"
        if self.description:
            _str += ': ' + self.description
        _str += '\n'
        _str += 'Hard bounds: (' + str(self.bounds_hard[0]) + ', ' + \
            str(self.bounds_hard[1]) + '); '
        _str += 'Plausible range: [' + str(self.range_plausible[0]) + ', ' + \
            str(self.range_plausible[1]) + ']; '
        _str += 'Typical value: ' + str(self.typical_value) + '\n'
        _str += 'Parameterization: ' + self.parameterization
        return _str


# p = Parameter('test', description='Test parameter', range_plausible=[-0.5, 0.5])
