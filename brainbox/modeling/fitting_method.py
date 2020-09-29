""" by Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019
"""
import scipy as sp


class FittingMethod:
    """ Abstract class with wrappers to e.g. skikit-learn functions
    Attributes:
        name # string
        algorithm # function handle
    Methods:
        fit
    """


class MaximumLikelihoodEstimation(FittingMethod):
    """ Maximum Likelihood Estimation
    """

    def __init__(self, data, model):
        self.name = 'maximum_likelihood_estimation'
        self.data = data
        self.model = model
        self.result = {}

    def fit(self):

        print('Maximum Likelihood fitting')
        # get the bounds and typical value
        x0 = [p.typical_value for p in self.model.parameter_list]
        bounds = [[p.bounds_hard[0], p.bounds_hard[1]] for p in self.model.parameter_list]

        def neg_loglikelihood_function(x, _model, _data):
            ll = _model.loglikelihood_function(x, _model, _data)
            return -1. * ll

        # run actual optimization
        res = sp.optimize.minimize(neg_loglikelihood_function, x0, args=(self.model, self.data),
                                   bounds=bounds, method='L-BFGS-B')

        # save the things we really care about
        self.result['parameters'] = res.x
        self.result['neg_loglikelihood'] = res.fun
        self.result['output'] = res
        self.result['algorithm'] = 'scipy_optimize_minimize'

    def plot(self, plot_data=True, plot_fit=True, **kwargs):

        # THIS JUST POINTS TO THE PLOTTING CODE IN MODEL
        assert hasattr(self, 'result'), 'Call .fit() before .plot() on a model'
        self.model.plot(self, plot_fit=True, **kwargs)

# class PosteriorEstimation(FittingMethod):
#     """ Maximum Likelihood Estimation
#     """
