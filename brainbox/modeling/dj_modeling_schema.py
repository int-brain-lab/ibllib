import datajoint as dj
from ibl_pipeline import subject, acquisition, behavior


schema = dj.schema('ibl_modeling')


@schema
class BehavioralDataKeys(dj.Manual):
    definition = """
    # Table saves primary keys of data sets arbituarily defined
    dataset_id          : int           # dataset index
    ---
    data_keys           : longblob      # list of dictoinaries of the primary keys for the current data set.
    dataset_description : varchar(511)  # explain the rationale of this dataset grouping
    """


@schema
class PreprocessedBehavioralDataSet(dj.Computed):
    definition = """
    -> BehavioralDataKeys
    ---
    dataset:    longblob     # currently as a list of dictionaries, DJ will support pandas dataframe in the future
    # with columns:
    # stimulus_side       : values to be -1 or 1
    # stimulus_strength   : non-negative contrast
    # choice              : -1 and 1, nan for missed trials
    # rewarded            : 0 and 1, including 0 contrast trials
    # ---
    # optional columns:
    # correct             : 1 for correct, 0 for incorrect, nan for 0 contrast trials or missed trials
    # reaction_time       : time diff of response time - stim on time
    # prob_left_block     : probability (in block structure) of stimulus_side == -1
    # trial_id            :
    # session_id          :
    """


@schema
class Model(dj.Manual):
    definition = """
    model_name          : varchar(32)           # name of this model
    ---
    model_description   : varchar(255)          # description of this model
    loglikelihood_function : varchar(128)       # function handle to be evaluated
    """

    class Parameter(dj.Part):
        definition = """
        # Parameters defined by this model
        -> master
        param_name                  : varchar(32)         # name of the parameter
        ---
        params_hard_bounds          : blob    # hard bounds of the parameters [lower, upper]
        params_plausible_range      : blob    # plausible range of the parameters [lower, upper]
        params_typical_value        : float   #
        param_description=null      : varchar(255)   # description
        parameterization='standard' : enum('standard', 'log')  # scale of the parameter
        """


@schema
class FittingType(dj.Lookup):
    definition = """
    # Method to fit a dataset
    fitting_type  : varchar(16)
    """
    contents = [
        ['mle', ''],
        ['posterior', '']
    ]


@schema
class OptimizationMethod(dj.Lookup):
    definition = """
    opt_method:  varchar(16)
    ---
    opt_algorithm:   varchar(32)    # function handle
    """


@schema
class FittedOutput:
    definition = """
    # results of a model fit
    -> PreprocessedBehavioralDataSet
    -> Model
    -> FittingType
    -> OptimizationMethod
    ---
    starting_points : longblob    # num_startingpoints x num_params
    loglikelihoods  : longblob    # num_startingpoints
    maximum_points  : longblob    # num_startingpoints x num_params
    """
