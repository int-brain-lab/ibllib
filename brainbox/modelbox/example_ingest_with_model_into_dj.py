
from ibl_pipeline import behavior
from ibl_pipeline.modeling import behavior as dj_modeling


def insert_dataset(b, description=''):
    data_keys = b.fetch('KEY')
    dataset_hash = hash(data_keys)

    # TODO: check the availability of the dataset, if not insert the dataset.
    behavior_dj_modeling.BehaviorDataKeys.insert1(
        dict(dataset_hash=dataset_hash,
             data_keys=data_keys,
             data_description=description)



# IMPORT SOME DATA FROM DATAJOINT
b = behavior.TrialSet.Trial * (acquisition.Session & 'task_protocol LIKE "%biased%"') \
    * (subject.Subject & 'subject_nickname="CSHL_015"') * subject.SubjectLab()

insert_dataset(b, 'a cool data set')

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

def insert_model(psychfunc):
    psychfunc = PsychometricFunction(model_name='erf_2lapses')

    model = dict(**psychfunc.attributes)

    dj_modeling.insert1(model)


insert_model(psy)

insert_fitting_methods()

pop_rel = dict()

dj_modeling.FittingOutput.populate('model_name="erf_2lapses"', 'fitting=""', limit=1)


f = (dj_modeling.FittingOutPut & pop_rel).fetch()
