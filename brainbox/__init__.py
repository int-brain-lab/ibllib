try:
    import one
except ModuleNotFoundError:
    logging.getLogger('ibllib').error('Missing dependency, please run `pip install ONE-api`')
