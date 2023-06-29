


def tuner(build_model):
   tuner = RandomSearch(build_model,
    objective='val_los',
    max_trials=5,
    executions_per_trial=3,
    directory='ml_models',
    project_name='price_optimizers')

   return tuner 