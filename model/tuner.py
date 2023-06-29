from kerastuner.tuners import RandomSearch

def tuner(build_model_func):
   tuner = RandomSearch(build_model_func,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='ml_models',
    project_name='price_optimizers')

   return tuner 