import os
import pickle
from data.monash import get_datasets
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
import numpy as np
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

# Specify the hyperparameter grid for each model
gpt3_hypers = dict(
    temp=0.7,
    alpha=0.9,
    beta=0,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)

llama_hypers = dict()

model_hypers = {
    'text-davinci-003': {'model': 'text-davinci-003', **gpt3_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'text-davinci-003': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/monash'
os.makedirs(output_dir, exist_ok=True)

datasets_to_run =  [
    "covid_deaths", "solar_weekly", "tourism_monthly", "australian_electricity_demand", "pedestrian_counts",
    "traffic_hourly", "hospital", "fred_md", "tourism_yearly", "tourism_quarterly", "us_births",
    "nn5_weekly","solar_10_minutes", "traffic_weekly", "saugeenday", "cif_2016",
]

max_history_len = 500
datasets = get_datasets()
for dsname in datasets_to_run:
    print(f"Starting {dsname}")
    data = datasets[dsname]
    train, test = data
    train = [x[-max_history_len:] for x in train]
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in ['text-davinci-003']:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_gpt(model) else False
        num_samples = 5
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            medians = preds['median']
            targets = np.array(test)
            maes = np.mean(np.abs(medians - targets), axis=1) # (num_series)        
            preds['maes'] = maes
            preds['mae'] = np.mean(maes)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    print(f"Finished {dsname}")
    