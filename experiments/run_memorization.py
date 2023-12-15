import os
import numpy as np
import matplotlib.pyplot as plt
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.darts import get_arima_predictions_data

gpt3_hypers = dict(
    temp=.7,
    alpha=[0.5, .7, 0.9, 0.99],
    beta=[0, .15, 0.3, .5],
    basic=[False],
    settings=[SerializerSettings(base=10, prec=prec, signed=True,half_bin_correction=True) for prec in [2,3]],
)

gp_hypers = dict(lr=[5e-3, 1e-2, 5e-2, 1e-1])

arima_hypers = dict(p=[12,20,30], d=[1,2], q=[0,1,2])

TCN_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    kernel_size=[3, 5], num_filters=[1, 3], 
    likelihood=['laplace', 'gaussian']
)

NHITS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)


NBEATS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)


model_hypers = {
    'gp': gp_hypers,
    'arima': arima_hypers,
    'TCN': TCN_hypers,
    'N-BEATS': NBEATS_hypers,
    'N-HiTS': NHITS_hypers,
    'text-davinci-003': {'model': 'text-davinci-003', **gpt3_hypers},
}

model_predict_fns = {
    'gp': get_gp_predictions_data,
    'arima': get_arima_predictions_data,
    'TCN': get_TCN_predictions_data,
    'N-BEATS': get_NBEATS_predictions_data,
    'N-HiTS': get_NHITS_predictions_data,
    'text-davinci-003': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003']])


import pickle
import matplotlib.pyplot as plt
from data.small_context import get_memorization_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

output_dir = 'outputs/memorization'
os.makedirs(output_dir, exist_ok=True)

datasets = get_memorization_datasets(predict_steps=30)
for dsname,data in datasets.items():
    train, test = data
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in ['text-davinci-003', 'gp', 'arima', 'N-HiTS']:
        if model in out_dict and not is_gpt(model):
            if out_dict[model]['samples'] is not None:
                print(f"Skipping {dsname} {model}")
                continue
            else:
                print('Using best hyper...')
                hypers = [out_dict[model]['best_hyper']]
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_gpt(model) else False
        num_samples = 20 if is_gpt(model) else 100
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=0, parallel=parallel)
            if preds.get('NLL/D', np.inf) < np.inf:
                out_dict[model] = preds
            else:
                print(f"Failed {dsname} {model}")
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    

    print(f"Finished {dsname}")
    