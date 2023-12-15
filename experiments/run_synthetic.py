import os
import pickle
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.darts import get_arima_predictions_data
from data.synthetic import get_synthetic_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

# Specify the hyperparameter grid for each model
gpt3_hypers = dict(
    model='text-davinci-003',
    alpha=0.1,
    basic=True,
    settings= SerializerSettings(10, prec=3,signed=True)
)

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)


llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

gp_hypers = dict(lr=[1e-2])

arima_hypers = dict(p=[12,20,30], d=[1,2], q=[0,1,2])

TCN_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    kernel_size=[3, 5], num_filters=[1, 3], 
    likelihood=['laplace', 'gaussian']
)

NHITS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)


NBEATS_hypers = dict(in_len=[10, 100, 400], out_len=[1], # 10 is almost always the best
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
    'gpt-4': {'model': 'gpt-4', **gpt4_hypers},
    'llama-70b': {'model': 'llama-70b', **llama_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'gp': get_gp_predictions_data,
    'arima': get_arima_predictions_data,
    'TCN': get_TCN_predictions_data,
    'N-BEATS': get_NBEATS_predictions_data,
    'N-HiTS': get_NHITS_predictions_data,
    'text-davinci-003': get_llmtime_predictions_data,
    'gpt-4': get_llmtime_predictions_data,
    'llama-70b': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/synthetic'
os.makedirs(output_dir, exist_ok=True)

datasets = get_synthetic_datasets()
for dsname,data in datasets.items():
    if dsname in ['rbf_0','rbf_1','matern_0','matern_1']:
        continue
    train, test = data
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in ['text-davinci-003', 'gpt-4', 'arima', 'TCN']:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        print(f"Starting {dsname} {model}")
        hypers = list(grid_iter(model_hypers[model]))
        
        parallel = True if is_gpt(model) else False
        num_samples = 20 if is_gpt(model) else 100
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    

    print(f"Finished {dsname}")
    