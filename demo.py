import os
import torch
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import openai
from time import perf_counter
#openai.api_key = os.environ['OPENAI_API_KEY']   #Comment if you don't have an API key yet
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data


def plot_prds_ploty(title,train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Truth'))
    fig.add_trace(go.Scatter(x=pred.index, y=pred, mode='lines', name=model_name))
    # shade 90% confidence interval
    samples = pred_dict['samples']
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            fig.add_trace(go.Scatter(x=pred.index, y=samples[i], mode='lines', line_color='rgba(0,0,0,0.3)'))
    fig.update_layout(title=model_name, xaxis_title='Date', yaxis_title=title, showlegend=True)
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            fig.update_layout(title= f'NLL/D:  {nll:.2f}')
    fig.show()

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

mistral_api_hypers = dict(
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)


llma2_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

gemini_pro_hypers = dict(
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)



promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

model_hypers = {
     'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
     'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
     'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
     'PromptCast GPT-3': {'model': 'text-davinci-003', **promptcast_hypers},
     'LLMA2': {'model': 'llama-7b', **llma2_hypers},
     'mistral': {'model': 'mistral', **llma2_hypers},
     'mistral-api-tiny': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     'mistral-api-small': {'model': 'mistral-api-small', **mistral_api_hypers},
     'mistral-api-medium': {'model': 'mistral-api-medium', **mistral_api_hypers},
     'mistral-api-stocks-tiny': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     'mistral-api-stocks-small': {'model': 'mistral-api-small', **mistral_api_hypers},
     'mistral-api-stocks-medium': {'model': 'mistral-api-stocks-medium', **mistral_api_hypers},
     'gemini-pro': {'model': 'gemini-pro', **gemini_pro_hypers},
     'ARIMA': arima_hypers,
 }


#uncomment to use a model
model_predict_fns = {
    #'LLMA2': get_llmtime_predictions_data,
    #'mistral': get_llmtime_predictions_data,
    #'LLMTime GPT-4': get_llmtime_predictions_data,
    #'mistral-api-tiny': get_llmtime_predictions_data,
    #'mistral-api-stocks-medium': get_llmtime_predictions_data,
    'gemini-pro': get_llmtime_predictions_data
}


model_names = list(model_predict_fns.keys())


datasets = get_datasets()
ds_name = 'AirPassengersDataset'
data = datasets[ds_name]
train, test = data # or change to your own data




out = {}
start_time = perf_counter()
for model in model_names:
    model_hypers[model].update({'dataset_name': ds_name})
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    pred_dict = get_llmtime_predictions_data(train, test, model, model_hypers[model]['settings'],num_samples)
    out[model] = pred_dict
    plot_prds_ploty(ds_name,train, test, pred_dict, model, show_samples=True)
passed_time = perf_counter() - start_time
print(f"Execution time  {passed_time}")