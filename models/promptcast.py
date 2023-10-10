from data.metrics import Evaluator
from tqdm import tqdm
from multiprocess import Pool
from functools import partial
import tiktoken
from functools import partial
from data.serialize import serialize_arr, deserialize_str, SerializerSettings
import openai
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from data.metrics import nll
import pandas as pd
from dataclasses import dataclass

@dataclass
class Scaler:
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=.9, beta=.3,basic=False):
    # shift (min - beta*(max-min)) to 0
    # then scale alpha quantile to 1
    # alpha = -1 means no scaling
    history = history[~np.isnan(history)]
    min_ = np.min(history) - beta*(np.max(history)-np.min(history))
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        # scale so that alpha fraction of values are within [0, 1]
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
        return Scaler(transform=transform, inv_transform=inv_transform)
    if alpha == -1:
        q = 1
    else:
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
    # scale so that alpha fraction of values are within [0, 1]
    def transform(x):
        return (x - min_) / q
    def inv_transform(x):
        return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def get_token_ids(tokens, model, input_string):
    encoding = tiktoken.encoding_for_model(model)
    ids = []
    for t in tokens:
        id = encoding.encode(t)
        if len(id) != 1:
            for i in id:
                ids.append(i)
            #raise ValueError(f'{t} is not a single token')
        else:
            ids.append(id[0])
    return ids

def get_avg_tokens_per_step(input_str,settings):
    input_tokens = sum([1 + len(x) / 2 for x in input_str.split(settings.time_sep)]) # add 1 for the comma, divide by 2 for the space
    input_steps = len(input_str.split(settings.time_sep))
    tokens_per_step = input_tokens / input_steps
    return tokens_per_step

def truncate(train, test, scaler, model, settings):
    tokens_perstep = get_avg_tokens_per_step(
        serialize_arr(
            scaler.transform(pd.concat([train,test]).values), 
            settings
        ),
        settings
    )
    if model == 'gpt-4':
        max_tokens=6000
    elif model == 'gpt-3.5-turbo':
        max_tokens = 4000
    else:
        max_tokens = 2000

    # 1.3 accounts for overhead in sampling
    if 1.35*tokens_perstep*(len(train)+len(test)) > max_tokens:
        total_timestep_budget = int(max_tokens/tokens_perstep)
        full_train_len = len(train)
        for num_try in range(10):
            sub_train = train.iloc[-(total_timestep_budget-len(test)):]
            if 1.35*tokens_perstep*(len(sub_train)+len(test)) <= max_tokens:
                train = sub_train
                print(f"Truncated train to {full_train_len} --> {len(train)} timesteps")
                break 
            total_timestep_budget = int(0.8 * total_timestep_budget)
        else:
            raise ValueError(f"After truncation, dataset is still too large for GPT-3, 1.3 * {tokens_perstep} * ({len(sub_train)} + {len(test)}) = {1.3*tokens_perstep*(len(sub_train)+len(test))} > {max_tokens}")
    return train

def sample_completions(model, input_str, steps, settings, num_samples, temp, logit_bias,**kwargs):
    ''' Sample completions from GPT-3
    Args:
        input_str: input sequence as a string
        steps: number of steps to predict
        precision: number of bits to use for encoding
        num_samples: number of samples to return
        temp: temperature for sampling
        prompt: additional prompt before the input string
        model: name of GPT-3 model to use
    Returns:
        list of completion strings
    '''
    # estimate avg number of tokens per step
    tokens_per_step = get_avg_tokens_per_step(input_str,settings)
    steps = int(steps * 1.3) # add some overhead to account for the fact that we don't know the exact number of tokens per step
    # if not logit_bias:
    #     input_str = input_str[:-1]
    if model in ['gpt-3.5-turbo','gpt-4']:
        chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {"role": "system", "content": chatgpt_sys_message},
                    {"role": "user", "content": extra_input+input_str+settings.time_sep}
                ],
            max_tokens=int(tokens_per_step*steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples,
            **kwargs
        )
        return [choice.message.content for choice in response.choices]
    else:
        response = openai.Completion.create(
            model=model,
            prompt=input_str, 
            max_tokens=int(tokens_per_step*steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.text for choice in response.choices]

def handle_prediction(input, pred, expected_length, strict=False):
    ''' Handle prediction with expected length of expected_length. 
        Useful for handling predictions that can't be deserialized or are too short or too long. 
    '''
    if strict:
        # must be a valid array (not None) and have the enough entries
        if pred is None or len(pred) < expected_length:
            print('Found invalid prediction')
            return None
        else:
            return pred[:expected_length]
    else:
        if pred is None:
            print('Warning: prediction failed to be deserialized, replaced with last value')
            return np.full(expected_length, input[-1])
        elif len(pred) < expected_length:
            print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
            return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        elif len(pred) > expected_length:
            return pred[:expected_length]
        else:
            return pred

def generate_predictions(
    model, 
    inputs, 
    steps, 
    settings: SerializerSettings, 
    scalers: None,
    num_samples=1, 
    temp=0.3, 
    prompts=None,
    post_prompts=None,
    parallel=True,
    return_input_strs=False,
    constrain_tokens=True,
    strict_handling=False,
    **kwargs,
):
    ''' Generate predictions from GPT-3 for a batch of inputs by calling sample_completions
    Args:
        inputs: np float array of shape (batch_size, history_len)
        steps: number of steps to predict
        precision: number of bits to use for encoding
        num_samples: number of samples to return
        temp: temperature for sampling
        prompt: None or a batch of additional prompts before the input string
        post_prompt: None or a batch of additional prompts after the input string (e.g. for promptcast)
        model: name of GPT-3 model to use
    Returns:
        np float array of shape (batch_size, num_samples, steps)
    '''
    if prompts is None:
        prompts = [''] * len(inputs)
    if post_prompts is None:
        post_prompts = [''] * len(inputs)
    assert len(prompts) == len(inputs), f'Number of prompts must match number of inputs, got {len(prompts)} prompts and {len(inputs)} inputs'
    assert len(post_prompts) == len(inputs), f'Number of post prompts must match number of inputs, got {len(post_prompts)} post prompts and {len(inputs)} inputs'
    
    if scalers is None:
        scalers = [Scaler() for _ in inputs]
    else:
        assert len(scalers) == len(inputs), 'Number of scalers must match number of inputs'

    transformed_inputs = np.array([scaler.transform(input_array) for input_array, scaler in zip(inputs, scalers)])
    input_strs = [serialize_arr(scaled_input_array, settings) for scaled_input_array in transformed_inputs]
    if post_prompts[0] != '':
        # removing last time separator for promptcast
        input_strs = [prompt + input_str.rstrip(settings.time_sep) + post_prompt for input_str, prompt, post_prompt in zip(input_strs, prompts, post_prompts)]
    else:
        input_strs = [prompt + input_str for input_str, prompt in zip(input_strs, prompts)]
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    logit_bias = {}
    if (model not in ['gpt-3.5-turbo','gpt-4']) and constrain_tokens: # logit bias not supported for chat models
        logit_bias = {id: 30 for id in get_token_ids(allowed_tokens, model,input_strs[0])}
    if not constrain_tokens:
        logit_bias = {id: 5 for id in get_token_ids(allowed_tokens, model,input_strs[0])}

    completions_list = []
    complete = lambda x: sample_completions(model, x, steps, settings, num_samples, temp, logit_bias,**kwargs)
    if parallel and len(inputs) > 1:
        with ThreadPoolExecutor(len(inputs)) as p:
            completions_list = list(tqdm(p.map(complete, input_strs), total=len(inputs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]
    # print(completions_list)
    def completion_to_pred(completion, transformed_input, inv_transform): 
        pred = handle_prediction(transformed_input, deserialize_str(completion, settings, ignore_last=False, steps=steps), expected_length=steps, strict=strict_handling)
        if pred is not None:
            return inv_transform(pred)
        else:
            return None
    preds = [[completion_to_pred(completion, transformed_input, scaler.inv_transform) for completion in completions] for completions, transformed_input, scaler in zip(completions_list, transformed_inputs, scalers)]
    if return_input_strs:
        return preds, completions_list, input_strs
    return preds, completions_list

def get_promptcast_predictions_data(train, test, model, settings, num_samples=10, temp=0.8, dataset_name='dataset', **kwargs):
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, list):
        # Assume single train/test case
        train = [train]
        test = [test]

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i])+len(train[i])))

    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

    # Identity scalers
    scalers = [Scaler() for _ in range(len(train))]
    
    for i in range(len(train)):
        train[i] = truncate(train[i], test[i], scalers[i], model, settings)
    
    prompt = f'The values in the {dataset_name} for the past {len(train[0])} time steps are '
    prompts = [prompt] * len(train)
    post_prompt = f'. What will the values for the next {len(test[0])} time steps will be? The values for the next {len(test[0])} time steps will be '
    post_prompts = [post_prompt] * len(train)

    # Create inputs for GPT model
    inputs = [train[i].values for i in range(len(train))]
    steps = test_len

    samples = None
    medians = None
    completions_list = None
    input_strs = None
    if num_samples > 0:
        # Generate predictions
        preds, completions_list, input_strs = generate_predictions(model, inputs, steps, settings, scalers,
                                                                    num_samples=num_samples, temp=temp, prompts=prompts, post_prompts=post_prompts,
                                                                    parallel=True, return_input_strs=True, constrain_tokens=False, strict_handling=True, **kwargs)
        # skip bad samples
        samples = [pd.DataFrame(np.array([p for p in preds[i] if p is not None]), columns=test[i].index) for i in range(len(preds))] 
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        print('Got %d properly formatted samples' % len(samples))
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model,
        },
        'completions_list': completions_list,
        'input_strs': input_strs,
    }

    out_dict['NLL/D'] = None

    return out_dict