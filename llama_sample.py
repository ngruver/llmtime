import os
import random
import torch
import glob
import itertools
import pandas as pd
import numpy as np
from jax import grad,vmap
from tqdm import tqdm
import argparse
from pathlib import Path
import transformers
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    LlamaConfig,
    LogitsProcessorList
)
from data.serialize import serialize_arr, deserialize_str, SerializerSettings

import pickle
import matplotlib.pyplot as plt
import pandas as pd
from data.small_context import get_datasets
from data.synthetic import get_synthetic_datasets
from scripts.real_benchmarks import get_benchmark_test_sets
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

def get_model_and_tokenizer(model_size, model_version=2, chat=False):
    assert model_version in [1, 2]
    assert model_size in ["7B", "13B", "30B", "70B"]
    
    if model_version == 1:
        base_path = "/checkpoint/mliuzzolino/llama_ckpts"
        base_path = os.path.join(base_path, model_size)

        print(f"Loading from {base_path}...")

        model = LlamaForCausalLM.from_pretrained(
            base_path,
            device_map="auto",   
            torch_dtype=torch.float16,
            # load_in_8bit=True
        )

        tokenizer = LlamaTokenizer.from_pretrained(
            os.path.join(base_path, "tokenizer.model"),
            use_fast=False,
        )

        model.eval()

        special_tokens_dict = dict()
        # if tokenizer.pad_token is None:
        #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        tokenizer.add_special_tokens(special_tokens_dict)
        # tokenizer.pad_token = tokenizer.eos_token

        assert len(tokenizer) == model.config.vocab_size

    elif model_version == 2:
        model = LlamaForCausalLM.from_pretrained(
            llama2_model_string(model_size, chat),
            device_map="auto",   
            torch_dtype=torch.float16,
        )

        tokenizer = LlamaTokenizer.from_pretrained(
            llama2_model_string(model_size, chat),
            use_fast=False,
        )

        model.eval()

        special_tokens_dict = dict()
        # if tokenizer.pad_token is None:
        #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        tokenizer.add_special_tokens(special_tokens_dict)
        # tokenizer.pad_token = tokenizer.eos_token

        assert len(tokenizer) == model.config.vocab_size

    return model, tokenizer

def nll_llama(input_arr, target_arr, model, tokenizer, settings:SerializerSettings, transform, count_seps=True, prompt=None, temp=1):
    """ Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM 
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
    Returns: NLL/D
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    if prompt:
        input_str = prompt + '\n' + input_str
    if not input_str.endswith(settings.time_sep):
        print('Appending time separator to input... Are you sure you want this?')
        prompt = input_str + settings.time_sep + target_str
    else:
        prompt = input_str + target_str
    
    batch = tokenizer(
        [prompt], 
        return_tensors="pt",
        add_special_tokens=True
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)


    good_tokens_str = list("0123456789,")
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    # if True:
    #     good_tokens += tokenizer("4 0 , 0")['input_ids']
    #     good_tokens = list(set(good_tokens))
    # all tokens that are not in good_tokens are bad tokens
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out['logits'][:,:,bad_tokens] = -100

    input_ids = batch['input_ids'][0][1:]
    logprobs = torch.nn.functional.log_softmax(out['logits'], dim=-1)[0][:-1]

    # good_tokens_str = list("0123456789,")
    # good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    # print(good_tokens_str)
    # print(good_tokens)
    # good_tokens_logprobs = logprobs[:,good_tokens].exp().sum(dim=-1)
    # print("logprobs:", good_tokens_logprobs[:100].mean())
    # print("logprobs:", good_tokens_logprobs[-100:].mean())
    # print(1/0)
    
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()

    tokens = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=False, 
        clean_up_tokenization_spaces=False
    )

    # print(tokens)
    # print(logprobs)

    input_len = len(tokenizer([input_str], return_tensors="pt",)['input_ids'][0])
    input_len = input_len - 2 # remove the BOS token

    # print(len(tokens), len(logprobs), input_len)

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum()/len(target_arr)

    #print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx

def sample_llama(
    model,
    tokenizer,
    prompt,
    max_tokens,
    batch_size=1,
    num_samples=20,
    temperature=0.9, 
    top_p=0.9,
):
    # print(prompt)

    gen_strs = []
    for _ in tqdm(range(num_samples // batch_size)):
        batch = tokenizer(
            [prompt], 
            return_tensors="pt",
        )

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}

        good_tokens_str = list("0123456789,")
        good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
        good_tokens += [tokenizer.eos_token_id]
        # if True:
        #     good_tokens += tokenizer("4 0 , 0")['input_ids']
        #     good_tokens = list(set(good_tokens))
            # good_tokens += [tokenizer.convert_tokens_to_ids(x) for x in ["_", "_,"]]
        # print(tokenizer.convert_ids_to_tokens(good_tokens))
        # print(1/0)
        
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature, 
            top_p=top_p, 
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )

        #print the strings corresponding to the input ids in batch
        # print(prompt)
        # print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
        # print(tokenizer.convert_ids_to_tokens(generate_ids[0]))
        # print(1/0)

        gen_strs += tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

    if len(gen_strs) < num_samples:
        print(f"Warning: only generated {len(gen_strs)} samples")
        #this won't be an issue as I'm planning to only us batch_size=1

    # print(gen_strs)

    return gen_strs

ASSISTANT_PROMPT = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence."

@dataclass
class Scaler:
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_avg_tokens_per_step(input_str,settings):
    input_tokens = sum([1 + len(x) for x in input_str.split(settings.time_sep)]) # add 1 for the comma, divide by 2 for the space
    input_steps = len(input_str.split(settings.time_sep))
    tokens_per_step = input_tokens / input_steps
    return tokens_per_step

def sample_completions(model, tokenizer, input_str, steps, settings, num_samples, temp, prompt, top_p=0.9):
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

    # print(input_str)

    # estimate avg number of tokens per step
    tokens_per_step = get_avg_tokens_per_step(input_str,settings)
    # print(tokens_per_step)
    # print(1/0)
    steps = int(steps * 1.3) # add some overhead to account for the fact that we don't know the exact number of tokens per step
    max_tokens=int(tokens_per_step*steps)

    if prompt:
        input_str = prompt + '\n' + input_str

    samples = sample_llama(
        model,
        tokenizer,
        input_str,
        max_tokens=max_tokens,
        batch_size=1,
        num_samples=num_samples,
        temperature=temp, 
        top_p=top_p,
    )

    samples = [x.replace(input_str, '').strip() for x in samples]

    return samples

def handle_prediction(input, pred, expected_length):
    ''' Handle prediction with expected length of expected_length. 
        Useful for handling predictions that can't be deserialized or are too short or too long. 
    '''
    # print(pred)

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
    tokenizer,
    inputs, 
    steps, 
    settings: SerializerSettings, 
    scalers: None,
    num_samples=1, 
    temp=0.3,
    top_p=0.9,
    prompt=None, 
    post_prompt=None,
    save_path=None, 
    return_input_strs=False,
    **kwargs
):
    ''' Generate predictions from GPT-3 for a batch of inputs by calling sample_completions
    Args:
        inputs: np float array of shape (batch_size, history_len)
        steps: number of steps to predict
        precision: number of bits to use for encoding
        num_samples: number of samples to return
        temp: temperature for sampling
        prompt: additional prompt before the input string
        post_prompt: additional prompt after the input string (e.g. for promptcast)
        model: name of GPT-3 model to use
    Returns:
        np float array of shape (batch_size, num_samples, steps)
    '''
    if scalers is None:
        scalers = [Scaler() for _ in inputs]
    else:
        assert len(scalers) == len(inputs), 'Number of scalers must match number of inputs'

    transformed_inputs = np.array([scaler.transform(input_array) for input_array, scaler in zip(inputs, scalers)])

    input_strs = [serialize_arr(scaled_input_array, settings) for scaled_input_array in transformed_inputs]
    
    if not isinstance(prompt, list):
        prompts = [prompt] * len(inputs)
        post_prompts = [post_prompt] * len(inputs)
    else:
        prompts = prompt
        post_prompts = post_prompt

    assert len(prompts) == len(inputs), 'Number of prompts must match number of inputs'

    # if post_prompts[0] is not None:
    #     assert len(post_prompts) == len(inputs), 'Number of post prompts must match number of inputs'

    # for input_str, prompt, post_prompt in zip(input_strs, prompts, post_prompts):

    #     input_strs = [prompt + input_str.rstrip(settings.time_sep) + post_prompt for input_str, prompt, post_prompt in zip(input_strs, prompts, post_prompts)]
    #     prompt = None # we already added the prompt to the input string

    # completions_list = []
    # complete = lambda x, prompt_: sample_completions(model, tokenizer, x, steps, settings, num_samples, temp, prompt_, top_p)
    # completions_list = [complete(input_str, prompt) for input_str, prompt in tqdm(zip(input_strs, prompts), total=len(inputs))]

    complete = lambda x, prompt: sample_completions(model, tokenizer, x, steps, settings, num_samples, temp, prompt, top_p)
    completions_list = [complete(input_str, prompt) for input_str, prompt in tqdm(zip(input_strs, prompt))]

    completion_to_pred = lambda completion, transformed_input, inv_transform: inv_transform(
        handle_prediction(
            transformed_input, deserialize_str(
                completion, settings, ignore_last=False, steps=steps
            ), 
            expected_length=steps
        )
    )
    preds = np.array([
        [
            completion_to_pred(
                completion, transformed_input, scaler.inv_transform
            ) for completion in completions
        ] 
            for completions, transformed_input, scaler in zip(completions_list, transformed_inputs, scalers)
    ])
    
    if save_path:
        np.save(save_path, preds)
        print('Saved predictions to', save_path)
    if return_input_strs:
        return preds, completions_list, input_strs
  
    return preds, completions_list

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

def get_llama_predictions_data(
    train, 
    test, 
    model_version,
    model_size, 
    model_is_chat,
    settings, 
    model=None,
    tokenizer=None,
    num_samples=10, 
    temp=0.8,
    top_p=0.9,
    alpha=0.6, 
    beta=0.3,
    constrain_tokens=True, 
    basic=False, 
    description=None, 
    no_scaler=False,
    **kwargs
):
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    num_samples = max(0, num_samples)
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

    # Create a unique scaler for each series
    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]
    if no_scaler:
        scalers = [Scaler() for i in range(len(train))]

    def truncate(train, test, scaler):
        tokens_perstep = get_avg_tokens_per_step(
            serialize_arr(
                scaler.transform(pd.concat([train,test]).values), 
                settings
            ),
            settings
        )
        if model_version == 2:
            max_tokens = 4000
        else:
            max_tokens = 2000

        # 1.3 accounts for overhead in sampling
        if 1.3*tokens_perstep*(len(train)+len(test)) > max_tokens:
            total_timestep_budget = int(max_tokens/tokens_perstep)
            full_train_len = len(train)
            for num_try in range(10):
                sub_train = train.iloc[-(total_timestep_budget-len(test)):]
                if 1.3*tokens_perstep*(len(sub_train)+len(test)) <= max_tokens:
                    train = sub_train
                    print(f"Truncated train to {full_train_len} --> {len(train)} timesteps")
                    break 
                total_timestep_budget = int(0.8 * total_timestep_budget)
            else:
                raise ValueError(f"After truncation, dataset is still too large for GPT-3, 1.3 * {tokens_perstep} * ({len(sub_train)} + {len(test)}) = {1.3*tokens_perstep*(len(sub_train)+len(test))} > {max_tokens}")
        return train

    for i in range(len(train)):
        train[i] = truncate(train[i], test[i], Scaler())#scalers[i])
        
    prompts = [None] * len(train)
    if description is not None:
        description = description.replace(settings.time_sep.replace(" ",""), "") # for likelihood calculation
        prompt = "Dataset description:\n" + description + "\nSequence:"
        prompt = ASSISTANT_PROMPT + "\n" + prompt
        prompts = [prompt] * len(train)

    # Create inputs for GPT model
    inputs = [train[i].values for i in range(len(train))]
    steps = test_len

    model_name = "llama" + str(model_version) + "-" + model_size + ("-chat" if model_is_chat else "")
    if model is None and tokenizer is None:
        # print(model_version, model_size, model_is_chat)
        model, tokenizer = get_model_and_tokenizer(model_size, model_version, model_is_chat)

    samples = None
    medians = None
    if num_samples > 0:
        # Generate predictions
        preds, _, _ = generate_predictions(
            model, tokenizer, inputs, 
            steps, settings, scalers,
            num_samples=num_samples, temp=temp, top_p=top_p, 
            prompt=prompts, 
            return_input_strs=True, **kwargs
        )
        # preds: (num_series, num_samples, num_steps)
        #medians = np.median(preds, axis=1)  # (num_series, num_steps)
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model_name,
        }
    }

    BPDs = [
        nll_llama(
            train[i].values, 
            test[i].values, 
            model, 
            tokenizer,
            settings, 
            scalers[i].transform, 
            count_seps=True, 
            prompt=prompts[i], 
            temp=temp
        ) for i in range(len(train))
    ]
    out_dict['NLL/D'] = np.mean(BPDs)

    return out_dict

def get_promptcast_predictions_data(
    train, 
    test, 
    model_version,
    model_size, 
    model_is_chat,
    settings, 
    num_samples=10, 
    temp=0.8,
    top_p=0.9,
    alpha=0.6, 
    beta=0.3,
    dataset_name='dataset', 
    **kwargs
):
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    num_samples = max(0, num_samples)
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

    def truncate(train, test, scaler):
        tokens_perstep = get_avg_tokens_per_step(
            serialize_arr(
                scaler.transform(pd.concat([train,test]).values), 
                settings
            ),
            settings
        )
        if model_version == 2:
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

    for i in range(len(train)):
        train[i] = truncate(train[i], test[i], scalers[i])

    prompt = f'The values in the {dataset_name} for the past {len(train[0])} time steps are '
    prompts = [prompt] * len(train)
    post_prompt = f'. What will the values for the next {len(test[0])} time steps will be? The values for the next {len(test[0])} time steps will be'
    post_prompts = [post_prompt] * len(train)

    # Create inputs for GPT model
    inputs = [train[i].values for i in range(len(train))]
    steps = test_len

    # print(model_version, model_size, model_is_chat)
    model_name = "llama" + str(model_version) + "-" + model_size + ("-chat" if model_is_chat else "")
    model, tokenizer = get_model_and_tokenizer(model_size, model_version, model_is_chat)

    samples = None
    medians = None
    if num_samples > 0:
        # Generate predictions
        preds, _, _ = generate_predictions(
            model, tokenizer, inputs, 
            steps, settings, scalers,
            num_samples=num_samples, temp=temp, top_p=top_p, 
            prompt=prompts, 
            post_prompt=post_prompts,
            return_input_strs=True, **kwargs
        )
        # preds: (num_series, num_samples, num_steps)
        #medians = np.median(preds, axis=1)  # (num_series, num_steps)
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model_name,
        },
    }

    out_dict['NLL/D'] = None

    return out_dict

def run_darts(
    args,
):    
    dsnames = [
        'AirPassengersDataset', 
        'AusBeerDataset', 
        'GasRateCO2Dataset', 
        'MonthlyMilkDataset', 
        'SunspotsDataset', 
        'WineDataset', 
        'WoolyDataset', 
        'HeartRateDataset'
    ]
    dsnames = ["darts-" + dsname for dsname in dsnames]

    i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    dsname = dsnames[int(i)]

    print(f"Running on {dsname}...")

    hypers = {
        "base": 10,
        "prec": args.prec,
        "time_sep": args.time_sep,
        "bit_sep": args.bit_sep,
    }

    promptcast_hypers = dict(
        base=10,
        prec=0, 
        signed=True, 
        time_sep=',',
        bit_sep='',
        plus_sign='',
        minus_sign='-',
        half_bin_correction=False,
        decimal_point=''
    )
    hypers = promptcast_hypers

    datasets = get_datasets()
    data = datasets[dsname.replace("darts-", "")]
    train, test = data

    out = get_promptcast_predictions_data(#get_llama_predictions_data(
        train, test, 
        args.model_version, 
        args.model_size, 
        args.model_is_chat, 
        hypers, 
        num_samples=args.num_samples,
        temp=args.temp,
        top_p=args.top_p,
        alpha=args.alpha,
        beta=args.beta,
        dataset_name=dsname.replace("darts-", ""),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, dsname)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(
        output_dir, 
        f"{args.temp}_{args.top_p}_{args.alpha}_{args.beta}_{args.prec}_{args.time_sep}_{args.bit_sep}.pkl"
    )
    with open(out_path,'wb') as f:
        pickle.dump(out,f)


def run_darts_subsample(
    args,   
):
    dsnames = [
        'AirPassengersDataset', 
        'AusBeerDataset', 
        'GasRateCO2Dataset', 
        'MonthlyMilkDataset', 
        'SunspotsDataset', 
        'WineDataset', 
        'WoolyDataset', 
        'HeartRateDataset'
    ]
    dsnames = ["darts-" + dsname for dsname in dsnames]
    train_fracs = [0.1, 0.2, 0.5, 1]

    ds_tuples = list(itertools.product(dsnames, train_fracs))

    model, tokenizer = get_model_and_tokenizer(args.model_size, args.model_version, args.model_is_chat)

    # i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    # ds_tuple = ds_tuples[int(i)]

    # betas = [.5]
    # alphas = [0.5, 0.7, 0.9, 0.99] #[0.99]
    # precs = [3]

    # combos = list(itertools.product(betas, alphas, precs))

    # for beta, alpha, prec in combos:

    beta = 0 # args.beta
    alpha = -1 # args.alpha
    prec = 0 # args.prec

    for ds_tuple in ds_tuples:
        print(ds_tuple)

        dsname, train_frac = ds_tuple

        print(f"Running on {dsname}...")

        hypers = {
            "base": 10,
            "prec": prec,
            "time_sep": args.time_sep,
            "bit_sep": args.bit_sep,
            "signed": True,
        }

        datasets = get_datasets()
        data = datasets[dsname.replace("darts-", "")]

        train, test = data
        sub_train = train[-int(len(train) * train_frac):]

        print(sub_train)

        out = get_llama_predictions_data(
            sub_train, test, 
            args.model_version, 
            args.model_size, 
            args.model_is_chat, 
            hypers,
            model=model,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            temp=args.temp,
            top_p=args.top_p,
            alpha=alpha,
            beta=beta,
            no_scaler=True,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = os.path.join(args.output_dir, dsname)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, f"p={train_frac}")
        os.makedirs(output_dir, exist_ok=True)

        out_path = os.path.join(
            output_dir, 
            f"{args.temp}_{args.top_p}_{alpha}_{beta}_{prec}_{args.time_sep}_{args.bit_sep}.pkl"
        )
        with open(out_path,'wb') as f:
            pickle.dump(out,f)

        print(out['NLL/D'])

def nan_corruption(x:pd.Series, p=0.0):
    x = x.copy()
    x.iloc[np.random.choice(len(x), int(p*len(x)),replace=False)] = np.nan
    # replace p % of the elements with nans
    return x

def interp_nans(x:pd.Series):
    x = x.copy()
    nans = np.isnan(x)
    f = lambda z: z.values.nonzero()[0]
    x[nans] = np.interp(f(nans), f(~nans), x[~nans])
    return x

def run_darts_missing(
    args,
):    
    dsnames = [
        'AirPassengersDataset', 
        'AusBeerDataset', 
        'GasRateCO2Dataset', 
        'MonthlyMilkDataset', 
        'SunspotsDataset', 
        'WineDataset', 
        'WoolyDataset', 
        'HeartRateDataset'
    ]
    dsnames = ["darts-" + dsname for dsname in dsnames]
    ps = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    num_seeds = 3

    i = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ds_idx = i // (len(ps)*num_seeds*2)
    p_idx = (i // (num_seeds*2)) % len(ps)
    seed = (i // 2) % num_seeds
    interpolated = i % 2

    dsname = dsnames[ds_idx]
    p = ps[p_idx]

    print(f"Running on {dsname}...")

    hypers = {
        "base": 10,
        "prec": args.prec,
        "time_sep": args.time_sep,
        "bit_sep": args.bit_sep,
        "missing_str": "NaN",
    }

    datasets = get_datasets()
    data = datasets[dsname.replace("darts-", "")]
    train, test = data

    np.random.seed(seed)
    corrupted_train = nan_corruption(train,p=p)
    if interpolated:
        train = interp_nans(corrupted_train)
    else:
        train = corrupted_train

    out = get_llama_predictions_data(
        train, test, 
        args.model_version, 
        args.model_size, 
        args.model_is_chat, 
        hypers, 
        num_samples=args.num_samples,
        temp=args.temp,
        top_p=args.top_p,
        alpha=args.alpha,
        beta=args.beta,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, dsname)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "interpolated" if interpolated else "corrupted")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f"p={p}")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f"seed={seed}")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(
        output_dir, 
        f"{args.temp}_{args.top_p}_{args.alpha}_{args.beta}_{args.prec}_{args.time_sep}_{args.bit_sep}.pkl"
    )
    with open(out_path,'wb') as f:
        pickle.dump(out,f)


def run_monash(
    args,
):    
    benchmarks = get_benchmark_test_sets()

    df = pd.read_csv(
        '/private/home/ngruver/time-series-lm/monash_details.csv'
    )
    planned = []
    for dsname, num_series in df[['dsname', 'num_series']].values:
        for idx in range(num_series):
            planned.append((dsname, idx))

    completed = [
        os.path.basename(x) for x in glob.glob(
            "/private/home/ngruver/time-series-lm/llama_2_monash/llama2_70B/*"
        ) if len(glob.glob(os.path.join(x, "*.pkl"))) > 0
    ]
    completed = [
        ('_'.join(x.split("_")[:-1]), int(x.split("_")[-1])) for x in completed
    ]

    to_do = sorted(list(set(planned) - set(completed)))

    # i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    # ds_tuple = to_do[int(i)]

    np.random.shuffle(to_do)
    ds_tuple = to_do[0]
    dsname, series_num = ds_tuple

    print(ds_tuple)
    print(len(benchmarks[dsname][0]))
    train, test = benchmarks[dsname][0][series_num]

    print(f"Running on {dsname}...")

    hypers = {
        "base": 10,
        "prec": args.prec,
        "time_sep": args.time_sep,
        "bit_sep": args.bit_sep,
        "signed": True,
    }

    # print(len(train), len(test))
    # print(train[0].shape, test[0].shape)
    # print(1/0)

    out = get_llama_predictions_data(
        train, test, 
        args.model_version, 
        args.model_size, 
        args.model_is_chat, 
        hypers, 
        num_samples=args.num_samples,
        temp=args.temp,
        top_p=args.top_p,
        alpha=args.alpha,
        beta=args.beta,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, f"{dsname}_{series_num}")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(
        output_dir, 
        f"{args.temp}_{args.top_p}_{args.alpha}_{args.beta}_{args.prec}_{args.time_sep}_{args.bit_sep}.pkl"
    )
    with open(out_path,'wb') as f:
        pickle.dump(out,f)

def run_autoformer(
    args,
):    
    test_length = 96

    df = pd.read_csv(
        '/private/home/ngruver/time-series-lm/autoformer/autoformer_details.csv'
    )
    planned = []
    for dsname, num_series in df[['dsname', 'num_series']].values:
        for idx in range(1, num_series+1):
            planned.append((dsname, idx))

    completed_path = f"/private/home/ngruver/time-series-lm/llama_2_autoformer/llama2_70B/*/{test_length}"
    completed = [
        x.replace(f"/{test_length}","") for x in glob.glob(completed_path) \
            if len(glob.glob(os.path.join(x, "*.pkl"))) > 0
    ]
    completed = [os.path.basename(x) for x in completed]
    completed = [
        ('_'.join(x.split("_")[:-1]), int(x.split("_")[-1])) for x in completed
    ]

    to_do = sorted(list(set(planned) - set(completed)))

    np.random.shuffle(to_do)
    ds_tuple = to_do[0]
    dsname, series_num = ds_tuple

    if dsname == "national_illness.csv":
        test_length = 36

    df = pd.read_csv(
        f"/private/home/ngruver/time-series-lm/autoformer/{dsname}.csv"
    )

    train = df.iloc[:-test_length,series_num]
    test = df.iloc[-test_length:,series_num]

    print(ds_tuple)
    print(f"Running on {dsname}...")

    hypers = {
        "base": 10,
        "prec": args.prec,
        "time_sep": args.time_sep,
        "bit_sep": args.bit_sep,
        "signed": True,
    }

    # print(len(train), len(test))
    # print(train[0].shape, test[0].shape)
    # print(1/0)

    out = get_llama_predictions_data(
        train, test, 
        args.model_version, 
        args.model_size, 
        args.model_is_chat, 
        hypers, 
        num_samples=args.num_samples,
        temp=args.temp,
        top_p=args.top_p,
        alpha=args.alpha,
        beta=args.beta,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, f"{dsname}_{series_num}")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f"{test_length}")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(
        output_dir, 
        f"{args.temp}_{args.top_p}_{args.alpha}_{args.beta}_{args.prec}_{args.time_sep}_{args.bit_sep}.pkl"
    )
    with open(out_path,'wb') as f:
        pickle.dump(out,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--model_version", type=int, required=True)
    parser.add_argument("--model_is_chat", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--prec", type=int, default=3)
    parser.add_argument("--time_sep", type=str, default=" ,")
    parser.add_argument("--bit_sep", type=str, default=" ")
    args = parser.parse_args()

    # run_darts(args)
    run_darts_subsample(args)
    # run_darts_missing(args)
    # run_monash(args)
    # run_autoformer(args)