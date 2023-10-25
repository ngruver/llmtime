from tqdm import tqdm
from data.serialize import serialize_arr, deserialize_str, SerializerSettings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass
from models.llms import completion_fns, nll_fns, tokenization_fns, context_lengths

STEP_MULTIPLIER = 1.2

@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.
    
    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    if model in tokenization_fns and model in context_lengths:
        tokenization_fn = tokenization_fns[model]
        context_length = context_lengths[model]
        input_str_chuncks = input_str.split(settings.time_sep)
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
            # add separator if not already present
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep
            input_tokens = tokenization_fn(truncated_input_str)
            num_input_tokens = len(input_tokens)
            avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
            if num_input_tokens + num_output_tokens <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str

def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]

def generate_predictions(
    completion_fn, 
    input_strs, 
    steps, 
    settings: SerializerSettings, 
    scalers: None,
    num_samples=1, 
    temp=0.7, 
    parallel=True,
    strict_handling=False,
    max_concurrent=10,
    **kwargs
):
    """
    Generate and process text completions from a language model for input time series.

    Args:
        completion_fn (callable): Function to obtain text completions from the LLM.
        input_strs (list of array-like): List of input time series.
        steps (int): Number of steps to predict.
        settings (SerializerSettings): Settings for serialization.
        scalers (list of Scaler, optional): List of Scaler objects. Defaults to None, meaning no scaling is applied.
        num_samples (int, optional): Number of samples to return. Defaults to 1.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        parallel (bool, optional): If True, run completions in parallel. Defaults to True.
        strict_handling (bool, optional): If True, return None for predictions that don't have exactly the right format or expected length. Defaults to False.
        max_concurrent (int, optional): Maximum number of concurrent completions. Defaults to 50.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: Tuple containing:
            - preds (list of lists): Numerical predictions.
            - completions_list (list of lists): Raw text completions.
            - input_strs (list of str): Serialized input strings.
    """
    
    completions_list = []
    complete = lambda x: completion_fn(input_str=x, steps=steps*STEP_MULTIPLIER, settings=settings, num_samples=num_samples, temp=temp)
    if parallel and len(input_strs) > 1:
        print('Running completions in parallel for each input')
        with ThreadPoolExecutor(min(max_concurrent, len(input_strs))) as p:
            completions_list = list(tqdm(p.map(complete, input_strs), total=len(input_strs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]
    def completion_to_pred(completion, inv_transform): 
        pred = handle_prediction(deserialize_str(completion, settings, ignore_last=False, steps=steps), expected_length=steps, strict=strict_handling)
        if pred is not None:
            return inv_transform(pred)
        else:
            return None
    preds = [[completion_to_pred(completion, scaler.inv_transform) for completion in completions] for completions, scaler in zip(completions_list, scalers)]
    return preds, completions_list, input_strs

def get_llmtime_predictions_data(train, test, model, settings, num_samples=10, temp=0.7, alpha=0.95, beta=0.3, basic=False, parallel=True, **kwargs):
    """
    Obtain forecasts from an LLM based on training series (history) and evaluate likelihood on test series (true future).
    train and test can be either a single time series or a list of time series.

    Args:
        train (array-like or list of array-like): Training time series data (history).
        test (array-like or list of array-like): Test time series data (true future).
        model (str): Name of the LLM model to use. Must have a corresponding entry in completion_fns.
        settings (SerializerSettings or dict): Serialization settings.
        num_samples (int, optional): Number of samples to return. Defaults to 10.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        alpha (float, optional): Scaling parameter. Defaults to 0.95.
        beta (float, optional): Shift parameter. Defaults to 0.3.
        basic (bool, optional): If True, use the basic version of data scaling. Defaults to False.
        parallel (bool, optional): If True, run predictions in parallel. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Dictionary containing predictions, samples, median, NLL/D averaged over each series, and other related information.
    """

    assert model in completion_fns, f'Invalid model {model}, must be one of {list(completion_fns.keys())}'
    completion_fn = completion_fns[model]
    nll_fn = nll_fns[model] if model in nll_fns else None
    
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

    # Create a unique scaler for each series
    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]

    # transform input_arrs
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    # serialize input_arrs
    input_strs = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in transformed_input_arrs]
    # Truncate input_arrs to fit the maximum context length
    input_arrs, input_strs = zip(*[truncate_input(input_array, input_str, settings, model, test_len) for input_array, input_str in zip(input_arrs, input_strs)])
    
    steps = test_len
    samples = None
    medians = None
    completions_list = None
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(completion_fn, input_strs, steps, settings, scalers,
                                                                    num_samples=num_samples, temp=temp, 
                                                                    parallel=parallel, **kwargs)
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
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
    # Compute NLL/D on the true test series conditioned on the (truncated) input series
    if nll_fn is not None:
        BPDs = [nll_fn(input_arr=input_arrs[i], target_arr=test[i].values, settings=settings, transform=scalers[i].transform, count_seps=True, temp=temp) for i in range(len(train))]
        out_dict['NLL/D'] = np.mean(BPDs)
    return out_dict
