from data.serialize import serialize_arr, SerializerSettings
import google.generativeai as genai
import tiktoken
import torch
import os
import numpy as np
from jax import grad,vmap

loaded_model=''
gemini_client={}

def init_google_gemini_pro_client(model):
    """
    Initialize the Gemini client for a specific LLM model.
    """
    global loaded_model, gemini_client
    if gemini_client == {} or loaded_model != model:
        loaded_model = model
        genai.configure(api_key=os.environ['GEMINI_PRO_KEY'])
        gemini_client = genai.GenerativeModel(model) 
    return gemini_client

def google_gemini_pro_tokenize_fn(str, model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return encoding.encode(str)

def get_allowed_ids(strs, model):
    """
    Retrieve the token IDs for a given list of strings for a specific GPT model.

    Args:
        strs (list of str): strs to be converted.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    ids = []
    for s in strs:
        id = encoding.encode(s) #init_Gemini_client(model).embeddings(model="Gemini-embed",input=s)
        ids.extend(id)
    return ids

def google_gemini_pro_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Generate text completions from Gemini using Google's API.

    Args:
        model (str): Name of the Gemini model to use.
        input_str (str): Serialized input time series data.
        steps (int): Number of time steps to predict.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate.
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    avg_tokens_per_step = len(google_gemini_pro_tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))
    # define logit bias to prevent Gemini from producing unwanted tokens
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    if model in ['gemini-pro']:
        sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        messages = [{'role':'user', 'parts': [sys_message,extra_input+input_str+settings.time_sep]}]
        response = init_google_gemini_pro_client(model).generate_content(
            contents=messages,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=['x'],
                #max_output_tokens=int(avg_tokens_per_step*steps),
                temperature=temp)
        )
    return [response.text]

    
def google_gemini_pro_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension of the target array according to the LLM.

    Args:
        model (str): Name of the LLM model to use.
        input_arr (array-like): Input array (history).
        target_arr (array-like): Ground target array (future).
        settings (SerializerSettings): Serialization settings.
        transform (callable): Transformation applied to the numerical values before serialization.
        count_seps (bool, optional): Whether to account for separators in the calculation. Should be true for models that generate a variable number of digits. Defaults to True.
        temp (float, optional): Temperature for sampling. Defaults to 1.

    Returns:
        float: Calculated NLL per dimension.
    """
    return -1
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), f'Input string must end with {settings.time_sep}, got {input_str}'
    full_series = input_str + target_str
    messages = [{'role':'user', 'parts': [full_series]}]
    response = init_google_gemini_pro_client(model).generate_content(
            contents=messages,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=['x'],
                temperature=temp)
    )
    tokens = np.array(response.text)
    softs = torch.nn.Softmax(response.text)
    print('softs')
    print(softs)
    logprobs = [np.log(soft) for soft in softs] 
    top5logprobs = torch.topk(logprobs,5)
    seps = tokens==settings.time_sep
    target_start = np.argmax(np.cumsum(seps)==len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens==settings.time_sep
    assert len(logprobs[seps]) == len(target_arr), f'There should be one separator per target. Got {len(logprobs[seps])} separators and {len(target_arr)} targets.'
    # adjust logprobs by removing extraneous and renormalizing (see appendix of paper)
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign, settings.bit_sep+settings.decimal_point]
    allowed_tokens = {t for t in allowed_tokens if len(t) > 0}
    p_extra = np.array([sum(np.exp(ll) for k,ll in top5logprobs[i].items() if not (k in allowed_tokens)) for i in range(len(top5logprobs))])
    if settings.bit_sep == '':
        p_extra = 0
    adjusted_logprobs = logprobs - np.log(1-p_extra)
    digits_bits = -adjusted_logprobs[~seps].sum()
    seps_bits = -adjusted_logprobs[seps].sum()
    BPD = digits_bits/len(target_arr)
    if count_seps:
        BPD += seps_bits/len(target_arr)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx