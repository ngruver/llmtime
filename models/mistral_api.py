from data.serialize import serialize_arr, SerializerSettings
from mistralai import Mistral, UserMessage
import tiktoken
import os
import numpy as np
from jax import grad,vmap

loaded_model=''
mistral_client={}

def init_mistral_client(model):
    """
    Initialize the Mistral client for a specific LLM model.
    """
    global loaded_model, mistral_client
    if mistral_client == {} or loaded_model != model:
        loaded_model = model
        mistral_client = Mistral(os.environ['MISTRAL_KEY'])
    return mistral_client

def tokenize_fn(str, model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    #encoding = init_mistral_client(model).embeddings(model="mistral-embed",input=str)
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
        id = encoding.encode(s) #init_mistral_client(model).embeddings(model="mistral-embed",input=s)
        ids.extend(id)
    return ids

def mistral_api_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Generate text completions from GPT using OpenAI's API.

    Args:
        model (str): Name of the GPT-3 model to use.
        input_str (str): Serialized input time series data.
        steps (int): Number of time steps to predict.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate.
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    avg_tokens_per_step = len(tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))
    # define logit bias to prevent GPT-3 from producing unwanted tokens
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    if model in ['mistral-tiny','mistral-small','mistral-medium']:
        mistral_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        response = init_mistral_client(model).chat(
            model=model,
            messages=[{'role':"system", 'content' : mistral_sys_message},
                      {'role':"user", 'content': extra_input+input_str+settings.time_sep}],
            max_tokens=int(avg_tokens_per_step*steps), 
            temperature=temp,
        )
        return [choice.message.content for choice in response.choices]
    
def mistral_api_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1):
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
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), f'Input string must end with {settings.time_sep}, got {input_str}'
    full_series = input_str + target_str
    response = init_mistral_client(model).chat.stream(model=model, messages={'role':"user",'content':full_series}, max_tokens=0, temperature=temp,)
    #print(response['choices'][0])
    return -1
