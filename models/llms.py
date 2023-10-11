from functools import partial
from models.gpt import gpt_completion_fn, gpt_nll_fn
from models.gpt import tokenize_fn as gpt_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn

# Required: Text completion function for each model
# -----------------------------------------------
# Each model is mapped to a function that samples text completions.
# The completion function should follow this signature:
# 
# Args:
#   - input_str (str): String representation of the input time series.
#   - steps (int): Number of steps to predict.
#   - settings (SerializerSettings): Serialization settings.
#   - num_samples (int): Number of completions to sample.
#   - temp (float): Temperature parameter for model's output randomness.
# 
# Returns:
#   - list: Sampled completion strings from the model.
completion_fns = {
    'text-davinci-003': partial(gpt_completion_fn, model='text-davinci-003'),
    'gpt-4': partial(gpt_completion_fn, model='gpt-4'),
    'gpt-3.5-turbo-instruct': partial(gpt_completion_fn, model='gpt-3.5-turbo-instruct'),
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-13b': partial(llama_completion_fn, model='13b'),
    'llama-70b': partial(llama_completion_fn, model='70b'),
    'llama-7b-chat': partial(llama_completion_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_completion_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_completion_fn, model='70b-chat'),
}

# Optional: NLL/D functions for each model
# -----------------------------------------------
# Each model is mapped to a function that computes the continuous Negative Log-Likelihood 
# per Dimension (NLL/D). This is used for computing likelihoods only and not needed for sampling.
# 
# The NLL function should follow this signature:
# 
# Args:
#   - input_arr (np.ndarray): Input time series (history) after data transformation.
#   - target_arr (np.ndarray): Ground truth series (future) after data transformation.
#   - settings (SerializerSettings): Serialization settings.
#   - transform (callable): Data transformation function (e.g., scaling) for determining the Jacobian factor.
#   - count_seps (bool): If True, count time step separators in NLL computation, required if allowing variable number of digits.
#   - temp (float): Temperature parameter for sampling.
# 
# Returns:
#   - float: Computed NLL per dimension for p(target_arr | input_arr).
nll_fns = {
    'text-davinci-003': partial(gpt_nll_fn, model='text-davinci-003'),
    'llama-7b': partial(llama_nll_fn, model='7b'),
    'llama-13b': partial(llama_nll_fn, model='13b'),
    'llama-70b': partial(llama_nll_fn, model='70b'),
    'llama-7b-chat': partial(llama_nll_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_nll_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_nll_fn, model='70b-chat'),
}

# Optional: Tokenization function for each model, only needed if you want automatic input truncation.
# The tokenization function should follow this signature:
#
# Args:
#   - str (str): A string to tokenize.
# Returns:
#   - token_ids (list): A list of token ids.
tokenization_fns = {
    'text-davinci-003': partial(gpt_tokenize_fn, model='text-davinci-003'),
    'gpt-3.5-turbo-instruct': partial(gpt_tokenize_fn, model='gpt-3.5-turbo-instruct'),
    'llama-7b': partial(llama_tokenize_fn, model='7b'),
    'llama-13b': partial(llama_tokenize_fn, model='13b'),
    'llama-70b': partial(llama_tokenize_fn, model='70b'),
    'llama-7b-chat': partial(llama_tokenize_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_tokenize_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_tokenize_fn, model='70b-chat'),
}

# Optional: Context lengths for each model, only needed if you want automatic input truncation.
context_lengths = {
    'text-davinci-003': 4097,
    'gpt-3.5-turbo-instruct': 4097,
    'llama-7b': 4096,
    'llama-13b': 4096,
    'llama-70b': 4096,
    'llama-7b-chat': 4096,
    'llama-13b-chat': 4096,
    'llama-70b-chat': 4096,
}