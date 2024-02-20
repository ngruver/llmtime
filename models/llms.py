from functools import partial
from models.gpt import gpt_completion_fn, gpt_nll_fn
from models.gpt import tokenize_fn as gpt_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn

from models.mistral import mistral_completion_fn, mistral_nll_fn
from models.mistral import tokenize_fn as mistral_tokenize_fn

from models.mistral_api import mistral_api_completion_fn, mistral_api_nll_fn
from models.mistral_api import tokenize_fn as mistral_api_tokenize_fn

from models.mistral_api_stocks import mistral_api_stocks_completion_fn
from models.mistral_api_stocks import  mistral_stocks_api_nll_fn 
from models.mistral_api_stocks import  mistral_api_stocks_tokenize_fn

from models.google_gemini_pro import google_gemini_pro_completion_fn, google_gemini_pro_tokenize_fn,google_gemini_pro_nll_fn

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
    'gpt-4-1106-preview':partial(gpt_completion_fn, model='gpt-4-1106-preview'),
    'gpt-3.5-turbo-instruct': partial(gpt_completion_fn, model='gpt-3.5-turbo-instruct'),
    'mistral': partial(mistral_completion_fn, model='mistral'),
    'mistral-api-tiny': partial(mistral_api_completion_fn, model='mistral-tiny'),
    'mistral-api-small': partial(mistral_api_completion_fn, model='mistral-small'),
    'mistral-api-medium': partial(mistral_api_completion_fn, model='mistral-medium'),
    'mistral-api-stocks-tiny': partial(mistral_api_stocks_completion_fn, model='mistral-tiny'),
    'mistral-api-stocks-small': partial(mistral_api_stocks_completion_fn, model='mistral-small'),
    'mistral-api-stocks-medium': partial(mistral_api_stocks_completion_fn, model='mistral-medium'),
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-13b': partial(llama_completion_fn, model='13b'),
    'llama-70b': partial(llama_completion_fn, model='70b'),
    'llama-7b-chat': partial(llama_completion_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_completion_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_completion_fn, model='70b-chat'),
    'gemini-pro': partial(google_gemini_pro_completion_fn, model='gemini-pro'),
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
    'mistral': partial(mistral_nll_fn, model='mistral'),
    'mistral-api-tiny': partial(mistral_api_nll_fn, model='mistral-tiny'),
    'mistral-api-small': partial(mistral_api_nll_fn, model='mistral-small'),
    'mistral-api-medium': partial(mistral_api_nll_fn, model='mistral-medium'),
    'mistral-api-stocks-tiny': partial(mistral_stocks_api_nll_fn, model='mistral-tiny'),
    'mistral-api-stocks-small': partial(mistral_stocks_api_nll_fn, model='mistral-small'),
    'mistral-api-stocks-medium': partial(mistral_stocks_api_nll_fn, model='mistral-medium'),
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-7b': partial(llama_nll_fn, model='7b'),
    'llama-13b': partial(llama_nll_fn, model='13b'),
    'llama-70b': partial(llama_nll_fn, model='70b'),
    'llama-7b-chat': partial(llama_nll_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_nll_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_nll_fn, model='70b-chat'),
    'gemini-pro': partial(google_gemini_pro_nll_fn, model='gemini-pro'),
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
    'mistral': partial(mistral_tokenize_fn, model='mistral'),
    'mistral-api-tiny': partial(mistral_api_tokenize_fn, model='mistral-tiny'),
    'mistral-api-small': partial(mistral_api_tokenize_fn, model='mistral-small'),
    'mistral-api-medium': partial(mistral_api_tokenize_fn, model='mistral-medium'),
    'mistral-api-stocks-tiny': partial(mistral_api_stocks_tokenize_fn, model='mistral-tiny'),
    'mistral-api-stocks-small': partial(mistral_api_stocks_tokenize_fn, model='mistral-small'),
    'mistral-api-stocks-medium': partial(mistral_api_stocks_tokenize_fn, model='mistral-medium'),
    'llama-7b': partial(llama_tokenize_fn, model='7b'),
    'llama-13b': partial(llama_tokenize_fn, model='13b'),
    'llama-70b': partial(llama_tokenize_fn, model='70b'),
    'llama-7b-chat': partial(llama_tokenize_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_tokenize_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_tokenize_fn, model='70b-chat'),
    'gemini-pro': partial(google_gemini_pro_tokenize_fn, model='gemini-pro'),
}

# Optional: Context lengths for each model, only needed if you want automatic input truncation.
context_lengths = {
    'text-davinci-003': 4097,
    'gpt-4': 32000,
    'gpt-3.5-turbo-instruct': 4097,
    'mistral-api-tiny': 4097,
    'mistral-api-small': 4097,
    'mistral-api-medium': 30000,
    'mistral-api-stocks-tiny': 4097,
    'mistral-api-stocks-small': 4097,
    'mistral-api-stocks-medium': 30000,
    'mistral': 4096,
    'llama-7b': 4096,
    'llama-13b': 4096,
    'llama-70b': 4096,
    'llama-7b-chat': 4096,
    'llama-13b-chat': 4096,
    'llama-70b-chat': 4096,
    'gemini-pro': 30000,
}