import torch
import numpy as np
from jax import grad,vmap
from tqdm import tqdm
import argparse
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
)
from data.serialize import serialize_arr, deserialize_str, SerializerSettings

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

loaded = {}

def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

def get_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1
    assert model_size in ["7b", "13b", "70b"]

    tokenizer = LlamaTokenizer.from_pretrained(
        llama2_model_string(model_size, chat),
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model_and_tokenizer(model_name, cache_model=False):
    if model_name in loaded:
        return loaded[model_name]
    name_parts = model_name.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]

    tokenizer = get_tokenizer(model_name)

    model = LlamaForCausalLM.from_pretrained(
        llama2_model_string(model_size, chat),
        device_map="auto",   
        torch_dtype=torch.float16,
    )
    model.eval()
    if cache_model:
        loaded[model_name] = model, tokenizer
    return model, tokenizer

def tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)

def llama_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1, cache_model=True):
    """ Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM 
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
        cache_model: whether to cache the model and tokenizer for faster repeated calls
    Returns: NLL/D
    """
    model, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_series = input_str + target_str
    
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out['logits'][:,:,bad_tokens] = -100

    input_ids = batch['input_ids'][0][1:]
    logprobs = torch.nn.functional.log_softmax(out['logits'], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()

    tokens = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=False, 
        clean_up_tokenization_spaces=False
    )
    
    input_len = len(tokenizer([input_str], return_tensors="pt",)['input_ids'][0])
    input_len = input_len - 2 # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum()/len(target_arr)

    #print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx

def llama_completion_fn(
    model,
    input_str,
    steps,
    settings,
    batch_size=5,
    num_samples=20,
    temp=0.9, 
    top_p=0.9,
    cache_model=True
):
    avg_tokens_per_step = len(tokenize_fn(input_str, model)['input_ids']) / len(input_str.split(settings.time_sep))
    max_tokens = int(avg_tokens_per_step*steps)
    
    model, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    gen_strs = []
    for _ in tqdm(range(num_samples // batch_size)):
        batch = tokenizer(
            [input_str], 
            return_tensors="pt",
        )

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}
        num_input_ids = batch['input_ids'].shape[1]

        good_tokens_str = list("0123456789" + settings.time_sep)
        good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
        # good_tokens += [tokenizer.eos_token_id]
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp, 
            top_p=top_p, 
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )
        gen_strs += tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
    return gen_strs
