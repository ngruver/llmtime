import numpy as np
from jax import vmap
import jax.numpy as jnp

def quantile_loss(target, pred, q):
    q_pred = jnp.quantile(pred, q, axis=0)
    return 2 * jnp.sum(
        jnp.abs((q_pred - target) * ((target <= q_pred) * 1.0 - q))
    )

def calculate_crps(target, pred, num_quantiles=20):
    quantiles = jnp.linspace(0, 1.0, num_quantiles+1)[1:]
    vec_quantile_loss = vmap(lambda q: quantile_loss(target, pred, q))
    crps = jnp.sum(vec_quantile_loss(quantiles))
    crps = crps / (jnp.sum(np.abs(target)) * len(quantiles))
    return crps

import jax
from jax import grad,vmap
from .serialize import serialize_arr, SerializerSettings
import openai

def nll(input_arr, target_arr, model, settings:SerializerSettings, transform, count_seps=True, prompt=None, temp=1):
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
    response = openai.Completion.create(model=model, prompt=prompt, logprobs=5, max_tokens=0, echo=True, temperature=temp)
    #print(response['choices'][0])
    logprobs = np.array(response['choices'][0].logprobs.token_logprobs, dtype=np.float32)
    tokens = np.array(response['choices'][0].logprobs.tokens)
    top5logprobs = response['choices'][0].logprobs.top_logprobs
    seps = tokens==settings.time_sep
    target_start = np.argmax(np.cumsum(seps)==len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens==settings.time_sep
    assert len(logprobs[seps]) == len(target_arr), f'There should be one separator per target. Got {len(logprobs[seps])} separators and {len(target_arr)} targets.'
    #adjust logprobs by removing extraneous and renormalizing (see appendix of paper)
    # logp' = logp - log(1-pk*pextra)
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
    #print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx

class Evaluator:

    def __init__(self):
        self.non_numerical_cols = [
            "serialized_history",
            "serialized_target",
            "serialized_prediction",
            "history_len",
            "num_channels",
            "example_num",
            "sample_num",
        ]

    def evaluate_df(self, gt_df, pred_df):
        cols = [c for c in gt_df.columns if c not in self.non_numerical_cols]
        num_channels = gt_df["num_channels"].iloc[0]
        history_len = gt_df["history_len"].iloc[0]
        gt_vals = gt_df[cols].to_numpy().reshape(len(gt_df), -1, num_channels) # (num_examples, history_len + target_len, num_channels)
        gt_vals = gt_vals[:, history_len:, :] # (num_examples, target_len, num_channels)
        
        cols = [c for c in pred_df.columns if c not in self.non_numerical_cols]
        num_channels = pred_df["num_channels"].iloc[0]
        pred_df = pred_df[cols + ["example_num"]]
        
        all_pred_vals = []
        for example_num in sorted(pred_df["example_num"].unique()):
            pred_vals = pred_df[pred_df["example_num"] == example_num][cols].to_numpy() # (num_samples, target_len * num_channels)
            pred_vals = pred_vals.reshape(pred_vals.shape[0], -1, num_channels) # (num_samples, target_len, num_channels)
            all_pred_vals.append(pred_vals)
           
        pred_vals = np.stack(all_pred_vals, axis=1) # (num_samples, num_examples, target_len, num_channels)
        assert gt_vals.shape == pred_vals.shape[1:]
        
        diff = (gt_vals[None] - pred_vals) # (num_samples, num_examples, target_len, num_channels)
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        crps = calculate_crps(gt_vals, pred_vals)

        return {
            "mse": mse,
            "mae": mae,
            "crps": crps,
        }
    
    def evaluate(self, gt, pred):
        ''' 
        gt: (batch_size, steps)
        pred: (batch_size, num_samples, steps)
        '''
        assert gt.shape == (pred.shape[0], pred.shape[2]), f"wrong shapes: gt.shape: {gt.shape}, pred.shape: {pred.shape}"
        diff = (gt[:, None, :] - pred) # (batch_size, num_samples, steps)
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        std = np.std(gt, axis=1) + 1e-8 # (batch_size,)
        normlized_diff = diff / std[:, None, None] # (batch_size, num_samples, steps)
        nmse = np.mean(normlized_diff**2)
        nmae = np.mean(np.abs(normlized_diff))

        return {
            "nmse": nmse,
            "nmae": nmae,
            "mse": mse,
            "mae": mae,
        }