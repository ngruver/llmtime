import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.utils import grid_iter
from dataclasses import is_dataclass
from typing import Any

def make_validation_dataset(train, n_val, val_length):
    """Partition the training set into training and validation sets.

    Args:
        train (list): List of time series data for training.
        n_val (int): Number of validation samples.
        val_length (int): Length of each validation sample.

    Returns:
        tuple: Lists of training data without validation, validation data, and number of validation samples.
    """
    assert isinstance(train, list), 'Train should be a list of series'

    train_minus_val_list, val_list = [], []
    if n_val is None:
        n_val = len(train)
    for train_series in train[:n_val]:
        train_len = max(len(train_series) - val_length, 1)
        train_minus_val, val = train_series[:train_len], train_series[train_len:]
        print(f'Train length: {len(train_minus_val)}, Val length: {len(val)}')
        train_minus_val_list.append(train_minus_val)
        val_list.append(val)

    return train_minus_val_list, val_list, n_val


def evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn):
    """Evaluate a set of hyperparameters on the validation set.

    Args:
        hyper (dict): Dictionary of hyperparameters to evaluate.
        train_minus_val (list): List of training samples minus validation samples.
        val (list): List of validation samples.
        get_predictions_fn (callable): Function to get predictions.

    Returns:
        float: NLL/D value for the given hyperparameters, averaged over each series.
    """
    assert isinstance(train_minus_val, list) and isinstance(val, list), 'Train minus val and val should be lists of series'
    return get_predictions_fn(train_minus_val, val, **hyper, num_samples=0)['NLL/D']


def get_autotuned_predictions_data(train, test, hypers, num_samples, get_predictions_fn, verbose=False, parallel=True, n_train=None, n_val=None):
    """
    Automatically tunes hyperparameters based on validation likelihood and retrieves predictions using the best hyperparameters. The validation set is constructed on the fly by splitting the training set.

    Args:
        train (list): List of time series training data.
        test (list): List of time series test data.
        hypers (Union[dict, list]): Either a dictionary specifying the grid search or an explicit list of hyperparameter settings.
        num_samples (int): Number of samples to retrieve.
        get_predictions_fn (callable): Function used to get predictions based on provided hyperparameters.
        verbose (bool, optional): If True, prints out detailed information during the tuning process. Defaults to False.
        parallel (bool, optional): If True, parallelizes the hyperparameter tuning process. Defaults to True.
        n_train (int, optional): Number of training samples to use. Defaults to None.
        n_val (int, optional): Number of validation samples to use. Defaults to None.

    Returns:
        dict: Dictionary containing predictions, best hyperparameters, and other related information.
    """
    if isinstance(hypers,dict):
        hypers = list(grid_iter(hypers))
    else:
        assert isinstance(hypers, list), 'hypers must be a list or dict'
    if not isinstance(train, list):
        train = [train]
        test = [test]
    if n_val is None:
        n_val = len(train)
    if len(hypers) > 1:
        val_length = min(len(test[0]), int(np.mean([len(series) for series in train])/2))
        train_minus_val, val, n_val = make_validation_dataset(train, n_val=n_val, val_length=val_length) # use half of train as val for tiny train sets
        # remove validation series that has smaller length than required val_length
        train_minus_val, val = zip(*[(train_series, val_series) for train_series, val_series in zip(train_minus_val, val) if len(val_series) == val_length])
        train_minus_val = list(train_minus_val)
        val = list(val)
        if len(train_minus_val) <= int(0.9*n_val):
            raise ValueError(f'Removed too many validation series. Only {len(train_minus_val)} out of {len(n_val)} series have length >= {val_length}. Try or decreasing val_length.')
        val_nlls = []
        def eval_hyper(hyper):
            try:
                return hyper, evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn)
            except ValueError:
                return hyper, float('inf')
            
        best_val_nll = float('inf')
        best_hyper = None
        if not parallel:
            for hyper in tqdm(hypers, desc='Hyperparameter search'):
                _,val_nll = eval_hyper(hyper)
                val_nlls.append(val_nll)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_hyper = hyper
                if verbose:
                    print(f'Hyper: {hyper} \n\t Val NLL: {val_nll:3f}')
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(eval_hyper,hyper) for hyper in hypers]
                for future in tqdm(as_completed(futures), total=len(hypers), desc='Hyperparameter search'):
                    hyper,val_nll = future.result()
                    val_nlls.append(val_nll)
                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_hyper = hyper
                    if verbose:
                        print(f'Hyper: {hyper} \n\t Val NLL: {val_nll:3f}')
    else:
        best_hyper = hypers[0]
        best_val_nll = float('inf')
    print(f'Sampling with best hyper... {best_hyper} \n with NLL {best_val_nll:3f}')
    out = get_predictions_fn(train, test, **best_hyper, num_samples=num_samples, n_train=n_train, parallel=parallel)
    out['best_hyper']=convert_to_dict(best_hyper)
    return out
    

def convert_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(elem) for elem in obj]
    elif is_dataclass(obj):
        return convert_to_dict(obj.__dict__)
    else:
        return obj