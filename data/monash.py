import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import datasets
from datasets import load_dataset
import os
import pickle

fix_pred_len = {
    'australian_electricity_demand': 336,
    'pedestrian_counts': 24,
    'traffic_hourly': 168,    
}

def get_benchmark_test_sets():
    test_set_dir = "datasets/monash"
    if not os.path.exists(test_set_dir):
        os.makedirs(test_set_dir)

    if len(os.listdir(test_set_dir)) > 0:
        print(f'Loading test sets from {test_set_dir}')
        test_sets = {}
        for file in os.listdir(test_set_dir):
            test_sets[file.split(".")[0]] = pickle.load(open(os.path.join(test_set_dir, file), 'rb'))
        return test_sets
    else:
        print(f'No files found in {test_set_dir}. You are not using our preprocessed datasets!')
    
    benchmarks = {
        "monash_tsf": datasets.get_dataset_config_names("monash_tsf"),
    }

    test_sets = defaultdict(list)
    for path in benchmarks:
        pred_lens = [24, 48, 96, 192] if path == "ett" else [None]
        for name in benchmarks[path]:
            for pred_len in pred_lens:
                if pred_len is None:
                    ds = load_dataset(path, name)
                else:
                    ds = load_dataset(path, name, multivariate=False, prediction_length=pred_len)
                
                train_example = ds['train'][0]['target']
                val_example = ds['validation'][0]['target']

                if len(np.array(train_example).shape) > 1:
                    print(f"Skipping {name} because it is multivariate")
                    continue

                pred_len = len(val_example) - len(train_example)
                if name in fix_pred_len:
                    print(f"Fixing pred len for {name}: {pred_len} -> {fix_pred_len[name]}")
                    pred_len = fix_pred_len[name]

                tag = name
                print("Processing", tag)

                pairs = []
                for x in ds['test']:
                    if np.isnan(x['target']).any():
                        print(f"Skipping {name} because it has NaNs")
                        break
                    history = np.array(x['target'][:-pred_len])
                    target = np.array(x['target'][-pred_len:])
                    pairs.append((history, target))
                else:
                    scaler = None
                    if path == "ett":
                        trainset = np.array(ds['train'][0]['target'])
                        scaler = StandardScaler().fit(trainset[:,None])
                    test_sets[tag] = (pairs, scaler)
    
    for name in test_sets:
        try:
            with open(os.path.join(test_set_dir,f"{name}.pkl"), 'wb') as f:
                pickle.dump(test_sets[name], f)
            print(f"Saved {name}")
        except:
            print(f"Failed to save {name}")

    return test_sets

def get_datasets():
    benchmarks = get_benchmark_test_sets()
    # shuffle the benchmarks
    for k, v in benchmarks.items():
        x, _scaler = v # scaler is not used
        train, test = zip(*x)
        np.random.seed(0)
        ind = np.arange(len(train))
        ind = np.random.permutation(ind)
        train = [train[i] for i in ind]
        test = [test[i] for i in ind]
        benchmarks[k] = [list(train), list(test)]

    df = pd.read_csv('data/last_val_mae.csv')
    df.sort_values(by='mae')

    df_paper = pd.read_csv('data/paper_mae_raw.csv') # pdf text -> csv
    datasets = df_paper['Dataset']
    name_map = {
        'Aus. Electricity Demand' :'australian_electricity_demand',
        'Kaggle Weekly': 'kaggle_web_traffic_weekly',
        'FRED-MD': 'fred_md',
        'Saugeen River Flow': 'saugeenday',
        
    }
    datasets = [name_map.get(d, d) for d in datasets]
    # lower case and repalce spaces with underscores
    datasets = [d.lower().replace(' ', '_') for d in datasets]
    df_paper['Dataset'] = datasets
    df_paper = df_paper.reset_index(drop=True)
    # for each dataset, add last value mae to df_paper
    for dataset in df_paper['Dataset']:
        if dataset in df['dataset'].values:
            df_paper.loc[df_paper['Dataset'] == dataset, 'Last Value'] = df[df['dataset'] == dataset]['mae'].values[0]
    # turn '-' into np.nan
    df_paper = df_paper.replace('-', np.nan)
    # convert all values to float
    for method in df_paper.columns[1:]:
        df_paper[method] = df_paper[method].astype(float)
    df_paper.to_csv('data/paper_mae.csv', index=False)
    # normalize each method by dividing by last value mae
    for method in df_paper.columns[1:-1]: # skip dataset and last value
        df_paper[method] = df_paper[method] / df_paper['Last Value']
    # sort df by minimum mae across methods
    df_paper['normalized_min'] = df_paper[df_paper.columns[1:-1]].min(axis=1)
    df_paper['normalized_median'] = df_paper[df_paper.columns[1:-1]].median(axis=1)
    df_paper = df_paper.sort_values(by='normalized_min')
    df_paper = df_paper.reset_index(drop=True)
    # save as csv
    df_paper.to_csv('data/paper_mae_normalized.csv', index=False)
    return benchmarks

def main():
    get_datasets()
    
if __name__ == "__main__":
    main()