import itertools
import pickle 
from data.serialize import SerializerSettings
from types import SimpleNamespace
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from scripts.real_benchmarks import get_benchmark_test_sets

def get_datasets():
    benchmarks = get_benchmark_test_sets()
    # shuffle the benchmarks
    for k, v in benchmarks.items():
        x, scaler = v # scaler is not used
        # seed
        np.random.seed(0)
        x = np.random.permutation(x)
        benchmarks[k] = x

    df = pd.read_csv('eval/last_value_results.csv')
    df.sort_values(by='mae')

    df_paper = pd.read_csv('eval/paper_mae_raw.csv') # pdf text -> csv
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
    # remove from df_paper datasets in df_paper but not in df
    df_paper = df_paper[df_paper['Dataset'].isin(df['dataset'])]
    df_paper = df_paper.reset_index(drop=True)
    # for each dataset, add last value mae to df_paper
    for dataset in df_paper['Dataset']:
        df_paper.loc[df_paper['Dataset'] == dataset, 'Last Value'] = df[df['dataset'] == dataset]['mae'].values[0]
    # turn '-' into np.nan
    df_paper = df_paper.replace('-', np.nan)
    # convert all values to float
    for method in df_paper.columns[1:]:
        df_paper[method] = df_paper[method].astype(float)
    df_paper.to_csv('eval/paper_mae.csv', index=False)
    # normalize each method by dividing by last value mae
    for method in df_paper.columns[1:-1]: # skip dataset and last value
        df_paper[method] = df_paper[method] / df_paper['Last Value']
    # sort df by minimum mae across methods
    df_paper['normalized_min'] = df_paper[df_paper.columns[1:-1]].min(axis=1)
    df_paper['normalized_median'] = df_paper[df_paper.columns[1:-1]].median(axis=1)
    df_paper = df_paper.sort_values(by='normalized_min')
    df_paper = df_paper.reset_index(drop=True)
    # save as csv
    df_paper.to_csv('eval/paper_mae_normalized.csv', index=False)

    predictable_datasets = df_paper.head(20)['Dataset']
    datasets = {k: benchmarks[k] for k in predictable_datasets}

    return datasets


# llama hypers
# parser.add_argument("--prec", type=int, default=3)
# parser.add_argument("--time_sep", type=str, default=" ,")
# parser.add_argument("--bit_sep", type=str, default=" ")
# hypers = {
#     "base": 10,
#     "prec": args.prec,
#     "time_sep": args.time_sep,
#     "bit_sep": args.bit_sep,
#     "signed": True,
# }

def run_gpt3(datasets):
    model = 'text-davinci-003'

    alphas = [0.9]
    temps = [0.7]
    max_hists = [500]
    prec = 3

    hypers = []
    for alpha, temp, max_hist in itertools.product(alphas, temps, max_hists):
        hypers.append(
            SimpleNamespace(
                temp=temp, alpha=alpha, beta=0.,max_history=max_hist,
                prompt=None,
                settings=SerializerSettings(base=10, prec=prec, signed=True),
            )
        )

    max_cost = 200
    max_series = 2#000
    max_series_for_tuning = 10
    samples = 5

    total_cost = 0

    finished_list = [
        "covid_deaths", "solar_weekly", "tourism_monthly", "australian_electricity_demand", "pedestrian_counts",
        "traffic_hourly", "hospital", "fred_md", "tourism_yearly", "tourism_quarterly", "us_births",
        "nn5_weekly","solar_10_minutes", "traffic_weekly", "saugeenday", "cif_2016",
        # "weather"
    ]

    load = True
    for ds_name, dataset in datasets.items():
        if ds_name in finished_list or ds_name == "kaggle_web_traffic_weekly":
            continue

        print(ds_name)
        print(len(dataset))

        path = f'eval/{ds_name}.pkl'
        # if load and os.path.exists(path):
        #     continue
        #     # print(f'Loading from file {path}')
        #     # with open(path, 'rb') as f:
        #     #     pred_dict = pickle.load(f)
        # else:
        
        pred_dict = get_gpt_predictions_dataset(
            dataset,
            model,
            hypers[0],
            max_cost,
            max_series,
            samples,
            descriptions=None,
            compute_nll=True,
            max_series_for_tuning=max_series_for_tuning,
            # parallel=False,
        )

        with open(path, 'wb') as f:
            pickle.dump(pred_dict, f)

def main():
    datasets = get_datasets()
    run_gpt3(datasets)

if __name__ == '__main__':
    main()