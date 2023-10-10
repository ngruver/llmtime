import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from data.small_context import get_datasets
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.darts import get_arima_predictions_data
import numpy as np
import pandas as pd
from collections import defaultdict
from data.serialize import SerializerSettings

def nan_corruption(x:pd.Series, p=0.0):
    x = x.copy()
    x.iloc[np.random.choice(len(x), int(p*len(x)),replace=False)] = np.nan
    # replace p % of the elements with nans
    return x

def interp_nans(x:pd.Series):
    x = x.copy()
    nans = np.isnan(x)
    f = lambda z: z.values.nonzero()[0]
    x[nans] = np.interp(f(nans), f(~nans), x[~nans])
    return x


input_dir = 'outputs/darts' # we use tuned hyperparameters from the darts experiments
output_dir = 'outputs/missing'
os.makedirs(output_dir, exist_ok=True)

datasets = get_datasets()
all_output = {}
for dsname,data in datasets.items():
    train, test = data
    if os.path.exists(f'{input_dir}/{dsname}.pkl'):
        with open(f'{input_dir}/{dsname}.pkl','rb') as f:
            in_dict = pickle.load(f)

        predictions_fns = [get_arima_predictions_data, get_TCN_predictions_data, get_NHITS_predictions_data, get_llmtime_predictions_data]
        model_names = ['arima','TCN','N-HiTS','text-davinci-003']

        output_dict = defaultdict(list)
        for p in [0.,0.1,0.2,0.3,0.4,0.5,.6,.7,.8,.9]:
            output_dict['p'].append(p)
            corrupted_train = nan_corruption(train,p=p)
            interpolated = interp_nans(corrupted_train)
            for predict,model in zip(predictions_fns,model_names):
                best_hyper = in_dict[model]['best_hyper']
                
                if 'settings' in best_hyper and isinstance(best_hyper['settings'], dict):
                    best_hyper['settings'] = SerializerSettings(**best_hyper['settings'])
                if model in ['text-davinci-003']:
                    output_dict[model+'-Nan'].append(predict(corrupted_train.copy(),test.copy(), **best_hyper, num_samples=0)['NLL/D'])
                output_dict[model].append(predict(interpolated.copy(),test.copy(), **best_hyper, num_samples=0)['NLL/D'])
        all_output[dsname] = output_dict

with open(f'{output_dir}/missing.pkl','wb') as f:
    pickle.dump(all_output, f)
            
            