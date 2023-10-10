import pandas as pd
from darts import TimeSeries
import darts.models
import numpy as np
from darts.utils.likelihood_models import LaplaceLikelihood, GaussianLikelihood
#from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
import torch

likelihoods = {'laplace': LaplaceLikelihood(), 'gaussian': GaussianLikelihood()}

def get_TCN_predictions_data(*args,**kwargs):
    out = get_chunked_AR_predictions_data(darts.models.TCNModel,*args,**kwargs)
    out['info']['Method'] = 'TCN'
    return out

def get_NHITS_predictions_data(*args,**kwargs):
    out = get_chunked_AR_predictions_data(darts.models.NHiTSModel,*args,**kwargs)
    out['info']['Method'] = 'NHiTS'
    return out

def get_NBEATS_predictions_data(*args,**kwargs):
    out = get_chunked_AR_predictions_data(darts.models.NBEATSModel,*args,**kwargs)
    out['info']['Method'] = 'NBEATS'
    return out

def get_chunked_AR_predictions_data(modeltype,train,test, epochs=400, in_len=12, out_len=12, likelihood='laplace', num_samples=100, n_train=None, **kwargs):
    if not isinstance(train, list):
        # assume single train/test case
        train = [train]
        test = [test]
    for i in range(len(train)):    
        # model expects training data to have len at least in_len+out_len
        in_len = min(in_len,len(train[i])-out_len)
        assert in_len > 0, f'Input length must be greater than 0, got {in_len} after subtracting out_len={out_len} from len(train)={len(train)}'
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index = pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index = pd.RangeIndex(len(train[i]),len(test[i])+len(train[i])))
    
    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'
    
    model = modeltype(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        random_state=42,
        likelihood=likelihoods[likelihood],
        pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": [0],
        "max_steps": 10000,
        },
        **kwargs
    )

    scaled_train_ts_list = []
    scaled_test_ts_list = []
    scaled_combined_series_list = []

    scaler = MinMaxScaler()

    # Concatenate all series and fit the scaler
    all_series = train + test
    combined = pd.concat(all_series)
    scaler.fit(combined.values.reshape(-1,1))

    # Iterate over each series in the train list
    for train_series, test_series in zip(train,test):
        scaled_train_series = scaler.transform(train_series.values.reshape(-1,1)).reshape(-1)
        scaled_train_series_ts = TimeSeries.from_times_and_values(train_series.index, scaled_train_series)
        scaled_train_ts_list.append(scaled_train_series_ts)

        scaled_test_series = scaler.transform(test_series.values.reshape(-1,1)).reshape(-1)
        scaled_test_series_ts = TimeSeries.from_times_and_values(test_series.index, scaled_test_series)
        scaled_test_ts_list.append(scaled_test_series_ts)
        
        scaled_combined_series = scaler.transform(pd.concat([train_series,test_series]).values.reshape(-1,1)).reshape(-1)
        scaled_combined_series_list.append(scaled_combined_series)
    
    print('************ Fitting model... ************')
    if n_train is not None:
        model.fit(scaled_train_ts_list[:n_train], epochs=epochs)
    else:
        model.fit(scaled_train_ts_list, epochs=epochs)

    rescaled_predictions_list = []
    BPD_list = []
    samples_list = []
    samples = None
    median = None

    with torch.no_grad():
        predictions = None
        if num_samples > 0:
            print('************ Predicting... ************')
            predictions = model.predict(n=test_len, series=scaled_train_ts_list, num_samples=num_samples)
            for i in range(len(predictions)):
                prediction = predictions[i].data_array()[:,0,:].T.values
                rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape(num_samples,-1)
                samples = pd.DataFrame(rescaled_prediction, columns=test[i].index)
                rescaled_predictions_list.append(rescaled_prediction)
                samples_list.append(samples)
            samples = samples_list if len(samples_list)>1 else samples_list[0]
            median = [samples.median(axis=0) for samples in samples_list] if len(samples_list)>1 else samples_list[0].median(axis=0)
        print('************ Getting likelihood... ************')
        for i in range(len(scaled_combined_series_list)):
            BPD = get_chunked_AR_likelihoods(model,scaled_combined_series_list[i],len(train[i]),in_len,out_len,scaler)
            BPD_list.append(BPD)
        
    out_dict = {
        'NLL/D': np.mean(BPD_list),
        'samples': samples,
        'median': median,
        'info': {'Method':str(modeltype), 'epochs':epochs, 'out_len':out_len}
    }

    return out_dict

def get_chunked_AR_likelihoods(model,scaled_series,trainsize,in_len,out_len,scaler):
    teacher_forced_inputs = torch.from_numpy(scaled_series[trainsize-in_len:][None,:,None])
    testsize = len(scaled_series)-trainsize
    n = 0
    nll_sum = 0
    while n < testsize:
        inp = teacher_forced_inputs[:,n:n+in_len]
        elems_left = min(out_len, testsize-n)
        params = model.model((inp,None))
        likelihood_params = params[:,-out_len:][:,:elems_left]
        likelihood_params2 = model.likelihood._params_from_output(likelihood_params)
        target = teacher_forced_inputs[:,in_len+n:in_len+n+elems_left]
        nll_sum += model.likelihood._nllloss(likelihood_params2,target).detach().numpy()*elems_left
        n += elems_left
    assert n == testsize
    nll_per_dimension = nll_sum/n
    nll_per_dimension -= np.log(scaler.scale_)#np.log(scaler._fitted_params[0].scale_)
    return nll_per_dimension.item()
    
#from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA as staARIMA
import types

def _new_arima_fit(self, series, future_covariates = None):
    super(darts.models.ARIMA,self)._fit(series, future_covariates)

    self._assert_univariate(series)

    # storing to restore the statsmodels model results object
    self.training_historic_future_covariates = future_covariates

    m = staARIMA(
        series.values(copy=False),
        exog=future_covariates.values(copy=False) if future_covariates else None,
        order=self.order,
        seasonal_order=self.seasonal_order,
        trend=self.trend,
        #initialization='approximate_diffuse',
    )
    self.model = m.fit()

    return self

def get_arima_predictions_data(train, test, p=12, d=1, q=0, num_samples=100, **kwargs):
    num_samples = max(num_samples, 1)
    if not isinstance(train, list):
        # assume single train/test case
        train = [train]
        test = [test]
    for i in range(len(train)):    
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index = pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index = pd.RangeIndex(len(train[i]),len(test[i])+len(train[i])))

    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

    model = darts.models.ARIMA(p=p, d=d, q=q)

    scaled_train_ts_list = []
    scaled_test_ts_list = []
    scaled_combined_series_list = []
    scalers = []


    # Iterate over each series in the train list
    for train_series, test_series in zip(train,test):
        # for ARIMA we scale each series individually
        scaler = MinMaxScaler()
        combined_series = pd.concat([train_series,test_series])
        scaler.fit(combined_series.values.reshape(-1,1))
        scalers.append(scaler)
        scaled_train_series = scaler.transform(train_series.values.reshape(-1,1)).reshape(-1)
        scaled_train_series_ts = TimeSeries.from_times_and_values(train_series.index, scaled_train_series)
        scaled_train_ts_list.append(scaled_train_series_ts)

        scaled_test_series = scaler.transform(test_series.values.reshape(-1,1)).reshape(-1)
        scaled_test_series_ts = TimeSeries.from_times_and_values(test_series.index, scaled_test_series)
        scaled_test_ts_list.append(scaled_test_series_ts)
        
        scaled_combined_series = scaler.transform(pd.concat([train_series,test_series]).values.reshape(-1,1)).reshape(-1)
        scaled_combined_series_list.append(scaled_combined_series)
        

    rescaled_predictions_list = []
    nll_all_list = []
    samples_list = []

    for i in range(len(scaled_train_ts_list)):
        try:
            model.fit(scaled_train_ts_list[i])
            prediction = model.predict(len(test[i]), num_samples=num_samples).data_array()[:,0,:].T.values
            scaler = scalers[i]
            rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape(num_samples,-1)
            fit_model = model.model.model.fit()
            fit_params = fit_model.conf_int().mean(1)
            all_model = staARIMA(
                    scaled_combined_series_list[i],
                    exog=None,
                    order=model.order,
                    seasonal_order=model.seasonal_order,
                    trend=model.trend,
            )
            nll_all = -all_model.loglikeobs(fit_params)
            nll_all = nll_all[len(train[i]):].sum()/len(test[i])
            nll_all -= np.log(scaler.scale_)
            nll_all = nll_all.item()
        except np.linalg.LinAlgError:
            rescaled_prediction = np.zeros((num_samples,len(test[i])))
            # output nan
            nll_all = np.nan

        samples = pd.DataFrame(rescaled_prediction, columns=test[i].index)
        
        rescaled_predictions_list.append(rescaled_prediction)
        nll_all_list.append(nll_all)
        samples_list.append(samples)
        
    out_dict = {
        'NLL/D': np.mean(nll_all_list),
        'samples': samples_list if len(samples_list)>1 else samples_list[0],
        'median': [samples.median(axis=0) for samples in samples_list] if len(samples_list)>1 else samples_list[0].median(axis=0),
        'info': {'Method':'ARIMA', 'p':p, 'd':d}
    }

    return out_dict

