import gpytorch
from gpytorch.kernels import SpectralMixtureKernel, RBFKernel, ScaleKernel, MaternKernel
import torch
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        covar = SpectralMixtureKernel(num_mixtures=12)
        covar.initialize_from_data(train_x, train_y)
        self.covar_module = ScaleKernel(covar)+RBFKernel()#+ScaleKernel(MaternKernel)
        #self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(x,y, epochs=300, lr=0.05):
    train_x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor(y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SpectralMixtureGPModel(train_x, train_y, likelihood)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood

def test_gp(model, likelihood, test_x,test_y):
    test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(-1)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        preds_mean = preds.mean

    rmse = torch.sqrt(torch.mean((preds_mean - test_y) ** 2)).item()
    return preds_mean, rmse

def get_gp_predictions_data(train, test, epochs=300, lr=0.05, num_samples=100, **kwargs):
    train = train.copy()
    test = test.copy()
    num_samples = max(1,num_samples)
    if not isinstance(train, list):
        # Assume single train/test case
        train = [train]
        test = [test]

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index = pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index = pd.RangeIndex(len(train[i]), len(test[i])+len(train[i])))

    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

    gp_models = []
    gp_likelihoods = []
    BPD_list = []
    gp_mean_list = []
    f_samples_list = []

    for train_series, test_series in zip(train, test):
        # Normalize series
        scaler = MinMaxScaler()
        train_y = scaler.fit_transform(train_series.values.reshape(-1,1)).reshape(-1)
        test_y = scaler.transform(test_series.values.reshape(-1,1)).reshape(-1)
        
        all_t = np.linspace(0, 1, train_series.shape[0]+test_series.shape[0])
        train_x = all_t[:train_series.shape[0]]
        test_x = all_t[train_series.shape[0]:]

        # Train the GP model
        gp_model, gp_likelihood = train_gp(train_x, train_y, epochs=epochs, lr=lr)
        gp_models.append(gp_model)
        gp_likelihoods.append(gp_likelihood)

        # Test the GP model
        with torch.no_grad():
            observed_pred = gp_likelihood(gp_model(torch.tensor(test_x, dtype=torch.float32).unsqueeze(-1)))
            BPD = -observed_pred.log_prob(torch.tensor(test_y, dtype=torch.float32))/(test_y.shape[0])
            BPD -= np.log(scaler.scale_)
            BPD_list.append(BPD.cpu().data.item())
            
            gp_mean = observed_pred.mean.numpy()
            gp_mean = scaler.inverse_transform(gp_mean.reshape(-1,1)).reshape(-1)
            

            f_samples = observed_pred.sample(sample_shape=torch.Size([num_samples])).numpy()
            f_samples = scaler.inverse_transform(f_samples)

            if isinstance(train, pd.Series):
                gp_mean = pd.Series(gp_mean, index=test.index)
                f_samples = pd.DataFrame(f_samples, columns=test.index)

            gp_mean_list.append(gp_mean)
            f_samples_list.append(f_samples)

    out_dict = {
        'NLL/D': np.mean(BPD_list),
        'median': gp_mean_list if len(gp_mean_list)>1 else gp_mean_list[0],
        'samples': f_samples_list if len(f_samples_list)>1 else f_samples_list[0],
        'info': {'Method': 'Gaussian Process','epochs':epochs, 'lr':lr}
    }

    return out_dict
