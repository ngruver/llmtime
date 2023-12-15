import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from darts.datasets import (
    AirPassengersDataset, 
    GasRateCO2Dataset,
    MonthlyMilkDataset,
    WineDataset
)

import os
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from models.llmtime import get_llmtime_predictions_data
from models.llms import nll_fns
from data.serialize import SerializerSettings
from data.synthetic import get_synthetic_datasets

datasets = get_synthetic_datasets()
# print(datasets.keys())
# print(1/0)
# data = datasets['xsin']
data = datasets['linear_cos']

train, test = data
train = train
testt = test

x = np.linspace(0, 1, len(train) + len(test))
train_x = x[:len(train)]
test_x = x[len(train):]
_train_y = train.values
test_y = test.values

# print(train_y)
# plt.plot(train_x, train_y)
# plt.show()

np.random.seed(0)
train_y = _train_y + np.random.normal(0, 0.05, len(_train_y))
# train_y = _train_y + np.linspace(2, 10, len(_train_y)) * np.random.normal(0, 1.0, len(_train_y))
# train_y = _train_y + np.linspace(1, 5, len(_train_y)) * np.random.normal(0, 1.0, len(_train_y))

# print(train_y)
# plt.plot(train_x, train_y)
# plt.show()
# print(1/0)

# # dataset = AirPassengersDataset().load().astype(np.float32)
# dataset = GasRateCO2Dataset().load().astype(np.float32)
# # dataset = MonthlyMilkDataset().load().astype(np.float32)
# # dataset = WineDataset().load().astype(np.float32)
# train, test = dataset[:-100], dataset[-100:]
# # -100 for CO2
# # train, test = dataset.split_before(pd.Timestamp("19580101"))
# x = np.linspace(0, 1, len(dataset))

# print(dataset.pd_dataframe().head())

# # train_y = train.pd_dataframe()["#Passengers"].values
# train_y = train.pd_dataframe()["CO2%"].values
# # train_y = train.pd_dataframe()["Pounds per cow"].values
# # train_y = train.pd_dataframe()["Y"].values
# train_x = x[:len(train_y)]

# # test_y = test.pd_dataframe()["#Passengers"].values
# test_y = test.pd_dataframe()["CO2%"].values
# # test_y = test.pd_dataframe()["Pounds per cow"].values
# # test_y = test.pd_dataframe()["Y"].values
# test_x = x[len(train_y):]

# all_x = np.concatenate([train_x, test_x])
# all_y = np.concatenate([train_y, test_y])
# x_mean, x_std = all_x.mean(), all_x.std()
# y_mean, y_std = all_y.mean(), all_y.std()

# # train_x = (train_x - x_mean) / x_std
# # train_y = (train_y - y_mean) / y_std
# # test_x = (test_x - x_mean) / x_std
# # test_y = (test_y - y_mean) / y_std

# train_x = train_x.reshape(-1, 1)
# train_y = train_y.reshape(-1, 1)
# test_x = test_x.reshape(-1, 1)
# test_y = test_y.reshape(-1, 1)

from pysr import PySRRegressor

model = PySRRegressor(
    niterations=100,  # < Increase me for better results
    binary_operators=["+", "*","-","/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        # "square",
        # "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    # constraints={
    #     "square": 4,
    #     "cube": 4,
    #     "exp": 4,
    # },
    maxsize=70,
    maxdepth=10,
    population_size=50,
    loss="loss(prediction, target) = abs(prediction - target)",
    model_selection='accuracy',
    # parsimony=0,
    # weight_mutate_constant=0.1,
    # weight_mutate_operator=0.75,
    # weight_randomize=0.01
)
model.fit(
    train_x.reshape(-1, 1), 
    train_y.reshape(-1, 1)
)

# model = PySRRegressor.from_file("/Users/nategruver/desktop/hall_of_fame_2023-10-04_114754.955.pkl")
# model = PySRRegressor.from_file("/Users/nategruver/desktop/hall_of_fame_2023-10-04_115256.922.pkl")

# model = PySRRegressor.from_file("hall_of_fame_20023-10-04_154505.764.pkl")
# model = PySRRegressor.from_file("hall_of_fame_2023-10-04_162049.705.pkl")

model = PySRRegressor.from_file("hall_of_fame_2023-10-05_133544.169.pkl")

# model = PySRRegressor.from_file("hall_of_fame_2023-10-05_145612.971.pkl")
# model = PySRRegressor.from_file("hall_of_fame_2023-10-05_170325.867.pkl")

step_size = 4
start_idx = 0
# idxs = list(range(start_idx,len(model.equations_),step_size))
idxs = [1, 5, 9, 17, 25]
fig, ax = plt.subplots(1, len(idxs), figsize=(30, 2))

results = defaultdict(list)

test_losses = []
train_losses = []
complexities = []
nlls = []
for i in range(len(idxs)):
    print(i)
    idx = idxs[i]
    model.sympy(i)

    y_train_pred = model.predict(train_x.reshape(-1, 1), index=idx)
    y_prediction = model.predict(test_x.reshape(-1, 1), index=idx)

    if np.any(np.isinf(y_prediction)):
        print("inf")
        continue

    # print(y_prediction.shape)
    # print(test_y.shape)

    loss = np.square(y_prediction - test_y).mean()
    test_losses.append(loss)

    print(model.equations_.iloc[idx])#.equation)
    print(f"Loss: {loss:.4f}")

    train_losses.append(model.equations_.iloc[idx].loss)
    complexities.append(model.equations_.iloc[idx].complexity)

    results['test_loss'].append(loss)
    results['train_loss'].append(model.equations_.iloc[idx].loss)
    results['complexity'].append(model.equations_.iloc[idx].complexity)
    results['equation'].append(model.equations_.iloc[idx].equation)
    results['test_preds'].append(y_prediction)
    results['train_preds'].append(y_train_pred)

    ax[i].plot(train_x, train_y, color='black')
    ax[i].plot(train_x, y_train_pred, color='red')
    ax[i].plot(test_x, y_prediction, color='red')
    ax[i].plot(test_x, test_y, color='blue')
    # ax[i].set_title(f"Test loss {loss:.1f}")
    # plt.show()
    # plt.close()

    ax[i].set_title(model.equations_.iloc[idx].equation)

    # plt.plot(test_x, (y_prediction - test_y[:,0]) ** 2, color='black')
    # plt.show()
    # plt.close()

    print(test_y)
    print(y_prediction)

    # nll = get_llmtime_predictions_data(
    #     _train_y.flatten(), #+ np.random.normal(0, 0.01, len(train_y.flatten())), 
    #     y_prediction.flatten(), #+ np.random.normal(0, 0.01, len(y_prediction.flatten())),
    #     model='text-davinci-003',
    #     alpha=0.99,
    #     basic=True,
    #     settings= SerializerSettings(10, prec=1, signed=True),
    #     num_samples=0,
    # )['NLL/D']

    # print(nll)
    # nlls.append(nll)

fig.savefig("/Users/nategruver/desktop/simplicity_bias.pdf")
plt.show()
plt.close()

nlls = [-1.2953098912349181, -1.393385559561969, -1.7726323615778958, -1.256776624112951, -1.117701657084411]
print(nlls)
results['nll'] = nlls

results['train_x'] = train_x
results['train_y'] = train_y
results['test_x'] = test_x
results['test_y'] = test_y

#save results as pickle 
import pickle
with open('simplicity_bias_results.pkl', 'wb') as f:
    pickle.dump(results, f)

#normalize test_losses between 0 and 1
test_losses = np.array(test_losses)
test_losses = (test_losses - test_losses.min()) / (test_losses.max() - test_losses.min())
nlls = np.array(nlls)
nlls = (nlls - nlls.min()) / (nlls.max() - nlls.min())
complexities = np.array(complexities)
complexities = (complexities - complexities.min()) / (complexities.max() - complexities.min())
train_losses = np.array(train_losses)
train_losses = (train_losses - train_losses.min()) / (train_losses.max() - train_losses.min())

fig, ax = plt.subplots(figsize=(2, 3))
ax.plot(train_losses, label='Train Loss')
ax.plot(complexities, label='Complexity')
ax.plot(test_losses, label='Test Loss')
ax.plot(nlls, label='LLM NLL')
# ax.set_ylim(0, 500)

# add two column legend above the figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    ncol=2,
    bbox_to_anchor=(1.0, 1.05),
)

fig.savefig("/Users/nategruver/desktop/simplicity_bias_plot.pdf", bbox_inches='tight')
plt.show()
plt.close()