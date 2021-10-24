import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('./../data/Melbourne_housing_FULL.csv')

df.iloc[100]
