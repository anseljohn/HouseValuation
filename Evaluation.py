import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import sys

runnum = 2
df = pd.read_csv('../ML-Exercises/data/Melbourne_housing_FULL.csv')

df.columns = df.columns.str.strip()
scrubbed = ['Address','Method','SellerG','Date','Postcode','Lattitude','Longtitude','Regionname','Propertycount']
for title in scrubbed:
    del df[title]

df.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
df = pd.get_dummies(df,columns=['Suburb','CouncilArea','Type'])

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=True, random_state=42)


model = ensemble.GradientBoostingRegressor()

hyperparams = {
        'n_estimators':range(200,350,50),
        'max_depth':range(3,8),
        'min_samples_split':range(2,7),
        'min_samples_leaf':range(4,9),
        'learning_rate':[round(num,1) for num in np.arange(0.1,0.3,0.1)],
        'max_features':[round(num,1) for num in np.arange(0.5,1.0,0.1)],
        }


ls1 = {'loss':['squared_error','lad','huber']}
ls2 = {'loss':['ls','lad','huber']}

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

logfile = open('./logs/log' + str(runnum) + '.txt', 'w')
original_stderr = sys.stderr
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = sys.stdout
try:
    hp = hyperparams.copy()
    hp.update(ls1)
    grid = GridSearchCV(model, hyperparams, n_jobs=4, verbose=10)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
except ValueError:
    hp = hyperparams.copy()
    hp.update(ls2)
    grid = GridSearchCV(model, hyperparams, n_jobs=4, verbose=10)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

sys.stdout = original_stdout
sys.stderr = original_stderr
logfile.close()
