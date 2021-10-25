import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import sys

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
        'n_estimators':[200,250,300],
        'max_depth':[4,5,6],
        'min_samples_split':[3,4,5],
        'min_samples_leaf':[5,6,7],
        'learning_rate':[0.01,0.02],
        'max_features':[0.7,0.8,0.9],
        'loss':['ls','lad','huber']
        }

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

logfile = open('log.txt', 'w')
original_stderr = sys.stderr
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = sys.stdout
grid = GridSearchCV(model, hyperparams, n_jobs=4, verbose=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
sys.stdout = original_stdout
sys.stderr = original_stderr

