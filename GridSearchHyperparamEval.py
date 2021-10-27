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

model = ensemble.GradientBoostingRegressor(learning_rate=0.1,loss='huber',max_depth=5, max_features=0.6,min_samples_leaf=6,min_samples_split=4,n_estimators=300)
#model = ensemble.GradientBoostingRegressor(learning_rate=0.02, loss='squared_error', max_depth=6, max_features=0.7, min_samples_leaf=5, min_samples_split=5, n_estimators=300)
model.fit(X_train,y_train)

mae_train = mean_absolute_error(y_train,model.predict(X_train))
print("TRAIN MAE: %.2f" % mae_train)

mae_test = mean_absolute_error(y_test,model.predict(X_test))
print("TEST MAE: %.2f" % mae_test)
