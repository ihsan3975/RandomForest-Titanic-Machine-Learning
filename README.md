# SVM-Titanic-Machine-Learning

## Preparing dataset ##
we are have big database that don't know the data looks like. to resolve the issue,

first looking for missing data
```py
data_train.isnull().any()
```

age, cabin and embarked has incomplete data. but age's data have big contributed to predicted survived or survived from titanic disaster

we use age data to predict target, but age incomplete data, so first we do is predicting missing data in age
```py
rfModel_age = RandomForestRegressor()
rfModel_age.fit(titanicWithAge[independentVariables], titanicWithAge['Age'])
generatedAgeValues = rfModel_age.predict(X = titanicWithoutAge[independentVariables])
generatedAgeValues
```

we get age data and save to other file
```py
dataset_comp = pd.concat([data, target], axis = 1)
dataset_comp.to_csv('dataset_train.csv', index = None)
```

## Predict with Random Forest Classifier ##
i am using basic model of random forest, this should be improved by tuning parameters

```py
rf = RandomForestClassifier(n_estimators=1000, random_state = 1506669923)
model = rf.fit(X,y)
```
