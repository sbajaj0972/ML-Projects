from numpy import array
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=names)

print(df.head())

array = df.values

X, y = array[:,0:8], array[:,8]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3, random_state=101)

#training the model

model = LogisticRegression()
model.fit(X_train,y_train)
print("[INFO]- Model has trained")

result = model.score(X_test, y_test)
print(f"[INFO] - model accuracy is {result}")

# When u are satisfied with accuracy save the model, so that we don't need to do model training again and again
# saving the model
# Can have two extensions .pkl, .sav
filename = 'diabetic_pred.pkl'
joblib.dump(model, filename)