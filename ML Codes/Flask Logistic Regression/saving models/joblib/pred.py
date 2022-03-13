import joblib
#Load the model

model = joblib.load('diabetic_pred.pkl')
data = model.predict([[6,3,1,8,9,2,4,8]])

if data[0] == 0:
    print('person is not diabetic')
else:
    print('person is diabetic')