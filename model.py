import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x_train=np.array([
    [4.5, 0],
    [5.0, 0],
    [5.5, 2],
    [6.0, 4],
    [6.4, 1],
    [6.8, 3],
    [7.0, 2],
    [7.2, 3],
    [7.5, 4],
    [7.8, 2],
    [7.0, 0],
    [7.1, 1],
    [7.5, 0],
    [8.0, 1],
    [8.5, 0],
    [9.0, 1],
])


y_train=np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1])

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x_train)

model=LogisticRegression()
model.fit(x_scaled,y_train)

joblib.dump(model, 'placement_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully")