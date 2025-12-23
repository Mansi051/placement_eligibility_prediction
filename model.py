import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x_train=np.array([
    [5.5, 4],
    [6.0, 3],
    [6.5, 2],
    [6.8,1],
    [7.0,0],
    [7.2,0],
    [7.5, 0],
    [7.8, 1],
    [8.0, 0],
    [8.5, 0],
])


y_train=np.array([0,0,0,0,1,1,1,1,1,1])

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x_train)

model=LogisticRegression()
model.fit(x_scaled,y_train)

joblib.dump(model, 'placement_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully")