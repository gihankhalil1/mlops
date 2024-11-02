from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
digits=load_digits()
x=digits.data
y=digits.target
model=RandomForestClassifier()
model.fit(x,y)
y_pred=model.predict(x)
acc=accuracy_score(y,y_pred)
print(acc)
