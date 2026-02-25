import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(csv_path):
    df=pd.read_csv(csv_path)
    X=df.drop('fraud',axis=1)
    y=df['fraud']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    return model, model.score(X_test,y_test)
