from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
    iris = datasets.load_iris()
    
    mlflow.log_param("nrows", nrows)

    return iris

def holdout(df):
    X = df.data
    y = df.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    return (X_train, X_test, y_train, y_test)