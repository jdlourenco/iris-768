from iris.data import get_data, holdout
from iris.pipeline import IrisPipeline
import joblib

class Trainer:
    
    def __init__(self):
        pass
    
    def save_model(self):
        joblib.dump(self.pipeline, 'model2.joblib')
    
    def train(self):
        print("get data")
        df = get_data()
        # print(df)
        
        print("holdout")
        X_train, X_test, y_train, y_test = holdout(df)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
        print("get pipeline")
        iris_pipe = IrisPipeline()
        self.pipeline = iris_pipe.get_pipeline(regressor="random_forest")
        print(self.pipeline)
        
        print("train pipeline")
        self.pipeline.fit(X_train, y_train)
        
        print("save model to disk")
        self.save_model()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
