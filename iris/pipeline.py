from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

class IrisPipeline():
    def __init__(self):
        pass
    
    def get_pipeline(self, regressor):
        
        if regressor == "linear_model":
            model = LinearRegression()
        elif regressor == "random_forest":
            model = RandomForestRegressor()
        
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
        
        return pipeline