import numpy as np
import pandas as pd
import matplotlib as plt
# Encoders and  error measurements
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Regression algorithms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.datasets import make_regression




class ModelRegressionPickerTester():
    """
    This is a class that will, based on pre-cleaned data, pick the best regression model based on heighest r-squared
    and lowest mean and absolute squared error 
    """

    def __init__(self, x_train, x_test, y_train, y_test):
        """
        This initializes the training and the test sets
        
        
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.algorithms = {
            'Multiple Linear': LinearRegression(),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Random Forest': RandomForestRegressor(n_estimators=10, random_state=11),
            'Gradient Boost': GradientBoostingRegressor(n_estimators=1000,max_depth=20, min_samples_split=2,learning_rate=0.01,loss='ls'),
            'Adaboost': AdaBoostRegressor(n_estimators=50, random_state=11),
            'Lasso': linear_model.Lasso(alpha=0.1),
            'ElasticNet': linear_model.ElasticNet(alpha=0.1)
        } 
        
    def get_max(self):
        top_score = {
        'name' : '',
        'lowest_mse' : 0,
        'highest_r_squared' : 0
        }
        
        r_squared_count = 0
        for name, algorithm in self.algorithms.items():
             
            algorithm.fit(self.x_train, self.y_train)
            predictions = algorithm.predict(self.x_test)
            mse = mean_squared_error(y_test,predictions )
            mae = mean_absolute_error(self.y_test, predictions)
            r_squared = r2_score(self.y_test, predictions)
            
            if r_squared < 0:
                print(" Low R Squared value for data may contain alot of noise")
                r_squared_count += 1
                
            if r_squared_count > 3:
                print("Your data may be too noisey")
                return 
            if top_score['name'] == '':
                top_score['lowest_mse'] = mse
                top_score['highest_r_squared'] = r_squared
                top_score['name'] = name
            if mse < top_score['lowest_mse'] and r_squared > top_score['highest_r_squared']:
                top_score['lowest_mse'] = mse
                top_score['highest_r_squared'] = r_squared
                top_score['name'] = name

            
        
        return(top_score)
        
            
if __name__ == "__main__":
    from sklearn.datasets import load_boston
    boston_set = load_boston()
    # print(boston_set.DESCR)
    boston_df = pd.DataFrame(boston_set.data)
    boston_df.columns = boston_set.feature_names
    boston_df['PRICE'] = boston_set.target
    y_train = boston_df['PRICE'][:450]
    x_train = boston_df[:450]
    y_test = boston_df['PRICE'][450:]
    x_test =boston_df[450:]
    test_regression_model = ModelRegressionPickerTester(x_train, x_test, y_train, y_test)
    print(test_regression_model.get_max())