from pathlib import Path
import yaml

class Config:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.modules_dir = self.project_root / "modules"
        
        self.model_params ={
            "HistGradientBoosterRegressor":{
                "max_depth": [None, 4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "min_samples_leaf": [10, 20, 30, 50],
                "max_leaf_nodes": [15, 31, 63]
            },
            "RandomForestRegressor":{
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 8, 16],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        }
        
        self.feauture_scaling = True
        self.encoding_strategy = "onehot"
        
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class FeatureEngineering:
    def __init__(self, config):
        self.config = config
        self.numerical_features =[
            "bedrooms", "bathrooms", "size_sqft", "floor", "year_built",
            "distance_to_metro_km", "distance_to_mall_km", "listing_month",
            "listing_quarter"
        ]
        self.categorical_features = [
            "community", "sub_community", "furnishing", "view", "property_type",
            "has_pool", "has_gym", "has_parking", "has_balcony", "chiller_free",
            "bills_included", "pet_friendly"
        ]
        
    def build_preprocessor(self):
        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()) if self.config.feauture_scaling else None
        ])

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            if self.config.encoding_strategy == "onehot"
            else OrdinalEncoder()  
        ])
        
        preprocessing_steps = [
            ("num", numeric_pipe, self.numerical_features),
            ("cat", cat_pipe, self.categorical_features)
        ]
        
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import joblib
import logging

def train_model(config, X_train, y_train):
    models = {
        "HistGradientBoosterRegressor": HistGradientBoostingRegressor(random_state=config.seed),
        "RandomForestRegressor": RandomForestRegressor(random_state=config.seed)
    }
    
    results = {}
    for model_name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", FeatureEngineering(config).build_preprocessor()),
            ("model", model)
        ])
        
        search = RandomizedSearchCV(
            pipe,
            param_distributions=config.model_params[model_name],
            cv=5,
            n_iter=10,
            scoring="neg_mean_absolute_error",
            random_state=config.seed,
            verbose=1,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        results[model_name] = {
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": -search.best_score_
        }
    return results

def evaluate_model(result, X_test, y_test):
    evaluation_results = {}
    for model_name, res in result.items():
        y_pred = res["best_model"].predict(X_test)
        evaluation_results[model_name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "params": result["best_params"]
        }
    return evaluation_results

def main():
    config = Config()
    df = pd.read_csv(config.data_dir / "real_estate_data.csv")
    
    df = df[df["rent_aed"].between(1000, 100000)]
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df["listing_month"] = df["listing_date"].dt.month
    df["listing_quarter"] = df["listing_date"].dt.quarter
    
    X = df.drop(columns=["rent_aed"])
    y = df["rent_aed"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.seed
        )
    
    results = train_model(config, X_train, y_train)
    
    eval_results = evaluate_model(results, X_test, y_test)
    
    best_model_name = min(eval_results.items(), key=lambda x: x[1]["MAE"])[0]
    best_model = results[best_model_name]["best_model"]
    
    joblib.dump(best_model.named_steps["preprocessor"], config.models_dir / "preprocessor.joblib")
    joblib.dump(best_model.named_steps["model"], config.models_dir / "rent_model.joblib")