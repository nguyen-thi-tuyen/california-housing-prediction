from dill import dump, load
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from src.data.make_dataset import main as make_dataset
from src.data.make_dataset import read_train_test_data
import os
import dotenv
import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

data_path = os.getenv("DATA_PATH")
processed_path = os.getenv("PROCESSED_PATH")
make_dataset(data_path, processed_path)
X_train, X_test, y_train, y_test = read_train_test_data(processed_path)
# Modelling
# Pipeline Definition
sc = StandardScaler()
lin_reg = LinearRegression()
pipeline_mlr = Pipeline([("data_scaling", sc), ("estimator", lin_reg)])
# Model Fit
pipeline_mlr.fit(X_train, y_train)
# Model Evaluation
predictions_mlr = pipeline_mlr.predict(X_test)
# Test score
pipeline_mlr.score(X_test, y_test)
print("MAE", metrics.mean_absolute_error(y_test, predictions_mlr))
print("MSE", metrics.mean_squared_error(y_test, predictions_mlr))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, predictions_mlr)))
print("Explained Var Score", metrics.explained_variance_score(y_test, predictions_mlr))


#load the path and define the model name
model_path = os.getenv("MODEL_PATH")
if not os.path.exists(model_path):
   os.mkdir(model_path)
model_file_name = model_path + "/LinearRegression.pkl"

#save the model
with open(model_file_name, "wb") as f:
    dump(pipeline_mlr, f)

#load model
with open(model_file_name, "rb") as f:
    pipeline_mlr = load(f)