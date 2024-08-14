from flask import jsonify
from app import app

import pickle
import base64
import pandas as pd
import pickle
import numpy as np
from io import StringIO

# Define the path to the pickle file
PICKLE_FILE_PATH = 'src/model.pkl'
CSV_PATH = 'src/learner_test_v4.csv'

@app.route('/')
def index():
  return "Hello, World!"

@app.route('/test')
def test():
  return jsonify({'result': True})


@app.route('/pkl_parser', methods=['POST'])
def main():
  # Load your actual dataset (replace with your own data)
  # actual_df = pd.read_csv("learner_test_v4.csv", nrows=2)
    # Mock CSV content with only one line
#   csv_data = """index,c_id,companyname en(e),tax id(f),year_in_business,equity,asset,total_lia,Leverage,BS_structure,Changes_EBIT,asset_growth,Liquidity,PD,changes_equity
# 0,1,PARA ENGINEERING LIMITED PARTNERSHIP,1.04E+11,20,1818745.52,16082426.69,14263681.17,0.11,0.56,12548412.43,0.09,-0.41,0.72,258597.82"""

#   # Use StringIO to simulate a file object
#   mock_csv = StringIO(csv_data)
#   actual_df = pd.read_csv(mock_csv)
  actual_df = pd.read_csv(CSV_PATH)
  # Preprocess your actual data in the same way as your training data
  # This includes any scaling, encoding, or feature engineering steps you might have done
  # For example, if your model uses the features "count_order", "count_due", etc., you should extract these features from your actual data
  X_actual = actual_df[["asset","equity","total_lia","Leverage", "BS_structure", "Changes_EBIT", "asset_growth","Liquidity","changes_equity"]]

  # print("::: X_actual")
  # print(X_actual)

  # Load your model
  with open(PICKLE_FILE_PATH, 'rb') as file:
    model = pickle.load(file)

  # # Inspect the loaded object
  # for key, value in inspect.getmembers(loaded_object):
  #   # Look for version information or specific attributes
  #   if hasattr(value, '__module__') and value.__module__.startswith('numpy'):
  #       print(f"Found numpy-related attribute: {key}, type: {type(value)}")

  # print("::: model")
  # print(model)
  # Predict probabilities for class 1 using clf_sigmoid
  y_probs_actual_isotonic = model.predict_proba(X_actual)[:, 1]

  # print("::: y_probs_actual_isotonic")
  # print(y_probs_actual_isotonic)

  # Round the predicted probabilities to two decimal places
  rounded_probs = [round(prob, 2) for prob in y_probs_actual_isotonic]

  # print("::: rounded_probs")
  # print(rounded_probs)

  # Add the predicted probabilities as a result column
  actual_df["predicted_proba"] = rounded_probs

  # Save the updated DataFrame or use it as needed
  # For example, if you want to save it to a CSV file:

  # print("::: actual_df")
  # print(actual_df)

  actual_df.to_csv("scoring_test_result_v4-iso.csv", index=False)

  # df = spark.read.format("csv").option("header","true").load("Files/scoring_test_result_v4-iso.csv")
  # display(df)
  return jsonify({'result': 'OK'})
