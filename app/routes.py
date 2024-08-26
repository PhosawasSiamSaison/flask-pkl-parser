from flask import jsonify, request, abort
from app import app

import pickle
import base64
import pandas as pd
import pickle
import numpy as np
from io import StringIO
from datetime import datetime
import json
import csv
import base64

# Define the path to the pickle file
# PICKLE_FILE_PATH = 'src/model.pkl'
# CSV_PATH = 'src/learner_test_v4.csv'
PICKLE_FILE_PATH = 'src/scoring_model_v1_tough_napkin_kk02b485.pkl'
CSV_PATH = 'src/learner_test_v1.csv'

@app.route('/')
def index():
  return "Hello, World!"

@app.route('/test')
def test():
  return jsonify({'result': True})


@app.route('/v1/pkl_parser', methods=['POST'])
def pkl_parser_v1():
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
  # X_actual = actual_df[["asset","equity","total_lia","Leverage", "BS_structure", "Changes_EBIT", "asset_growth","Liquidity","changes_equity"]]
  
  # v1
  X_actual = actual_df[["asset_2022","equity_2022","total_lia_2022","Leverage", "BS_structure", "Changes_EBITDA", "asset_growth","Liquidity"]]

  # Load your model
  with open(PICKLE_FILE_PATH, 'rb') as file:
    model = pickle.load(file)

  # # Inspect the loaded object
  # for key, value in inspect.getmembers(loaded_object):
  #   # Look for version information or specific attributes
  #   if hasattr(value, '__module__') and value.__module__.startswith('numpy'):
  #       print(f"Found numpy-related attribute: {key}, type: {type(value)}")

  # Predict probabilities for class 1 using clf_sigmoid
  y_probs_actual_isotonic = model.predict_proba(X_actual)[:, 1]

  # Round the predicted probabilities to two decimal places
  rounded_probs = [round(prob, 2) for prob in y_probs_actual_isotonic]

  # Add the predicted probabilities as a result column
  actual_df["predicted_proba"] = rounded_probs

  # Save the updated DataFrame or use it as needed
  # For example, if you want to save it to a CSV file:

  # actual_df.to_csv("scoring_test_result_v4-iso.csv", index=False)
  actual_df.to_csv("scoring_test_result_v1-iso.csv", index=False)

  # df = spark.read.format("csv").option("header","true").load("Files/scoring_test_result_v4-iso.csv")
  # display(df)
  return jsonify({'result': 'OK', 'rounded_probs': rounded_probs})

@app.route('/v4/pkl_parser', methods=['POST'])
def pkl_parser_v4():
  pickle_path = 'src/model.pkl'
  csv_path = 'src/learner_test_v4.csv'
  actual_df = pd.read_csv(csv_path)
  # Preprocess your actual data in the same way as your training data
  # This includes any scaling, encoding, or feature engineering steps you might have done
  # For example, if your model uses the features "count_order", "count_due", etc., you should extract these features from your actual data
  X_actual = actual_df[["asset","equity","total_lia","Leverage", "BS_structure", "Changes_EBIT", "asset_growth","Liquidity","changes_equity"]]

  # Load your model
  with open(pickle_path, 'rb') as file:
    model = pickle.load(file)

  # # Inspect the loaded object
  # for key, value in inspect.getmembers(loaded_object):
  #   # Look for version information or specific attributes
  #   if hasattr(value, '__module__') and value.__module__.startswith('numpy'):
  #       print(f"Found numpy-related attribute: {key}, type: {type(value)}")

  # Predict probabilities for class 1 using clf_sigmoid
  y_probs_actual_isotonic = model.predict_proba(X_actual)[:, 1]

  # Round the predicted probabilities to two decimal places
  rounded_probs = [round(prob, 2) for prob in y_probs_actual_isotonic]

  # Add the predicted probabilities as a result column
  actual_df["predicted_proba"] = rounded_probs

  # Save the updated DataFrame or use it as needed
  # For example, if you want to save it to a CSV file:

  actual_df.to_csv("scoring_test_result_v4-iso.csv", index=False)
  return jsonify({'result': 'OK', 'rounded_probs': rounded_probs})

@app.route('/pkl_model_calculate', methods=['POST'])
def pkl_model_calculate():
  data = request.get_json()

  request_data = data.get('data')
        
  # Extract the base64-encoded file from the JSON data
  pkl_file_base64 = data.get('pkl_file')
  

  if request_data:
    try:
        # If data is a string, parse it as JSON
        if isinstance(request_data, str):
            data = json.loads(request_data)
        else:
            data = request_data
    except (json.JSONDecodeError, TypeError):
        abort(400, description="Invalid JSON format for data field")
  else:
    abort(400, description="No data field provided")

  # Extract the financial data
  financial_data = data.get('financial', [])

  if not financial_data:
      abort(400, description="No financial data provided")

  current_year = str(datetime.now().year)
  previous_year = str(datetime.now().year - 1)
  previous_2_year = str(datetime.now().year - 2)

  # Find financial data for the current year and the previous year
  current_year_data = next((item for item in financial_data if item['fiscalYear'] == current_year), None)
  previous_year_data = next((item for item in financial_data if item['fiscalYear'] == previous_year), None)
  previous_2_year_data = next((item for item in financial_data if item['fiscalYear'] == previous_2_year), None)
  
  if not current_year_data:
    current_year_data = previous_year_data
    current_year = previous_year
    previous_year_data = previous_2_year_data
    previous_year = previous_2_year

  # Check if both years' data are present
  if not current_year_data or not previous_year_data:
    missing_years = []
    if not current_year_data:
      missing_years.append(current_year)
    if not previous_year_data:
      missing_years.append(previous_year)
    abort(400, description=f"Missing financial data for years: {', '.join(missing_years)}")

  try:
    # data for current_year
    asset = validate_field(current_year_data, "totalAssets", current_year)
    equity = validate_field(current_year_data, "totalShareholdersEquity", current_year)
    total_lia = validate_field(current_year_data, "totalLiabilities", current_year)
    leverage = round(equity / asset, 2)
    total_current_lia = validate_field(current_year_data, "totalCurrentLiabilities", current_year)
    bs_structure = round(total_current_lia / total_lia, 2)
    current_income_loss_before_interest_and_income_taxes = validate_field(current_year_data, "incomeBeforeInterestAndIncomeTaxes", current_year)
    total_current_asset = validate_field(current_year_data, "totalCurrentAssets", current_year)
    liquidity = round((total_current_asset - total_current_lia)/ asset, 2)

    # data for previous_year
    previous_income_loss_before_interest_and_income_taxes = validate_field(previous_year_data, "incomeBeforeInterestAndIncomeTaxes", previous_year)
    previous_asset = validate_field(previous_year_data, "totalAssets", previous_year)
    previous_equity = validate_field(previous_year_data, "totalShareholdersEquity", previous_year)

    # calculate for both year
    changes_ebit = round(current_income_loss_before_interest_and_income_taxes - previous_income_loss_before_interest_and_income_taxes, 2)
    asset_growth = round((asset - previous_asset)/ previous_asset, 2)
    changes_equity = round(equity - previous_equity, 2)
  except ValueError as e:
    abort(400, description=str(e))

  # insert data to model
  headers = [
    "asset","equity","total_lia","Leverage", "BS_structure",
    "Changes_EBIT","asset_growth","Liquidity","changes_equity"
  ]
  
  # Define the calculated row data
  calculated_data = {
    "asset": asset,
    "equity": equity,
    "total_lia": total_lia,
    "Leverage": leverage,
    "BS_structure": bs_structure,
    "Changes_EBIT": changes_ebit,
    "asset_growth": asset_growth,
    "Liquidity": liquidity,
    "changes_equity": changes_equity,
  }

  output = StringIO()
  writer = csv.DictWriter(output, fieldnames=headers)
  writer.writeheader()
  writer.writerow(calculated_data)
  csv_string = output.getvalue()

  mock_csv = StringIO(csv_string)
  actual_df = pd.read_csv(mock_csv)

  # Preprocess your actual data in the same way as your training data
  # This includes any scaling, encoding, or feature engineering steps you might have done
  # For example, if your model uses the features "count_order", "count_due", etc., you should extract these features from your actual data
  X_actual = actual_df[["asset","equity","total_lia","Leverage", "BS_structure", "Changes_EBIT", "asset_growth","Liquidity","changes_equity"]]

  try:
    pkl_file = base64.b64decode(pkl_file_base64)
    # Read the file content
    # file_content = pkl_file.read()

    # Load the model from the pickle file
    model = pickle.loads(pkl_file)

    # # Inspect the loaded object
    # for key, value in inspect.getmembers(loaded_object):
    #   # Look for version information or specific attributes
    #   if hasattr(value, '__module__') and value.__module__.startswith('numpy'):
    #       print(f"Found numpy-related attribute: {key}, type: {type(value)}")

    # Call the predict_proba method
    y_probs_actual_isotonic = model.predict_proba(X_actual)[:, 1]


    # Round the predicted probabilities to two decimal places
    rounded_probs = [round(prob, 2) for prob in y_probs_actual_isotonic]

    # Add the predicted probabilities as a result column
    actual_df["predicted_proba"] = rounded_probs
  except Exception as e:
    return jsonify({
      "result": "NG",
      "error": "An error occurred while processing the file.",
      "message": str(e)
    }), 400
  
  # calculate credit_limit
  try:
    # data for current_year
    total_revenue = validate_field(current_year_data, "totalRevenue", current_year)
    total_current_lia
    total_current_asset
    cr_calculate = round((total_current_asset / 3) - total_current_lia)
    tr_calculate = round(total_revenue / 100)
    if cr_calculate >= 0:
      ss_credit_line = cr_calculate
    else:
      ss_credit_line = tr_calculate
    check_creden_line = round((total_current_asset - total_current_lia) / 10)
    if check_creden_line < 0:
      creden_credit_line = round(total_revenue / 100)
    else:
      creden_credit_line = check_creden_line
    suggest_initial_limit = round(max(ss_credit_line, creden_credit_line) * 20 / 100)
  except ValueError as e:
    abort(400, description=str(e))

  return jsonify(
    {
      'result': "OK",
      'fs_year': current_year,
      'asset': asset,
      'equity': equity,
      'total_lia': total_lia,
      'leverage': leverage,
      'bs_structure': bs_structure,
      'changes_ebit': changes_ebit,
      'asset_growth': asset_growth,
      'liquidity': liquidity,
      'changes_equity': changes_equity,
      'rounded_prob': rounded_probs[0],
      'ss_credit_line': ss_credit_line,
      'creden_credit_line': creden_credit_line,
      'suggest_initial_limit': suggest_initial_limit
    }
  )
  

@app.errorhandler(400)
def handle_400_error(e):
    response = jsonify({
      "result": "NG",
      "error": "Bad Request",
      "message": str(e.description)
    })
    response.status_code = 400
    return response

@app.errorhandler(ZeroDivisionError)
def handle_zero_division_error(e):
  response = jsonify({
    "result": "NG",
    "error": "Division by Zero Error",
    "message": "An attempt was made to divide by zero."
  })
  response.status_code = 400
  return response


def validate_field(data, field_name, year):
  field_value = data.get(field_name)
  if field_value is None:
    raise ValueError(f"Value not found for '{field_name}' in year {year}")

  try:
    return float(field_value.replace(',', '').strip())
  except ValueError:
    raise ValueError(f"Invalid value for field '{field_name}' in year {year}")
