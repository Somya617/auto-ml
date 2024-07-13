import requests
from flask import Flask, request, jsonify
import pandas as pd
import traceback
import h2o
from h2o.automl import H2OAutoML
import io

app = Flask(__name__)

h2o.init()

def process_data(df):
    h2o_df = h2o.H2OFrame(df)
    numeric_cols = [col for col in h2o_df.columns if h2o_df.types[col] in ["numeric", "int"]]
    categorical_cols = [col for col in h2o_df.columns if h2o_df.types[col] in ["enum", "string"]]
    return h2o_df, numeric_cols, categorical_cols

def run_automl(h2o_df, target, model_type):
    print("4.4")
    features = [col for col in h2o_df.columns if col != target]
    print("4.5")
    aml = H2OAutoML(max_runtime_secs=3000, seed=42)
    print("4.6")
    aml.train(x=features, y=target, training_frame=h2o_df)
    print("4.7")
    leaderboard = aml.leaderboard.as_data_frame()
    print("4.8")
    
    # Check if any models were trained
    if len(leaderboard) > 0:
        best_model = h2o.get_model(leaderboard.loc[0, 'model_id'])
        metric = 'auc' if model_type == 'classification' else 'rmse'
        print("4.9")
        return best_model, leaderboard, metric
    
    else:
        print("4.9.2")
        return None, None, None

@app.route('/best_model', methods=['POST'])
def best_model():
    try:
        model_type = request.form.get('model_type')

        # Check if the URL field is provided
        file_path = request.form.get("file")
        file_path = request.files['file']
        print(file_path)
        
        if not file_path:
            return jsonify({"error": "No file path provided."}), 400
        
        # Read the Excel file into a DataFrame
        file_path.stream.seek(0)
        file = file_path.read()
        print("A")
        print(file.decode('utf-8'))
        filestream = io.StringIO(file.decode('utf-8'))
        
        df = pd.read_csv(filestream)
        
        if request.form.get('testing') == 'true':
            df = df.head(100)

        print("!")

        h2o_df, numeric_cols, categorical_cols = process_data(df)
        print("@")
        
        if model_type == 'classification':
            potential_targets = categorical_cols
        elif model_type == 'regression':
            potential_targets = numeric_cols
        else:
            return jsonify({"error": "Unsupported model type. Use classification or regression."}), 400
        
        print("3")
        if len(potential_targets) == 0:
            return jsonify({"error": "Unsupported model type for the given data"}), 400
        
        best_target = None
        best_model = None
        best_leaderboard = None
        best_metric = float('-inf') if model_type == 'classification' else float('inf')
        print("4")
        print(potential_targets)
        for target in potential_targets:
            print("4.4")
            model, leaderboard, metric = run_automl(h2o_df, target, model_type)
            print("4.5")
            
            if model is not None:
                current_metric = model.auc() if model_type == 'classification' else model.rmse()
                
                if (model_type == 'classification' and current_metric > best_metric) or \
                   (model_type == 'regression' and current_metric < best_metric):
                    best_target = target
                    best_model = model
                    best_leaderboard = leaderboard
                    best_metric = current_metric
        
        if best_model is None:
            return jsonify({"error": "No models trained successfully."}), 400
        
        print("5")
        
        return jsonify({
            "best_target": best_target,
            "best_model": best_model.model_id,
            "leaderboard": best_leaderboard.to_dict(orient='records')
        })
    
    except Exception as e:
        print(traceback.format_exc())
        # Log the exception (you can use any logging library or method)
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

