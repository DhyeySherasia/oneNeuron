import pandas as pd
import os
import joblib

def prepare_data(df):
    x = df.drop("y", axis=1)
    y = df["y"]

    return x, y

def save_model(model, filename):
    model_dir = "models"

    try:
        os.mkdir(model_dir)  # Ceate if 'model_dir' does not exist
    except OSError as e:
        print()

    # Joins the path of file to the directory
    file_path = os.path.join(model_dir, filename)  # Output: 'models/filename'

    joblib.dump(model, file_path)

