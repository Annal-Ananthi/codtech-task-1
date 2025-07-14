import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# === EXTRACT ===
def extract_inbuilt_data():
    print("Loading inbuilt dataset (diabetes)...")
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    print("Data loaded successfully.")
    return df

# === TRANSFORM ===
def transform_data(df):
    print("Transforming data...")

    # Add some missing values to simulate real-world cases
    df.iloc[1, 2] = None
    df.iloc[3, 4] = None

    # Fill missing values using mean
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

    print("Data transformed successfully.")
    return df_scaled

# === LOAD ===
def load_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Transformed data saved to: {output_file}")

# === MAIN PIPELINE ===
def run_etl(output_file):
    raw_data = extract_inbuilt_data()
    transformed_data = transform_data(raw_data)
    load_data(transformed_data, output_file)

# === ENTRY POINT ===
if __name__ == "__main__":
    run_etl("diabetes_processed.csv")
