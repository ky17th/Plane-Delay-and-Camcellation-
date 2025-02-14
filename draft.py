import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

def run_prediction_app():
    # Load the data
    file_path = r'C:\Users\kayod\OneDrive\For Uni\Year 4 (Masters)\Final Year Project\The Program\Dataset\flights_sample_3m.csv'
    df = pd.read_csv(file_path)

    # Display the first few rows of the dataframe
    print(df.head())

    # Select features and target variable
    features = ['AIRLINE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DISTANCE']
    target_delay = 'DEP_DELAY'
    target_cancel = 'CANCELLED'

    # Preprocessing
    X = df[features]
    y_delay = df[target_delay].apply(lambda x: 1 if x > 15 else 0)  # Binary classification for delay
    y_cancel = df[target_cancel]

    # Train-test split
    X_train, X_test, y_train_delay, y_test_delay = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel = train_test_split(X, y_cancel, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['CRS_DEP_TIME', 'DISTANCE']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['AIRLINE', 'ORIGIN', 'DEST'])
        ]
    )

    # Model pipeline with with_mean=False in StandardScaler
    model_pipeline_delay = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LogisticRegression())
    ])

    model_pipeline_cancel = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LogisticRegression())
    ])

    # Train the models
    model_pipeline_delay.fit(X_train, y_train_delay)
    model_pipeline_cancel.fit(X_train_cancel, y_train_cancel)

    # Evaluate the models
    y_pred_delay = model_pipeline_delay.predict(X_test)
    y_pred_cancel = model_pipeline_cancel.predict(X_test_cancel)

    print('Delay Model Accuracy:', accuracy_score(y_test_delay, y_pred_delay))
    print('Cancellation Model Accuracy:', accuracy_score(y_test_cancel, y_pred_cancel))

    def visualize_model_performance():
        # Confusion Matrix
        cm_delay = confusion_matrix(y_test_delay, y_pred_delay)
        cm_cancel = confusion_matrix(y_test_cancel, y_pred_cancel)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(cm_delay, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title('Confusion Matrix - Delay')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        sns.heatmap(cm_cancel, annot=True, fmt='d', ax=axes[1], cmap='Blues')
        axes[1].set_title('Confusion Matrix - Cancellation')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.show()

        # Classification Report
        print('Classification Report - Delay:')
        print(classification_report(y_test_delay, y_pred_delay))

        print('Classification Report - Cancellation:')
        print(classification_report(y_test_cancel, y_pred_cancel))

    def predict():
        # Get input values
        airline = airline_entry.get()
        origin = origin_entry.get()
        dest = dest_entry.get()
        crs_dep_time = int(crs_dep_time_entry.get())
        distance = int(distance_entry.get())
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'AIRLINE': [airline],
            'ORIGIN': [origin],
            'DEST': [dest],
            'CRS_DEP_TIME': [crs_dep_time],
            'DISTANCE': [distance]
        })
        
        # Predict delay and cancellation
        delay_pred = model_pipeline_delay.predict(input_data)[0]
        cancel_pred = model_pipeline_cancel.predict(input_data)[0]
        
        # Display the result
        delay_result.set(f"Delayed: {'Yes' if delay_pred == 1 else 'No'}")
        cancel_result.set(f"Cancelled: {'Yes' if cancel_pred == 1 else 'No'}")


    # Create the main window
    root = tk.Tk()
    root.title("Flight Delay and Cancellation Prediction")

    # Create and place the widgets
    ttk.Label(root, text="Airline").grid(row=0, column=0)
    global airline_entry
    airline_entry = ttk.Entry(root)
    airline_entry.grid(row=0, column=1)

    ttk.Label(root, text="Origin").grid(row=1, column=0)
    global origin_entry
    origin_entry = ttk.Entry(root)
    origin_entry.grid(row=1, column=1)

    ttk.Label(root, text="Destination").grid(row=2, column=0)
    global dest_entry
    dest_entry = ttk.Entry(root)
    dest_entry.grid(row=2, column=1)

    ttk.Label(root, text="CRS Departure Time").grid(row=3, column=0)
    global crs_dep_time_entry
    crs_dep_time_entry = ttk.Entry(root)
    crs_dep_time_entry.grid(row=3, column=1)

    ttk.Label(root, text="Distance").grid(row=4, column=0)
    global distance_entry
    distance_entry = ttk.Entry(root)
    distance_entry.grid(row=4, column=1)

    predict_button = ttk.Button(root, text="Predict", command=predict)
    predict_button.grid(row=5, column=0, columnspan=2)

    visualize_button = ttk.Button(root, text="Visualize Models", command=visualize_model_performance)
    visualize_button.grid(row=6, column=0, columnspan=2)

    global delay_result
    delay_result = tk.StringVar()
    ttk.Label(root, textvariable=delay_result).grid(row=7, column=0, columnspan=2)

    global cancel_result
    cancel_result = tk.StringVar()
    ttk.Label(root, textvariable=cancel_result).grid(row=8, column=0, columnspan=2)

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    run_prediction_app()

