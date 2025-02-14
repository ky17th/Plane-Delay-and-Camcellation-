import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy import stats

# Load the data
file_path = r'C:\Users\kayod\OneDrive\For Uni\Year 4 (Masters)\Final Year Project\The Program\Dataset\flights_sample_3m.csv'
df = pd.read_csv(file_path)

# Outlier Detection using Z-score method for CRS_DEP_TIME and DISTANCE
df = df[(np.abs(stats.zscore(df[['CRS_DEP_TIME', 'DISTANCE']])) < 3).all(axis=1)]

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

# Hyperparameter Tuning with GridSearchCV
model = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Model pipeline with GridSearchCV for delay and cancellation
model_pipeline_delay = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', grid_search)  # GridSearchCV used here
])

model_pipeline_cancel = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', grid_search)  # GridSearchCV used here
])

# Train the models
model_pipeline_delay.fit(X_train, y_train_delay)
model_pipeline_cancel.fit(X_train_cancel, y_train_cancel)

# Predictions on test data
y_pred_delay = model_pipeline_delay.predict(X_test)
y_pred_cancel = model_pipeline_cancel.predict(X_test_cancel)

# Evaluation metrics for delay prediction
delay_accuracy = accuracy_score(y_test_delay, y_pred_delay)
delay_precision = precision_score(y_test_delay, y_pred_delay)
delay_recall = recall_score(y_test_delay, y_pred_delay)
delay_f1 = f1_score(y_test_delay, y_pred_delay)
delay_auc_roc = roc_auc_score(y_test_delay, y_pred_delay)

# Evaluation metrics for cancellation prediction
cancel_accuracy = accuracy_score(y_test_cancel, y_pred_cancel)
cancel_precision = precision_score(y_test_cancel, y_pred_cancel)
cancel_recall = recall_score(y_test_cancel, y_pred_cancel)
cancel_f1 = f1_score(y_test_cancel, y_pred_cancel)
cancel_auc_roc = roc_auc_score(y_test_cancel, y_pred_cancel)

# Initialize global variables for storing the predictions
delay_pred = None
cancel_pred = None

def predict():
    global delay_pred, cancel_pred
    
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
    
    # Display results
    delay_result.set(f"Delayed: {'Yes' if delay_pred == 1 else 'No'}")
    cancel_result.set(f"Cancelled: {'Yes' if cancel_pred == 1 else 'No'}")
    
    # Display evaluation metrics
    delay_metrics.set(f"Accuracy: {delay_accuracy:.2f}, Precision: {delay_precision:.2f}, "
                      f"Recall: {delay_recall:.2f}, F1: {delay_f1:.2f}, AUC-ROC: {delay_auc_roc:.2f}")
    cancel_metrics.set(f"Accuracy: {cancel_accuracy:.2f}, Precision: {cancel_precision:.2f}, "
                       f"Recall: {cancel_recall:.2f}, F1: {cancel_f1:.2f}, AUC-ROC: {cancel_auc_roc:.2f}")
def visualize():
    global delay_pred, cancel_pred
    
    if delay_pred is None or cancel_pred is None:
        print("Please make a prediction first.")
        return
    
    # Visualize the prediction results
    plt.figure(figsize=(6, 4))
    categories = ['Delay', 'Cancellation']
    values = [delay_pred, cancel_pred]
    
    plt.bar(categories, values, color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel('Prediction')
    plt.title('Prediction Results')
    plt.xticks([0, 1], ['Delay', 'Cancellation'])
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.show()

def plot_time_series_button():
    # Assuming you have time-series data
    # Example: Using y_test_delay and y_pred_delay with a dummy time index
    time_series_data = range(len(y_test_delay))
    plot_time_series(y_test_delay, y_pred_delay, time_series_data)

def plot_time_series(actual, predicted, time):
    plt.figure(figsize=(10, 6))
    plt.plot(time, actual, label='Actual', color='blue')
    plt.plot(time, predicted, label='Predicted', color='orange', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time-Series Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_plot_button():
    scatter_plot_with_regression_line(y_test_delay, y_pred_delay)

def scatter_plot_with_regression_line(actual, predicted):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=predicted, y=actual, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Scatter Plot with Regression Line')
    plt.grid(True)
    plt.show()

def residual_plot_button():
    residual_plot(y_test_delay, y_pred_delay)

def residual_plot(actual, predicted):
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix_button():
    plot_confusion_matrix(y_test_delay, y_pred_delay)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Create the main window
root = tk.Tk()
root.title("Flight Delay and Cancellation Prediction")

# Create and place the widgets
ttk.Label(root, text="Airline").grid(row=0, column=0)
airline_entry = ttk.Entry(root)
airline_entry.grid(row=0, column=1)

ttk.Label(root, text="Origin").grid(row=1, column=0)
origin_entry = ttk.Entry(root)
origin_entry.grid(row=1, column=1)

ttk.Label(root, text="Destination").grid(row=2, column=0)
dest_entry = ttk.Entry(root)
dest_entry.grid(row=2, column=1)

ttk.Label(root, text="CRS Departure Time").grid(row=3, column=0)
crs_dep_time_entry = ttk.Entry(root)
crs_dep_time_entry.grid(row=3, column=1)

ttk.Label(root, text="Distance").grid(row=4, column=0)
distance_entry = ttk.Entry(root)
distance_entry.grid(row=4, column=1)

predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.grid(row=5, column=0, columnspan=2)

# Labels for displaying results
delay_result = tk.StringVar()
ttk.Label(root, textvariable=delay_result).grid(row=6, column=0, columnspan=2)

cancel_result = tk.StringVar()
ttk.Label(root, textvariable=cancel_result).grid(row=7, column=0, columnspan=2)

# Labels for evaluation metrics
delay_metrics = tk.StringVar()
ttk.Label(root, textvariable=delay_metrics).grid(row=8, column=0, columnspan=2)

cancel_metrics = tk.StringVar()
ttk.Label(root, textvariable=cancel_metrics).grid(row=9, column=0, columnspan=2)

# Add buttons for different visualizations
ttk.Button(root, text="Visualize Prediction Results", command=visualize).grid(row=6, column=0, columnspan=2)
ttk.Button(root, text="Time-Series Plot", command=plot_time_series_button).grid(row=7, column=0, columnspan=2)
ttk.Button(root, text="Scatter Plot with Regression", command=scatter_plot_button).grid(row=8, column=0, columnspan=2)
ttk.Button(root, text="Residual Plot", command=residual_plot_button).grid(row=9, column=0, columnspan=2)
ttk.Button(root, text="Confusion Matrix", command=plot_confusion_matrix_button).grid(row=10, column=0, columnspan=2)

delay_result = tk.StringVar()
ttk.Label(root, textvariable=delay_result).grid(row=11, column=0, columnspan=2)

cancel_result = tk.StringVar()
ttk.Label(root, textvariable=cancel_result).grid(row=12, column=0, columnspan=2)

# Run the application
root.mainloop()
