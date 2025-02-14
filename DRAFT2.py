import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
file_path = r'C:\Users\kayod\OneDrive\For Uni\Year 4 (Masters)\Final Year Project\The Program\Dataset\flights_sample_3m.csv'
df = pd.read_csv(file_path)

print(df)

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

# Logistic Regression with class_weight for handling imbalances
model_delay = LogisticRegression(max_iter=1000, class_weight='balanced')
model_cancel = LogisticRegression(max_iter=1000, class_weight='balanced')

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Model pipeline for delay and cancellation using Logistic Regression
model_pipeline_delay = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', model_delay)
])

model_pipeline_cancel = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', model_cancel)
])

# Model pipeline for delay using Decision Tree
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', decision_tree_model)
])

# Train the models
model_pipeline_delay.fit(X_train, y_train_delay)
model_pipeline_cancel.fit(X_train_cancel, y_train_cancel)
decision_tree_pipeline.fit(X_train, y_train_delay)  # Fit the decision tree model

# Generate predictions
y_pred_delay = model_pipeline_delay.predict(X_test)
y_pred_cancel = model_pipeline_cancel.predict(X_test_cancel)

# Classification reports
delay_report = classification_report(y_test_delay, y_pred_delay)
cancel_report = classification_report(y_test_cancel, y_pred_cancel)

# Get unique values for "Airline", "Origin", and "Destination" for the dropdowns
unique_airlines = df['AIRLINE'].unique().tolist()
unique_origins = df['ORIGIN'].unique().tolist()
unique_destinations = df['DEST'].unique().tolist()

# Initialize global variables for storing the predictions and metrics
delay_pred = None
cancel_pred = None
delay_metrics = None
cancel_metrics = None

def predict():
    global delay_pred, cancel_pred
    
    # Get input values from the dropdown boxes
    airline = airline_combobox.get()
    origin = origin_combobox.get()
    dest = dest_combobox.get()
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
    
    # Preprocess the input data using the same pipeline
    input_data_transformed = model_pipeline_delay['preprocessor'].transform(input_data)
    
    # Predict delay and cancellation
    delay_pred = model_pipeline_delay.predict(input_data)[0]
    cancel_pred = model_pipeline_cancel.predict(input_data)[0]
    
    # Calculate accuracy
    accuracy_delay = accuracy_score(y_test_delay, y_pred_delay)
    accuracy_cancel = accuracy_score(y_test_cancel, y_pred_cancel)

    # Display results
    delay_result.set(f"Delayed: {'Yes' if delay_pred == 1 else 'No'}")
    cancel_result.set(f"Cancelled: {'Yes' if cancel_pred == 1 else 'No'}")
    
    # Update metrics
    delay_metrics.set(f"Delay Model Accuracy: {accuracy_delay:.2f}")
    cancel_metrics.set(f"Cancellation Model Accuracy: {accuracy_cancel:.2f}")

    # Update text box with classification reports
    metrics_text.delete(1.0, tk.END)  # Clear previous content
    metrics_text.insert(tk.END, f"Delay Model Report:\n{delay_report}\n")
    metrics_text.insert(tk.END, f"Cancellation Model Report:\n{cancel_report}\n")

# Visualization functions
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

# Function to plot decision tree
def plot_decision_tree():
    plt.figure(figsize=(12, 8))
    plot_tree(decision_tree_model, 
              filled=True, 
              feature_names=preprocessor.transformers_[0][2] + preprocessor.transformers_[1][1].get_feature_names_out().tolist(),
              class_names=['No Delay', 'Delay'])
    plt.title('Decision Tree for Delay Prediction')
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Flight Delay and Cancellation Prediction")

# Create and place the widgets
ttk.Label(root, text="Airline").grid(row=0, column=0)
airline_combobox = ttk.Combobox(root, values=unique_airlines)
airline_combobox.grid(row=0, column=1)

ttk.Label(root, text="Origin").grid(row=1, column=0)
origin_combobox = ttk.Combobox(root, values=unique_origins)
origin_combobox.grid(row=1, column=1)

ttk.Label(root, text="Destination").grid(row=2, column=0)
dest_combobox = ttk.Combobox(root, values=unique_destinations)
dest_combobox.grid(row=2, column=1)

ttk.Label(root, text="CRS Departure Time").grid(row=3, column=0)
crs_dep_time_entry = ttk.Entry(root)
crs_dep_time_entry.grid(row=3, column=1)

ttk.Label(root, text="Distance").grid(row=4, column=0)
distance_entry = ttk.Entry(root)
distance_entry.grid(row=4, column=1)

# Prediction button
predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.grid(row=5, column=0, columnspan=2)

# Visualization buttons
ttk.Button(root, text="Visualize Prediction Results", command=visualize).grid(row=6, column=0, columnspan=2)
ttk.Button(root, text="Time-Series Plot", command=plot_time_series_button).grid(row=7, column=0, columnspan=2)
ttk.Button(root, text="Scatter Plot with Regression", command=scatter_plot_button).grid(row=8, column=0, columnspan=2)
ttk.Button(root, text="Residual Plot", command=residual_plot_button).grid(row=9, column=0, columnspan=2)
ttk.Button(root, text="Confusion Matrix", command=plot_confusion_matrix_button).grid(row=10, column=0, columnspan=2)
ttk.Button(root, text="Decision Tree", command=plot_decision_tree).grid(row=11, column=0, columnspan=2)

# StringVars for displaying results and metrics
delay_result = tk.StringVar()
cancel_result = tk.StringVar()
delay_metrics = tk.StringVar()
cancel_metrics = tk.StringVar()

# Labels for displaying the prediction results
ttk.Label(root, textvariable=delay_result).grid(row=12, column=0, columnspan=2)
ttk.Label(root, textvariable=cancel_result).grid(row=13, column=0, columnspan=2)

# Create text box for displaying the model metrics and reports
metrics_text = tk.Text(root, height=15, width=80)
metrics_text.grid(row=14, column=0, columnspan=2)

# Run the application
root.mainloop()
