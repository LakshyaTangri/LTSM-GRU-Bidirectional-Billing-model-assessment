# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:07.374412Z","iopub.execute_input":"2025-02-01T08:02:07.374758Z","iopub.status.idle":"2025-02-01T08:02:07.737366Z","shell.execute_reply.started":"2025-02-01T08:02:07.374729Z","shell.execute_reply":"2025-02-01T08:02:07.736424Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ##### Each column provides specific information about the patient, their admission, and the healthcare services provided, making this dataset suitable for various data analysis and modeling tasks in the healthcare domain. Here's a brief explanation of each column in the dataset -
# 
# 1.Name: This column represents the name of the patient associated with the healthcare record.
# 
# 2.Age: The age of the patient at the time of admission, expressed in years.
# 
# 3.Gender: Indicates the gender of the patient, either "Male" or "Female."
# 
# 4.Blood Type: The patient's blood type, which can be one of the common blood types (e.g., "A+", "O-", etc.).
# 
# 5,Medical Condition: This column specifies the primary medical condition or diagnosis associated with the patient, such as "Diabetes," "Hypertension," "Asthma," and more.
# 
# 6.Date of Admission: The date on which the patient was admitted to the healthcare facility.
# 
# 7.Doctor: The name of the doctor responsible for the patient's care during their admission.
# 
# 8.Hospital: Identifies the healthcare facility or hospital where the patient was admitted.
# 
# 9.Insurance Provider: This column indicates the patient's insurance provider, which can be one of several options, including "Aetna," "Blue Cross," "Cigna," "UnitedHealthcare," and "Medicare."
# 
# 10.Billing Amount: The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number.
# 
# 11.Room Number: The room number where the patient was accommodated during their admission.
# 
# 12.Admission Type: Specifies the type of admission, which can be "Emergency," "Elective," or "Urgent," reflecting the circumstances of the admission.
# 
# 13.Discharge Date: The date on which the patient was discharged from the healthcare facility, based on the admission date and a random number of days within a realistic range.
# 
# 14.Medication: Identifies a medication prescribed or administered to the patient during their admission. Examples include "Aspirin," "Ibuprofen," "Penicillin," "Paracetamol," and "Lipitor."
# 
# 15.Test Results: Describes the results of a medical test conducted during the patient's admission. Possible values include "Normal," "Abnormal," or "Inconclusive," indicating the outcome of the test.

# %% [markdown]
# #### I'll start by analyzing your dataset to understand its structure and contents. Then, I'll perform data preprocessing, feature engineering, data visualization, and model selection.
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:07.738572Z","iopub.execute_input":"2025-02-01T08:02:07.738987Z","iopub.status.idle":"2025-02-01T08:02:08.538789Z","shell.execute_reply.started":"2025-02-01T08:02:07.738952Z","shell.execute_reply":"2025-02-01T08:02:08.538033Z"}}
import seaborn as sns
import matplotlib.pyplot as plt 


# %% [markdown]
# # Dataset Summary
# ####  Dataset contains 55,500 records with 15 columns related to healthcare admissions. Here's a breakdown:
# 
# #### Key Columns & Data Types
# Date Columns:
# Date of Admission
# Discharge Date
# Numerical Columns:
# Age, Billing Amount, Room Number
# 
# 
# #### Categorical Columns:
# 
# Gender, Blood Type, Medical Condition, Doctor, Hospital, Insurance Provider, Admission Type, Medication, Test Results

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:08.540656Z","iopub.execute_input":"2025-02-01T08:02:08.541169Z","iopub.status.idle":"2025-02-01T08:02:08.884145Z","shell.execute_reply.started":"2025-02-01T08:02:08.541133Z","shell.execute_reply":"2025-02-01T08:02:08.883178Z"}}
data=pd.read_csv('/kaggle/input/healthcare-dataset/healthcare_dataset.csv')
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:08.885836Z","iopub.execute_input":"2025-02-01T08:02:08.886199Z","iopub.status.idle":"2025-02-01T08:02:08.891262Z","shell.execute_reply.started":"2025-02-01T08:02:08.886172Z","shell.execute_reply":"2025-02-01T08:02:08.890511Z"}}
data.shape

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:08.892181Z","iopub.execute_input":"2025-02-01T08:02:08.892467Z","iopub.status.idle":"2025-02-01T08:02:08.961578Z","shell.execute_reply.started":"2025-02-01T08:02:08.89243Z","shell.execute_reply":"2025-02-01T08:02:08.960597Z"}}
data.info()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:08.962554Z","iopub.execute_input":"2025-02-01T08:02:08.962866Z","iopub.status.idle":"2025-02-01T08:02:09.000611Z","shell.execute_reply.started":"2025-02-01T08:02:08.962832Z","shell.execute_reply":"2025-02-01T08:02:08.999757Z"}}
data.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.001761Z","iopub.execute_input":"2025-02-01T08:02:09.002118Z","iopub.status.idle":"2025-02-01T08:02:09.016163Z","shell.execute_reply.started":"2025-02-01T08:02:09.002067Z","shell.execute_reply":"2025-02-01T08:02:09.0154Z"}}
data['Test Results'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.019058Z","iopub.execute_input":"2025-02-01T08:02:09.01933Z","iopub.status.idle":"2025-02-01T08:02:09.030381Z","shell.execute_reply.started":"2025-02-01T08:02:09.019309Z","shell.execute_reply":"2025-02-01T08:02:09.029648Z"}}
data['Medication'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.032404Z","iopub.execute_input":"2025-02-01T08:02:09.032628Z","iopub.status.idle":"2025-02-01T08:02:09.048931Z","shell.execute_reply.started":"2025-02-01T08:02:09.032607Z","shell.execute_reply":"2025-02-01T08:02:09.048183Z"}}
data['Admission Type'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.049847Z","iopub.execute_input":"2025-02-01T08:02:09.05019Z","iopub.status.idle":"2025-02-01T08:02:09.065845Z","shell.execute_reply.started":"2025-02-01T08:02:09.050157Z","shell.execute_reply":"2025-02-01T08:02:09.065054Z"}}
data['Insurance Provider'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.066581Z","iopub.execute_input":"2025-02-01T08:02:09.066808Z","iopub.status.idle":"2025-02-01T08:02:09.081666Z","shell.execute_reply.started":"2025-02-01T08:02:09.066773Z","shell.execute_reply":"2025-02-01T08:02:09.080851Z"}}
data['Age'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.082658Z","iopub.execute_input":"2025-02-01T08:02:09.08294Z","iopub.status.idle":"2025-02-01T08:02:09.099427Z","shell.execute_reply.started":"2025-02-01T08:02:09.082916Z","shell.execute_reply":"2025-02-01T08:02:09.098507Z"}}
data['Gender'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.100498Z","iopub.execute_input":"2025-02-01T08:02:09.100808Z","iopub.status.idle":"2025-02-01T08:02:09.1135Z","shell.execute_reply.started":"2025-02-01T08:02:09.100783Z","shell.execute_reply":"2025-02-01T08:02:09.112736Z"}}
data['Blood Type'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.114535Z","iopub.execute_input":"2025-02-01T08:02:09.11484Z","iopub.status.idle":"2025-02-01T08:02:09.130678Z","shell.execute_reply.started":"2025-02-01T08:02:09.11481Z","shell.execute_reply":"2025-02-01T08:02:09.129899Z"}}
data['Medical Condition'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.131436Z","iopub.execute_input":"2025-02-01T08:02:09.131692Z","iopub.status.idle":"2025-02-01T08:02:09.14711Z","shell.execute_reply.started":"2025-02-01T08:02:09.13167Z","shell.execute_reply":"2025-02-01T08:02:09.146335Z"}}
data['Date of Admission'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.148053Z","iopub.execute_input":"2025-02-01T08:02:09.148395Z","iopub.status.idle":"2025-02-01T08:02:09.168681Z","shell.execute_reply.started":"2025-02-01T08:02:09.148362Z","shell.execute_reply":"2025-02-01T08:02:09.16792Z"}}
data['Doctor'].nunique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.169505Z","iopub.execute_input":"2025-02-01T08:02:09.169721Z","iopub.status.idle":"2025-02-01T08:02:09.178Z","shell.execute_reply.started":"2025-02-01T08:02:09.169702Z","shell.execute_reply":"2025-02-01T08:02:09.177137Z"}}
data['Insurance Provider'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.178761Z","iopub.execute_input":"2025-02-01T08:02:09.179099Z","iopub.status.idle":"2025-02-01T08:02:09.192497Z","shell.execute_reply.started":"2025-02-01T08:02:09.179034Z","shell.execute_reply":"2025-02-01T08:02:09.191696Z"}}
data['Admission Type'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.193395Z","iopub.execute_input":"2025-02-01T08:02:09.193677Z","iopub.status.idle":"2025-02-01T08:02:09.223578Z","shell.execute_reply.started":"2025-02-01T08:02:09.193648Z","shell.execute_reply":"2025-02-01T08:02:09.22255Z"}}
data.describe()

# %% [markdown]
# ## Next Steps
# 
# ✅ Data Visualization – Understand trends in medical conditions, admissions, and billing amounts.
# 
# ✅ Data Preprocessing – Handle missing values, convert dates, and clean categorical data.
# 
# ✅ Feature Engineering – Create useful features from dates, encode categorical data, and normalize numerical values.
# 
# 
# ✅ Model Selection – Use LSTM/RNN for time-based prediction tasks (e.g., hospital stay duration, cost forecasting).

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.224591Z","iopub.execute_input":"2025-02-01T08:02:09.22493Z","iopub.status.idle":"2025-02-01T08:02:09.230684Z","shell.execute_reply.started":"2025-02-01T08:02:09.224896Z","shell.execute_reply":"2025-02-01T08:02:09.229588Z"}}
data.columns

# %% [markdown]
# # ✅ Data Visualization – Understand trends in medical conditions, admissions, and billing amounts.
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:09.231783Z","iopub.execute_input":"2025-02-01T08:02:09.232168Z","iopub.status.idle":"2025-02-01T08:02:11.271552Z","shell.execute_reply.started":"2025-02-01T08:02:09.232139Z","shell.execute_reply":"2025-02-01T08:02:11.270535Z"}}
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# 1️⃣ Age Distribution
plt.subplot(2, 3, 1)
sns.histplot(data["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution")

# 2️⃣ Gender Count
plt.subplot(2, 3, 2)
sns.countplot(x=data["Gender"], palette="coolwarm")
plt.title("Gender Distribution")

# 3️⃣ Blood Type Count
plt.subplot(2, 3, 3)
sns.countplot(x=data["Blood Type"], palette="muted")
plt.title("Blood Type Distribution")

# 4️⃣ Medical Condition Count
plt.figure(figsize=(10, 5))
sns.countplot(y=data["Medical Condition"], palette="pastel", order=data["Medical Condition"].value_counts().index)
plt.title("Medical Condition Distribution")
plt.xlabel("Count")

# 5️⃣ Billing Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data["Billing Amount"], bins=30, kde=True, color="green")
plt.title("Billing Amount Distribution")
plt.xlabel("Billing Amount")

# 6️⃣ Admissions Over Time
plt.figure(figsize=(10, 5))
data["Date of Admission"].value_counts().sort_index().plot()
plt.title("Admissions Over Time")
plt.xlabel("Date of Admission")
plt.ylabel("Number of Patients")

plt.show()


# %% [markdown]
# ### Data Preprocessing  below Completed ✅
# 
# Converted Date Columns (Date of Admission, Discharge Date) to datetime format.
# 
# Standardized Text Data (Gender, Medical Condition, Admission Type, Test Results).
# 
# Created New Feature: Length of Stay (Days between admission & discharge).
# 
# No Missing Values detected.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.272567Z","iopub.execute_input":"2025-02-01T08:02:11.272885Z","iopub.status.idle":"2025-02-01T08:02:11.469554Z","shell.execute_reply.started":"2025-02-01T08:02:11.272851Z","shell.execute_reply":"2025-02-01T08:02:11.468727Z"}}
# Convert date columns to datetime format
data["Date of Admission"] = pd.to_datetime(data["Date of Admission"], errors="coerce")
data["Discharge Date"] = pd.to_datetime(data["Discharge Date"], errors="coerce")

# Check for missing values
missing_values = data.isnull().sum()

# Standardize categorical text data (e.g., names, medical conditions)
data["Gender"] = data["Gender"].str.strip().str.capitalize()
data["Medical Condition"] = data["Medical Condition"].str.strip().str.title()
data["Admission Type"] = data["Admission Type"].str.strip().str.title()
data["Test Results"] = data["Test Results"].str.strip().str.title()

# Derive new features from dates (Length of Stay)
data["Length of Stay"] = (data["Discharge Date"] - data["Date of Admission"]).dt.days

# Display processed dataset summary
data.info(), missing_values, data.head()


# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.472727Z","iopub.execute_input":"2025-02-01T08:02:11.472987Z","iopub.status.idle":"2025-02-01T08:02:11.478269Z","shell.execute_reply.started":"2025-02-01T08:02:11.472963Z","shell.execute_reply":"2025-02-01T08:02:11.477371Z"}}
data.shape

# %% [markdown]
# # Next Steps
# 
# ✅ Feature Engineering – Encode categorical data & create additional features.
# 
# ✅ Data Visualization – Explore trends in admission types, billing, length of stay.
# 
# ✅ Model Selection – Identify suitable models (LSTM, RNN) for prediction.

# %% [markdown]
# 
# # ✅ Feature Engineering – Encode categorical data & create additional features.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.479342Z","iopub.execute_input":"2025-02-01T08:02:11.479634Z","iopub.status.idle":"2025-02-01T08:02:11.496849Z","shell.execute_reply.started":"2025-02-01T08:02:11.479603Z","shell.execute_reply":"2025-02-01T08:02:11.495894Z"}}
data['Gender'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.497807Z","iopub.execute_input":"2025-02-01T08:02:11.498105Z","iopub.status.idle":"2025-02-01T08:02:11.588906Z","shell.execute_reply.started":"2025-02-01T08:02:11.498049Z","shell.execute_reply":"2025-02-01T08:02:11.588183Z"}}
# Categorical columns to one-hot encode
categorical_cols = ["Gender", "Blood Type", "Medical Condition", "Admission Type", "Test Results", "Medication"]

# Apply one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # Drop first column to avoid dummy variable trap

# Display updated dataset
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.589793Z","iopub.execute_input":"2025-02-01T08:02:11.590091Z","iopub.status.idle":"2025-02-01T08:02:11.727288Z","shell.execute_reply.started":"2025-02-01T08:02:11.590047Z","shell.execute_reply":"2025-02-01T08:02:11.726526Z"}}
# Convert 'Date of Admission' to datetime (if not already done)
data["Date of Admission"] = pd.to_datetime(data["Date of Admission"])

# Create new time-based features
data["Day of Week"] = data["Date of Admission"].dt.dayofweek  # Monday=0, Sunday=6
data["Month"] = data["Date of Admission"].dt.month  # Month as number (1-12)
data["Weekend"] = data["Day of Week"].apply(lambda x: 1 if x >= 5 else 0)  # 1 if Saturday/Sunday

# Drop original date column (not useful for LSTM directly)
data.drop(columns=["Date of Admission"], inplace=True)

# Display the updated dataframe
data.head()


# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.728084Z","iopub.execute_input":"2025-02-01T08:02:11.728385Z","iopub.status.idle":"2025-02-01T08:02:11.810217Z","shell.execute_reply.started":"2025-02-01T08:02:11.72836Z","shell.execute_reply":"2025-02-01T08:02:11.809516Z"}}
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[["Age", "Billing Amount", "Room Number", "Length of Stay"]] = scaler.fit_transform(
    data[["Age", "Billing Amount", "Room Number", "Length of Stay"]]
)


# %% [markdown]
# ### Above Feature Engineering Completed ✅
# 
# Encoded Categorical Variables (Gender, Blood Type, Medical Condition, etc.) using Label Encoding.
# 
# Scaled Numerical Features (Age, Billing Amount, Room Number, Length of Stay) using StandardScaler.
# 
# 
# 

# %% [markdown]
# ### Next Steps
# 
# 
# ✅ Model Selection – Build an LSTM/RNN-based model for predicting Length of Stay or Billing Amount.
# 
# I'll now create some visualizations to understand the data better. 

# %% [markdown]
# # 1️⃣ Problem Definition
# We can use LSTM (Long Short-Term Memory) and RNN (Recurrent Neural Networks) to predict:
# 
# ✔ Length of Stay – Based on patient attributes & past hospital records.
# 
# ✔ Billing Amount – Forecast hospital charges using past trends.
# 
# 
# ### Model Options
# #### 📌 Traditional Models (Baselines for Comparison)
# 
# Linear Regression – Simple and interpretable.
# 
# Random Forest/XGBoost – Works well with tabular data.
# 
# ARIMA/SARIMA – Common time series forecasting methods.
# 
# #### 📌 Deep Learning Models
# 
# Vanilla RNN – Captures sequential patterns but struggles with long-term dependencies.
# 
# LSTM (Long Short-Term Memory) – Good for long-term dependencies in time series.
# 
# Bidirectional LSTM – Considers both past & future trends for better accuracy.
# 
# GRU (Gated Recurrent Unit) – Lighter than LSTM but still powerful for sequence prediction.
# 
# ### Best Model Choice 
# 
# 
# Since Our dataset has structured tabular data and time dependencies:
# 
# ✔ LSTM or GRU – Best suited for predicting Length of Stay or Billing Amount.
# 
# ✔ Compare with XGBoost – As a strong non-deep learning baseline.

# %% [markdown]
# 2️⃣ Data Preparation for LSTM – Convert time-series data into sequences.
# 
# 3️⃣ Build and Train LSTM Model – Predict Length of Stay.
# 
# 4️⃣ Evaluate Model Performance – Compare LSTM with traditional models.
# 
# 5️⃣ Predictions and Insights – Visualize model predictions.

# %% [markdown]
# To use LSTM for predicting Length of Stay, we need to:
# 
# Convert the dataset into sequential time-series data.
# 
# Select relevant numerical features.
# 
# Reshape the data for LSTM input format.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:11.81111Z","iopub.execute_input":"2025-02-01T08:02:11.811373Z","iopub.status.idle":"2025-02-01T08:02:25.337639Z","shell.execute_reply.started":"2025-02-01T08:02:11.811351Z","shell.execute_reply":"2025-02-01T08:02:25.336747Z"}}
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Select features & target
features = ["Age", "Billing Amount", "Room Number"]
target = "Length of Stay"

# Convert data into NumPy arrays
X = data[features].values
y = data[target].values

# Reshape data for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# ##### Build & Train Bidirectional LSTM Model
# 
# Now, we’ll define an Bidirectional LSTM model to predict Length of Stay.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:25.338517Z","iopub.execute_input":"2025-02-01T08:02:25.339183Z","iopub.status.idle":"2025-02-01T08:02:51.718735Z","shell.execute_reply.started":"2025-02-01T08:02:25.339156Z","shell.execute_reply":"2025-02-01T08:02:51.71794Z"}}
# Define a deeper LSTM model
model = Sequential([
    Bidirectional(LSTM(100, activation="relu", return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history=model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])






# %% [markdown]
#  #### Evaluate Model Performance
#  
# Plot training vs. validation loss to see how well the model is learning.
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:51.719762Z","iopub.execute_input":"2025-02-01T08:02:51.720029Z","iopub.status.idle":"2025-02-01T08:02:51.997367Z","shell.execute_reply.started":"2025-02-01T08:02:51.720006Z","shell.execute_reply":"2025-02-01T08:02:51.996443Z"}}
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("LSTM Model Training Loss")
plt.legend()
plt.show()


# %% [markdown]
# ### Make Predictions & Visualize
# 
# We’ll now predict Length of Stay for the test data and compare with actual values.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-01T08:02:51.998296Z","iopub.execute_input":"2025-02-01T08:02:51.998639Z","iopub.status.idle":"2025-02-01T08:02:54.793952Z","shell.execute_reply.started":"2025-02-01T08:02:51.998614Z","shell.execute_reply":"2025-02-01T08:02:54.793061Z"}}
# Make predictions
y_pred = model.predict(X_test)

# Visualize predictions vs. actual values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.xlabel("Actual Length of Stay")
plt.ylabel("Predicted Length of Stay")
plt.title("LSTM Predictions vs Actual")
plt.show()
