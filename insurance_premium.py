import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")

st.title("Insurance Premium Price Prediction")

col1, col2 = st.columns(2)

# Load the dataset
df = pd.read_csv('insurance.csv')

# Prepare the data
numerical_columns = ['Height', 'Weight']  # Removed 'PremiumPrice'
categorical_columns = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants', 
                       'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily', 
                       'NumberOfMajorSurgeries']

# Standardize numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

X = df[numerical_columns + categorical_columns]
y = df['PremiumPrice']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Decision Tree Regressor with specified hyperparameters
dt_regressor = DecisionTreeRegressor(
    max_depth=7,
    min_samples_leaf=1,
    min_samples_split=10
)
dt_regressor.fit(X_train, y_train)

# Save the model using pickle
model_filename = 'decision_tree_regressor.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(dt_regressor, file)

# Load the model from the pickle file
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Create input fields for user data
yes_or_no_columns = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']

age = col2.selectbox("Enter Age:(18 to 71)", list(range(18, 71)))
# Initialize dictionary to store values for yes/no columns
yes_no_values = {}
for col in yes_or_no_columns:
    yes_no_values[col] = col1.selectbox(f"Select {col} from the dropdown: (No-0,Yes-1)", [0, 1])

num_of_surgeries = col2.selectbox("Enter the number of surgeries if any (0 to 3):", list(range(0, 3)))
height = col2.selectbox("Enter Height (cm): (120 to 220) ", list(range(120, 220)))
weight = col2.selectbox("Enter Weight (kg): (50 to 150)", list(range(50, 150)))

# Prepare input for prediction, in the order of X
input_data = pd.DataFrame({
    'Height': [height],
    'Weight': [weight],
    'Age': [age],
    'Diabetes': [yes_no_values['Diabetes']],
    'BloodPressureProblems': [yes_no_values['BloodPressureProblems']],
    'AnyTransplants': [yes_no_values['AnyTransplants']],
    'AnyChronicDiseases': [yes_no_values['AnyChronicDiseases']],
    'KnownAllergies': [yes_no_values['KnownAllergies']],
    'HistoryOfCancerInFamily': [yes_no_values['HistoryOfCancerInFamily']],
    'NumberOfMajorSurgeries': [num_of_surgeries]
})

# Scale numerical input data
input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

# Add button to make prediction
if st.button('Get Premium Price'):
    try:
        # Predict using the model
        predicted_premium = model.predict(input_data)
        # Display the predicted premium price in col2
        with col2:
            st.write(f"**Predicted Premium Price in currency:** {predicted_premium[0]:,.2f}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
