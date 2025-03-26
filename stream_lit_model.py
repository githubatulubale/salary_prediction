import streamlit as st
import pickle
import numpy as np

# Load the trained model from the local file system
model_path = "linear_regression_model.pkl"  # Ensure this file exists in your project directory
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'linear_regression_model.pkl' is in the project directory.")

# Set the title of the Streamlit app
st.title("Salary Prediction App")

# Add a brief description
st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

# Add input widget for user to enter years of experience
years_experience = st.number_input("Enter Years of Experience:", min_value=1.0, max_value=50.0, value=1.0, step=0.5)

# When the button is clicked, make predictions
if st.button("Predict Salary"):
    if 'model' in locals():  # Ensure model is loaded before predicting
        # Make a prediction using the trained model
        experience_input = np.array([[years_experience]])  # Convert input to a 2D array
        prediction = model.predict(experience_input)

        # Display the result
        st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")
    else:
        st.error("Model not loaded. Please check the file path.")

# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.")
