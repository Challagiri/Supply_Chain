import pandas as pd
import streamlit as st
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet



# Load the model
pickle_in = open('Model.pkl', 'rb')
model = pickle.load(pickle_in)




# Define the welcome message function
def welcome():
    return 'Welcome all'

# Define the prediction function

def prediction(Date):
    #<p>Input data to get real-time predictions:</p>
    # Convert the input date into the format expected by Prophet
    df = pd.DataFrame({'ds': [pd.to_datetime(Date)]})
    
    try:
        # Make the prediction
        prediction = model.predict(df)
        return prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper',]]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Define the main function for the Streamlit app
def main():
    # Set the title of the web ap
    st.set_page_config(page_title="Supply Chain", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")


    # Set the front end elements of the web page
    html_temp = """

    <style>
    h1 {
      font-family: 'Times New Roman', serif;
      text-align:center;
    }
    h2 {
      color: #007bff;
      font-family: 'Georgia', serif;
    }
    form {
      background-color: #e9ecef;
      padding: 20px;
      border-radius: 8px;
    }
   
    input[type="Show Graphs "]:hover {
      background-color: #218838;
  </style>
    <div style="background-color:DodgerBlue;padding:10px">
    <h2 style="color:white;text-align:center;">  \ Supply Chain Forecast /</h2>
    </div>
    """
    
    # Display the front end aspects
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write(" ")
    st.subheader("Model's Performance ")
    st.write("After Building all the models and forecast the sales for a period of time, Team compared the Models performance and error. It turns out to be the prophet model is performing well for this data and gives the better and reliable results. Team also tabulated the comparison results below.")
    model_data = {
    "Model": ["ARIMA", "SARIMAX","Exponential smoothing", "LSTM", "PROPHET"],
    "RMSE": [145.98,145.57,145.79,55.67,3.84],
    "R2 Score":[-2.22,0.005,0.002,0.16,0.68]
    }
    df = pd.DataFrame(model_data)
    df.index= df.index+1

    if st.button("Model's "):
        st.table(df)

    st.write("After Comparing all the models the prophet model is better than compared to others. The Forecasted Sales Graphs are shown in below. ")
    if st.button("Show Graphs "):
        
            future = model.make_future_dataframe(periods=1000)
            forecast = model.predict(future)

            st.title(":blue[Model Forecast ]")
            st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

   
    
    # Create a text input for the date
    st.subheader("Interact with the Model")
    st.write(" ")
    Date = st.date_input(":red[ Choose the Date  : ]",'today',)

    if Date:
        st.write("The Selected date is :   ", Date)
    
    # Initialize the result variable
    result = ""
    
    # When the 'Predict' button is clicked, call the prediction function

    if st.button("Predict"):
        result = prediction(Date)
        if result is not None:
            st.success(f'The Forecast is : ')
            st.write(result)
        else:
            st.error("Please check the date format and try again.")

# Run the main function
if __name__=='__main__':
    main()
