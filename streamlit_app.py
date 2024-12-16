import streamlit as st
import pickle
import inspect
from sklearn.ensemble import RandomForestRegressor
from keras.models import load_model  # Assuming you're using Keras for LSTM
import numpy as np

# Fungsi untuk memuat model dari file pickle
def load_model_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk menampilkan parameter model
def display_model_parameters(model):
    st.write("Model Parameters:")
    # Menampilkan parameter menggunakan inspect
    model_params = inspect.signature(model.__init__).parameters
    for param in model_params:
        st.write(f"{param}: {model_params[param]}")

# Fungsi untuk memuat data dan memberikan prediksi (misal data acak)
def make_predictions(model_lstm, model_rf, input_data):
    # Asumsi model hybrid mengambil input data dan memberikan prediksi
    lstm_pred = model_lstm.predict(input_data)  # Prediksi menggunakan LSTM
    rf_pred = model_rf.predict(input_data)      # Prediksi menggunakan RandomForest
    hybrid_pred = (lstm_pred + rf_pred) / 2     # Gabungkan hasil prediksi (misalnya rata-rata)
    return hybrid_pred

# Muat model LSTM dan Random Forest
model_lstm = load_model_from_pickle('model_lstm.pkl')  # Sesuaikan dengan path file model LSTM Anda
model_rf = load_model_from_pickle('model_rf.pkl')      # Sesuaikan dengan path file model RF Anda

# Menampilkan parameter model LSTM dan Random Forest
st.header("LSTM Model Parameters")
display_model_parameters(model_lstm)
st.header("Random Forest Model Parameters")
display_model_parameters(model_rf)

# Data input (misalnya data acak atau data yang sudah disiapkan)
input_data = np.random.random((1, 10))  # Misal input 10 fitur, sesuaikan dengan input model

# Menampilkan prediksi hasil model
hybrid_prediction = make_predictions(model_lstm, model_rf, input_data)

st.header("Hybrid Model Prediction")
st.write("Prediksi Hybrid Model:", hybrid_prediction)
