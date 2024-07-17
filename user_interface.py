import streamlit as st
import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# Set page config
st.set_page_config(
    page_title="Laptop Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply some custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        .main {
            background-color: black;
        }
        .title {
            color: white;
            text-align: center;
            font-family: 'Roboto', sans-serif;
        }
        .section-header {
            color: white;
            font-size: 20px;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Import the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.markdown("<h1 class='title'>Laptop Price Predictor</h1>", unsafe_allow_html=True)

# Input fields
st.markdown("<h2 class='section-header'>Select Laptop Configuration</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of the Laptop (kg)')
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS', ['No', 'Yes'])

with col2:
    screen_size = st.number_input('Screen Size (inches)')
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())
    os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    try:
        # Prepare the query
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(float, resolution.split('x'))
        
        if screen_size <= 0:
            st.error("Screen size must be greater than zero")
        else:
            ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
           
            # Transform the query data
            step1_transformer = pipe.named_steps['step1']
            transformed_query = step1_transformer.transform(query)

            # Ensure the transformed query is of the expected type and shape
            transformed_query = np.array(transformed_query, dtype=np.float32)

            # Predict the price
            predicted_price = pipe.named_steps['step2'].predict(transformed_query)
            st.success("The predicted price of this configuration is ₹ " + str(int(np.exp(predicted_price[0]))))
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Query shape:", query.shape)
        st.write("Query content:", query)
