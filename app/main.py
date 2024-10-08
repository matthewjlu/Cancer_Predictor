import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from io import StringIO



def main():
    st.set_page_config(
        page_title = "Cancer Predictor App",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    page = st.sidebar.selectbox("Select a page", ["Landing Page", "Cancer Data Analysis", "Upload Lab Data"])

    if page == "Landing Page":
        with st.container():
            #use html and css to make title
            st.markdown("""
                <style>
                .title {
                    font-size:70px !important;
                    font-family: 'Georgia', serif;
                    background-image: linear-gradient(to bottom, red 40%, pink 70%, white 100%);
                    -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent;
                }
                .subtitle {
                    font-size:20px !important;
                    font-family: 'Georgia', serif;
                    color: white;  
                }
                </style>
                """, unsafe_allow_html=True)

            st.markdown('<p class="title">Cancer Prediction Application</p>', 
                unsafe_allow_html=True)
            
            st.markdown(
                '''<p class="subtitle"> Please connect this app to a lab! 
                This is not medical advice. You have the option to either input 
                data manually or by uploading a csv file. </p>''', 
                unsafe_allow_html=True)
            
        
        gif_path = "misc/graphics.gif"
        st.image(gif_path, use_column_width=True)
            

    elif page == "Cancer Data Analysis":
        with st.container():
            #creates h1 header
            st.title("Breast Cancer Predictor")
            #creates paragraph element
            st.write(" You can use text box to update the parameters and the"
                    " predictions will all be made through a logistic regression"
                    " model.")
        

        #creates two columns where the first column is 4 times as big as the second
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1: 
            input_data = add_sidebar()

        with col2:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)

        with col3:
            add_predictions(input_data)


    else:
        with st.container():
            #creates h1 header
            st.title("Lab Data")
            #creates paragraph element
            st.write("You can upload lab data here! It will output a csv file" 
                    " that tells you if each diagnosis is benign or malicious. ")
            
        uploaded_files = st.file_uploader(
            "Upload Cancer Data", accept_multiple_files=True, type = ["csv"]
            )
            
        if uploaded_files:
            for uploaded_file in uploaded_files:
                #make a df out of the uploaded file
                df = pd.read_csv(uploaded_file)
                #show the data frame
                st.write(f"Data from {uploaded_file.name}:")
                st.write(df)
        else:
            st.write("Please upload a CSV file.")



def add_sidebar():

    with open("model/pickle_folder/slider_labels.pkl", "rb") as slider_file:
        slider_labels = pickle.load(slider_file)
    with open("model/pickle_folder/max_values.pkl", "rb") as max_file:
        max_values = pickle.load(max_file)
    with open("model/pickle_folder/mean_values.pkl", "rb") as mean_file:
        mean_values = pickle.load(mean_file)
    
    #creating pair where key = column name in data, value = text value
    input_dict = {}
    
    for label, key in slider_labels:
        value = float(st.text_input(label, mean_values[key]))
        input_dict[key] = value

    # for label, key in slider_labels:
    #     input_dict[key] = st.sidebar.slider(
    #         label,
    #         min_value = 0.0,
    #         max_value = max_values[key],
    #         value = mean_values[key]
    #     )

    return input_dict

def scaled_values(input_dict):
    with open("model/pickle_folder/features.pkl", "rb") as feature_file:
        X = pickle.load(feature_file)
    
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()

        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = scaled_values(input_data)

    categories = ["Radius", "Texture", "Perimeter", "Area", "Smoothness",
                  "Compactness", "Concavity", "Concave Points", "Symmetry",
                  "Fractal Dimension"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data["radius_mean"], input_data["texture_mean"], 
           input_data["perimeter_mean"], input_data["area_mean"],
           input_data["smoothness_mean"], input_data["compactness_mean"],
           input_data["concavity_mean"], input_data["concave points_mean"],
           input_data["symmetry_mean"], input_data["fractal_dimension_mean"]],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data["radius_se"], input_data["texture_se"], 
           input_data["perimeter_se"], input_data["area_se"],
           input_data["smoothness_se"], input_data["compactness_se"],
           input_data["concavity_se"], input_data["concave points_se"],
           input_data["symmetry_se"], input_data["fractal_dimension_se"]],
        theta=categories,
        fill='toself',
        name="Standard Error"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data["radius_worst"], input_data["texture_worst"], 
           input_data["perimeter_worst"], input_data["area_worst"],
           input_data["smoothness_worst"], input_data["compactness_worst"],
           input_data["concavity_worst"], input_data["concave points_worst"],
           input_data["symmetry_worst"], input_data["fractal_dimension_worst"]],
        theta=categories,
        fill='toself',
        name="Worst Value"
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_predictions(input_data):
    with open("model/pickle_folder/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("model/pickle_folder/scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

if __name__ == '__main__':
    main()