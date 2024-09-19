import streamlit as st
import tensorflow as tf
import numpy as np

#tensorflow model prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_model.h5")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict([input_arr])
    return np.argmax(predictions)#it will return the index of maximum element
    

#sidebar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",{"Prediction"})

#Prediction Pages
if(app_mode=="Prediction"):  
    st.header("Model Prediction")
    image_path="home_img.jpeg"
    st.image(image_path)
    test_image=st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True) #retrieving from uploader
#Predict Button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        #Reading Labels
        with open("labels.txt")as f:
            content=f.readlines()
        label=[]
        st.write(content)
        for i in content:
            label.append(i[:-1])
        st.success("Model is predicting it is a {} ".format(label[result_index]))