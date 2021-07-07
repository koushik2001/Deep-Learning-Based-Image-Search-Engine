import streamlit as st
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.models import Model


def Feature_Extractor(img):
    base_model = VGG16(weights="imagenet")
    model = Model(inputs=base_model.input,outputs=base_model.get_layer("fc1").output)
    
    img = img.resize((224,224)).convert("RGB")
    
    x = image.img_to_array(img)
    
    x = np.expand_dims(x,axis=0)
    
    x = preprocess_input(x)
    
    feature = model.predict(x)[0]
    
    return feature/np.linalg.norm(feature)


import os


def image_features():
    for img_path in sorted(os.listdir(r"C:\Users\saiko\Desktop\ISE-NEW\Images")):

        final_path = r"C:\Users\saiko\Desktop\ISE-NEW\Images" + "\\" + img_path

        features = Feature_Extractor(img = Image.open(final_path))

        feature_path = r"C:\Users\saiko\Desktop\ISE-NEW\features" + "\\" +os.path.splitext(img_path)[0] + ".npy"

        np.save(feature_path,features)

#image_features()







st.sidebar.title("About Project")
st.sidebar.info("This is a application demonstrates Deep Learning Based Image Search Engine.")
st.header("Deep-Learning Based Image Search Engine")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    
    image1 = Image.open(uploaded_file)

   
    with st.spinner('Loading Image....'):
        time.sleep(3)

    st.image(image1, caption='Input Image.', use_column_width=True)

    
    x = st.slider('Top k results')
    
    features = []
    img_paths = []

    for feature_path in os.listdir(r"C:\Users\saiko\Desktop\ISE-NEW\features"):
        features.append(np.load(r"C:\Users\saiko\Desktop\ISE-NEW\features" + "\\" + feature_path))
        img_paths.append(r"C:\Users\saiko\Desktop\ISE-NEW\Images" + "\\" + os.path.splitext(feature_path)[0] + ".jpeg")
    features = np.array(features)
    
    query = Feature_Extractor(image1)
    dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:x]  # Top 30 results
    scores = [[dists[id], img_paths[id]] for id in ids]
    for i in range(0,len(scores)):
        print(scores[i][1])
        img = Image.open(scores[i][1])
        st.image(img)







