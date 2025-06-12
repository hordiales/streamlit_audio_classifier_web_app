#importar librerias
import streamlit as st
# import pickle
import pandas as pd
import numpy as np
import librosa
import json
import tensorflow as tf
import keras
from io import StringIO

def preprocess_input(example_filename, sample_rate=None, audio_duration=None):
    n_fft = 2048
    hop_length = 4*256

    # Open file with original sample rate
    example_audio, orig_sr = librosa.load(example_filename, sr=sample_rate)

    if audio_duration is not None:
        stop = int(audio_duration*orig_sr)
        X_truncated = example_audio[0:stop]
        X_truncated = np.hstack([X_truncated, np.zeros(stop-len(X_truncated))])
    else:
        X_truncated = example_audio

    mel_spec = librosa.feature.melspectrogram(y=X_truncated, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    image_input = mel_spec_db.reshape(*mel_spec_db.shape , 1)
    image_input = np.repeat(image_input, 3, axis=-1)
    image_input = np.expand_dims(image_input, axis=0)
    # image.shape

    return X_truncated, orig_sr, mel_spec_db, image_input

FILES_PATH="./"

# Load classes encoding from the JSON file
with open(FILES_PATH+'zero_encoded_classes.json', 'r') as json_file:
    loaded_zero_encoded_classes = json.load(json_file)
    print(loaded_zero_encoded_classes)    


# LOAD model
h5_file = FILES_PATH+"saved_model.hdf5"
loaded_model = tf.keras.models.load_model(h5_file)


def decode_one_hot(encoded_output, class_labels):
    # Find the index of the highest value in the one-hot encoded array
    index = np.argmax(encoded_output)

    # Map the index to the corresponding class label
    class_label = class_labels[index]

    return class_label

def classify(uploaded_file):
    sample_rate = 44100
    audio_duration = 3
    _, _, _, image_input = preprocess_input(uploaded_file, sample_rate=sample_rate, audio_duration=audio_duration)
    #%time predictions = loaded_model.predict( image_input )
    predictions = loaded_model.predict( image_input )
    return decode_one_hot(predictions[0], loaded_zero_encoded_classes)


def main():
    #titulo
    st.title('Clasificador de audio')

    #titulo de sidebar
    # st.sidebar.header('User Input Parameters')
    #funcion para poner los parametrso en el sidebar
    # def user_input_parameters():
    #     sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    #     sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    #     petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    #     petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    #     data = {'sepal_length': sepal_length,
    #             'sepal_width': sepal_width,
    #             'petal_length': petal_length,
    #             'petal_width': petal_width,
    #             }
    #     features = pd.DataFrame(data, index=[0])
    #     return features

    # df = user_input_parameters()

    #ver sample_rate
    # st.write("Input esperado. Arhivo wav. Sample Rate: 44100")
    st.write("Categorias posibles: ", ', '.join(map(str, loaded_zero_encoded_classes))) #escribe en pantalla categorias posibles
    uploaded_file = st.sidebar.file_uploader("Input esperado. Arhivo wav. Sample Rate: 44100")
    if uploaded_file is not None:
    #     # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data) #escribe en la pantalla principal
        with open("tmp-out.wav", "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(bytes_data)
    #     # To convert to a string based IO:
    #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #     st.write(stringio)

    #json
    # cat_uploaded_file = st.sidebar.file_uploader("Choose a json file")
    # if cat_uploaded_file is not None:
    #      # To read file as string:
    #      string_data = StringIO.read()
    #      st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)
    
    #escoger el modelo preferido
    # option = ['Audio Classifier TL','Linear Regression', 'Logistic Regression', 'SVM']
    option = ['GuitarraPreparada LH']
    # model = st.sidebar.selectbox('Which model you like to use?', option)
    model = st.sidebar.selectbox('Modelo a utilizar', option)

    # st.subheader('Parameters')
    # st.subheader('Parámetros')
    st.subheader(model)
    # st.write(df) # user_input_parameters()

    if st.button('RUN'):
        if model == 'GuitarraPreparada LH':
            st.success(classify("tmp-out.wav"))
        # elif model == 'Linear Regression':
        #     st.success(classify(lin_reg.predict(df)))
        # elif model == 'Logistic Regression':
        #     st.success(classify(log_reg.predict(df)))
        # else:
        #     st.success(classify(svc_m.predict(df)))


if __name__ == '__main__':
    main()
    