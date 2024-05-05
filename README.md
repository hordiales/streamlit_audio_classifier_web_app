# streamlit_audio_classifier_web_app
Audio Classifier ML web app

## To run streamlit web
(si falla por algo de toml, borrar archivos en ~/.streamlit)
    $ streamlit run iris_web.py

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog

## To run streamlit web as Docker service
### Build the Docker image
    $ docker build --rm -t streamlit-audioclassifier-webapp .

### Run the Docker image as a service
    $ docker run -d -p 80:8501 --name audioclassifier-ml-webapp streamlit-audioclassifier-webapp
