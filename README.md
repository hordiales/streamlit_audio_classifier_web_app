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

# Tracking large files with https://git-lfs.com/
An open source Git extension for versioning large files
Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

git lfs track "*.hdf5"
git add .gitattributes
git add saved_model.hdf5
git commit -m "Add model file"
git push origin main