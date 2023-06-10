# Knowledge Test Project

This project consists of two main components: a Handwritten Detection and an Audio Recognition system. The project utilizes deep learning algorithms implemented with TensorFlow to achieve accurate classification and analysis.

## Installation

1. Create python environment `python -m venv virtualenv`
2. Clone the repository to virtual environment: `git clone https://github.com/example/repo.git`
3. Install the required `pip install -r requirements.txt`
4. Open a terminal or command prompt and navigate to the project directory.
5. Run the following command to start the application: `streamlit run app.py`
6. The application will open in your web browser, presenting the main menu.

## Handwritten Detection

The Handwritten Detection project recognizes and interprets handwritten text. It utilizes the MNIST dataset and deep learning algorithms to achieve accurate digit recognition.

To use the Handwritten Detection feature:

1. Select "Handwritten Detection" from the main menu.
2. Follow the usage instructions provided on the web application.
3. Upload an image file containing a handwritten digit or select a sample image.
4. Click the "Predict" button to initiate the recognition process.
5. The application will analyze the image using advanced deep learning algorithms and display the predicted digit.

## Audio Recognition

The Audio Recognition project classifies and analyzes audio data. It is specifically designed to classify audio into two classes: "yes" and "no". The underlying model has been trained using the Speech Commands Dataset.

To use the Audio Recognition feature:

1. Select "Audio Recognition" from the main menu.
2. Follow the usage instructions provided on the web application.
3. Click the "Record" button to start recording audio for 5 seconds. The recording will automatically finish after 5 seconds.
4. Alternatively, you can drag and drop an audio file to classify it.
5. The application will process the audio and display the predicted class label ("yes" or "no").

## About

The Knowledge Test project showcasing two projects: Handwritten Detection and Audio Recognition. Both projects utilize advanced deep learning algorithms implemented with TensorFlow.

The Handwritten Detection project achieves a commendable 97% accuracy on the test dataset. However, the Audio Recognition model's performance is comparatively lower, with a modest 52% test accuracy. This discrepancy is due to the limited amount of data available for training the audio recognition model.

Efforts are being made to improve the audio recognition model by acquiring and incorporating additional data to enhance its performance. The goal is to refine the audio recognition system and achieve more accurate results.


## Issues

During the deployment of the project to Streamlit Cloud, an issue was encountered due to the lack of PortAudio support. Streamlit Cloud does not currently provide native support for PortAudio, which is a library required by the project for audio functionality.

As a result, the deployment process was not successful due to the inability to access PortAudio on the Streamlit Cloud platform. This limitation prevents the audio recognition feature of the project from functioning as intended in the deployed version.

To resolve this issue, alternative deployment options should be explored that support PortAudio or provide similar audio functionality. Alternatively, adjustments can be made to the project code to accommodate the limitations of the deployment platform.

Please note that it is important to thoroughly review the documentation and requirements of the deployment platform to ensure compatibility with the project's dependencies and libraries.

If you encounter any further issues or require additional assistance with the deployment process, please let me know, and I'll be happy to help.
