import streamlit as st
import matplotlib.pyplot as plt
import librosa
import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

image_icon = Image.open('temp/icon/speech.png')

st.cache_data()
def load_audio():
    model = load_model('temp/load_file/model/modelANNv2.h5')
    return model

st.cache_data()
def load_handwritten():
    model = load_model('temp/load_file/model/modelHandwritten.h5')
    return model

model_audio = load_audio()

model_number = load_handwritten()

st.set_page_config(
    page_title='Knowlegde Test', 
    layout='wide', 
    initial_sidebar_state='auto', 
    page_icon=image_icon)

with st.sidebar:
    st.image(image_icon)
    st.title('Knowlegde Test Project')
    choice = st.selectbox('Main Menu', ['Handwritten Detection', 'Audio Recognition', 'About'])
    st.sidebar.markdown("---")
    st.info('This website showcases two projects: Handwritten Detection and Audio Recognition. The Handwritten Detection project recognizes and interprets handwritten text, while the Audio Recognition project classifies and analyzes audio data. Both projects utilize advanced Deep Learning algorithms.')

if choice == 'Handwritten Detection':
    
    def predict_external_data(image_path):
        # Preprocess the input image
        input_image = plt.imread(image_path)  # Read the image using plt
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        input_image = cv2.resize(input_image, (28, 28))  # Resize to (28, 28)
        input_image = input_image / 255.0  # Normalize array

        # Make predictions
        predictions = model_number.predict(np.expand_dims(input_image, axis=0))

        # Interpret the results
        predicted_digit = np.argmax(predictions)

        return predicted_digit
    
    img_folder = 'temp/number_sample'  # Path to the folder containing the images
    st.title("Let's classify the number!")
    st.markdown('This web application is designed for handwritten digit recognition using the MNIST dataset. To utilize this project, simply upload an image containing a handwritten digit and click the "Predict" button. The underlying model will then analyze the image and provide a prediction for the corresponding digit. This application leverages advanced deep learning algorithms to achieve accurate and efficient digit recognition.')
    st.markdown("## Usage Instructions")
    st.markdown("1. **Upload:** Click the 'Upload' button to select an image file containing a handwritten digit. Supported formats include JPEG, and JPG.")
    st.markdown("2. **Predict:** After uploading the image, click the 'Predict' button to initiate the recognition process.")
    st.markdown("3. **View Prediction:** The application will analyze the uploaded image using advanced deep learning algorithms. The predicted digit will be displayed, indicating the recognized value of the handwritten digit.")
    st.caption("Here's some sample images")

    # Loop through the images in the folder
    columns = st.columns(4)  # Create 4 columns
    index = 0
    selected_image_path = None
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.raw'):
            image_path = os.path.join(img_folder, filename)
            with columns[index % 4]:  # Cycle through the columns
                button = st.button(label="predict-sample", key=filename)
                image = Image.open(image_path)
                image.thumbnail((5000, 5000))  # Adjust the size of the thumbnail
                st.image(image, caption=f'Sample Image: {filename}', use_column_width=True)
                if button:
                    st.markdown(f"**Selected Sample Image: {filename}**")
                    sample_result = predict_external_data(image_path)
                    st.info(f'Image predicted as {sample_result}')
            index += 1

    image_input = st.file_uploader("Upload an image file", type=["jpg", "jpeg"], accept_multiple_files=False)


    if image_input is not None:
        # Display the uploaded image
        image = Image.open(image_input)
        image = image.resize((300, 300))
        st.image(image, caption="Uploaded Image")

    predict_button = st.button('Predict', type='primary')
    st.markdown("---")
    
    if predict_button:
        if image_input is None and selected_image_path is None:
            st.warning("Please upload an image or select a sample image.")
        elif image_input is not None:
            predict = predict_external_data(image_input)
            st.info(f'Image predicted as {predict}')
            
if choice == 'Audio Recognition':
    st.title("Let's classify with recorded audio!")
    st.markdown('This web application serves as an audio classifier, specifically designed to classify audio into two classes: "yes" and "no". The underlying model has been trained using the Speech Commands Dataset, ensuring accurate and reliable predictions. Utilizing this web application is effortless—simply click the "Record" button in 5 second recording state automatically finish also we can predict file as easy as drag and drop. The application will process the recorded audio and promptly display the corresponding class prediction.')
    st.markdown("## Usage Instructions")
    st.markdown("1. **Record:** Click the 'Record' button to start recording audio for 5 seconds. The recording will automatically finish after 5 seconds.")
    st.markdown("2. **Upload:** Alternatively, you can drag and drop an audio file to classify it.")
    st.markdown("3. **Prediction:** After recording or uploading an audio file, the application will process the audio and display the predicted class label ('yes' or 'no').")

    st.markdown("---")

    # Function to handle audio recording
    # def record_audio(duration):
    #     audio = sd.rec(int(duration * sr), samplerate=None, channels=1)
    #     sd.wait()  # Wait for the recording to complete
    #     return audio.flatten()

    # Set the desired length for reshaping the audio
    target_length = 16000
    
    # Function to preprocess an audio file using librosa and reshape to target_length
    def preprocess_audio(file_path):
        audio, sr = librosa.load(file_path, sr=None)  # Load audio file
        if len(audio) != target_length:
            audio = librosa.util.fix_length(audio, size=target_length)  # Reshape audio to target_length
        return audio

    # Function to predict the class label of an audio file
    def predict_audio(file_path):
        # Preprocess the audio file and reshape it to target_length
        audio = preprocess_audio(file_path)

        # Pad the preprocessed audio
        padded_audio = pad_sequences([audio], padding='post', truncating='post', dtype='float32')

        # Make the prediction
        prediction = model_audio.predict(padded_audio)

        # Map the prediction to class labels
        if prediction[0] >= 0.5:
            class_label = 'yes'
        else:
            class_label = 'no'

        return class_label

    # # Function to save the recorded audio to a WAV file
    # def save_audio_to_wav(audio, file_path):
    #     sf.write(file_path, audio, sr)
        
    # # Configuration for audio recording
    # sr = 44100
    # duration = 5  # Duration in seconds

    audio_uploader = st.file_uploader("Upload an audio file", type=["wav", "raw", "mp3"], accept_multiple_files=False)
    # record = st.button('Record', type='primary')

    # if record:
    #     st.info('Recording started... Please speak into the microphone')
        
    #     audio = record_audio(duration)
    #     st.info('Recording completed!')

    #     # Save the recorded audio to a WAV file
    #     file_path = 'temp/audio/recorded_audio.wav'  # Specify the file path and name
    #     save_audio_to_wav(audio, file_path)

    #     # Example code to display the recorded waveform
    #     fig, ax = plt.subplots()
    #     ax.plot(audio)
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Amplitude')

    #     # Center the image
    #     col1, col2, col3 = st.columns(3)
    #     col2.write("")  # Create an empty column for center alignment
    #     col2.pyplot(fig)

    #     # Reduce the size of the image
    #     fig.set_figwidth(6)  # Adjust the width as needed
    #     fig.set_figheight(4)  # Adjust the height as needed
        
    #     # Add a play button to listen to the recorded audio
    #     st.audio(file_path)

    #     # Predict the class label of the audio file
    #     prediction = predict_audio("temp/audio/recorded_audio.wav")
    #     st.info(f'Audio predicted as: {prediction}')
        
    if audio_uploader:
        st.audio(audio_uploader)
        prediction = predict_audio(audio_uploader)
        st.info(f'Audio predicted as: {prediction}')


if choice == 'About':
    st.title('About')
    st.markdown("---")
    st.header('Knowledge Test PT WIDYA INFORMASI NUSANTARA')
    st.markdown('This project encompasses two main components: a handwritten digits classifier and an audio recognition system. Both models have been developed utilizing the TensorFlow framework with Deep Neural Network Architecture. In addition, the audio recognition model has been trained using the LSTM-GRU algorithm to enhance its performance. These models demonstrate the ability to classify both numbers and audio data.')
    st.markdown("It is important to note that while the handwritten digits classifier achieves a commendable 97% accuracy on the test dataset, the audio recognition model's performance is comparatively lower. This discrepancy can be attributed to a limited amount of data available for training the audio recognition model. As a result, the model achieves a modest 52% test accuracy on the audio recognition task.")
    st.markdown("Efforts are being made to improve the audio recognition model by acquiring and incorporating additional data to enhance its performance. By addressing the data limitations, we aim to further refine the audio recognition system and achieve more accurate results.")
    st.markdown("")