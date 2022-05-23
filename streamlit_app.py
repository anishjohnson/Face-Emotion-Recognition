# Import the required libraries.
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# Load the model that we saved in h5 format.
classifier =load_model('model_78.h5')

# Define the emotion labels.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Detect the face using open_cv.
try:
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")


# Now lets create the function that will detect faces and predict their emotions.
class Faceemotion(VideoTransformerBase):
    # Function to detect face and predict its emotion.
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.32, 5)

                 
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),4)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]

            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        return img

# App to deploy on streamlit.
def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Anish Johnson    
            Email : anishjohnson05@gmail.com 
            [LinkedIn] (linkedin.com/in/anish-johnson-594110208)""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                    This web application is capable of detecting a users face using OpenCV 
                    and predicting the emotion He/She is displaying.
                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Anish Johnson using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at anishjohnson05@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()