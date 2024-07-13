
# Math Gesture AI Model

Math Gesture AI is a project that utilizes computer vision and machine learning to recognize hand gestures and perform mathematical operations based on those gestures. This project is built using Gemini AI developer tools, OpenCV, NumPy, and MediaPipe.

# Features
- Real-time hand gesture recognition
- Supports basic mathematical operations (addition, subtraction, multiplication, division)
- User-friendly interface
- Accurate and fast gesture detection


# Steps To Run

- Clone the Repository
	git clone https://github.com/yourusername/math-gesture-ai.git
	cd math-gesture-ai

- Create a virtual environment
	python -m venv venv
	source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

- Install cvzone and MediaPipe, streamlite and other imported libraries you can see in the top of main.py

- To Run 
	streamlit run main.py

# Technologies Used
- Gemini AI developer tools - For building and 		deploying AI models.
- OpenCV - For real-time computer vision.
- NumPy - For numerical operations.
- MediaPipe - For hand tracking and gesture 		recognition.

## Usage/Examples

```python
import os
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
st.header('Math Gestures AI model')

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.empty()

# Suppress gRPC logging messages
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'

# Configure the Google Generative AI API
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

```
- Setting up Streamlit and Initial Configuration
This segment imports necessary libraries and initializes the Streamlit interface.
It configures the page layout, sets up a header, and creates two columns: one for the video feed and one for the output text.
It also suppresses gRPC logging messages and configures the Google Generative AI API with the provided API key.

```python

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 680)
cap.set(4, 680)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

```
- Initializing Webcam and Hand Detector
This segment initializes the webcam to capture video at a resolution of 680x680 pixels.
It sets up the HandDetector from the cvzone library with specified parameters for hand detection, including detection confidence and tracking confidence.

```javascript
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

```
- Function to Get Hand Information
This function processes the input image to detect hands using the HandDetector.
It returns a list of fingers that are up (fingers) and a list of landmarks (lmList) if a hand is detected.
If no hands are detected, it returns None.

```javascript
prev_pos = None
canvas = None
output_text = ""

while run:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, fingers, canvas) or ""  

    image_combines = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combines, channels="BGR")
    output_text_area.text(output_text)

cap.release()
cv2.destroyAllWindows()

```
- Main Loop for Real-time Hand Gesture Recognition
This loop runs continuously while the run checkbox is checked.
It captures frames from the webcam, flips them for a mirror view, and initializes a drawing canvas.
It gets hand information from the current frame and uses the draw function to update the canvas based on the detected gesture.
If a specific gesture is detected, the sendToAI function sends the canvas to the AI model to solve the math problem, updating the output_text with the response.
The combined image of the webcam feed and the drawing canvas is displayed in Streamlit, along with the AI model's output.



## License

[MIT](https://choosealicense.com/licenses/mit/)


## Tech Stack

* Python
* OpenCV
* cvzone
* NumPy
* MediaPipe
* Streamlit
* Google Generative AI (gemini-1.5-flash)
* PIL (Python Imaging Library)
* gRPC


## Authors [Developer]
- [@Disitha395ra](https://github.com/Disitha395ra)


## FAQ

#### What is Math Gesture AI?

Math Gesture AI is an application that uses hand gesture recognition to perform mathematical operations. It utilizes computer vision and machine learning to interpret gestures and solve math problems in real-time.

####  How does Math Gesture AI recognize hand gestures?

The application uses the cvzone library's HandDetector class along with MediaPipe to detect hand landmarks and recognize gestures.

#### What mathematical operations are supported?

The application currently supports basic mathematical operations such as addition, subtraction, multiplication, and division.and other All mathmatical operation like intergration , diveration , matrix solving etc


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


