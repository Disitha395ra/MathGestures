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

col1,col2 = st.columns([2,1])
with col1:
    run = st.checkbox('Run',value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")

# Suppress gRPC logging messages
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'

# Configure the Google Generative AI API
genai.configure(api_key="AIzaSyB0FUWPPLTbiuWuRONZjoWjyCvoK9BpU-c")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
# The '0' indicates the first camera connected to your computer; adjust if needed
cap = cv2.VideoCapture(0)
cap.set(3, 680)
cap.set(4, 680)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    elif fingers == [1,1,1,1,1]:
        canvas = np.zeros_like(img)

    return current_pos, canvas

def sendToAI(model, fingers, canvas):
    if fingers == [1, 1, 0, 0, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["solve the math problem",pil_image])
        return response.text

prev_pos = None
canvas = None
image_combines = None
output_text = ""
# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
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
        output_text = sendToAI(model, fingers, canvas)  # Pass the canvas to the sendToAI function

    image_combines = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combines,channels="BGR")

    output_text_area.text("output_text")

    # Display the image in a window
    #cv2.imshow("Image",img)
    #cv2.imshow("Canvas",canvas)
    #cv2.imshow("Image_combines", image_combines)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
