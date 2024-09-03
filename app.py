import streamlit as st
import requests
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import easyocr
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Replace 'en' with your desired language code(s)

# Function to process image
def process_image(frame, area, class_list, reader):
    processed_numbers = []  # Set to store unique plate numbers (hashable)
    result_frame = frame.copy()  # Create a copy of the frame to draw on
    detected_text = None

    # Process the image (no resize needed)
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)

        if result >= 0:
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # EasyOCR text detection
            results = reader.readtext(gray)

            # Handle potential empty detection list
            if results:
                for detection in results:
                    # Extract the detected text
                    text = detection[1]

                    # Only add unique plates (extracted text) to the set
                    if text not in processed_numbers:
                        processed_numbers.append(text)
                        detected_text = text
                        st.write(f"Detected Number Plate: {text}")
                        # Uncomment these lines if you want to save results to a file
                        # with open("car_plate_data.txt", "a") as file:
                        #     file.write(f"{text}\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                    # Draw bounding box
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.polylines(result_frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    return result_frame, detected_text

def display_form(detected_text, frame):
    # Form for user details and detected number plate
    st.header("Visitor Information Form")
    with st.form("visitor_form"):
        name = st.text_input("Name")
        visit_from = st.text_input("Visit From")
        number_plate = st.text_input("Detected Number Plate", value=detected_text or "", disabled=True)
        
        # Hidden input to hold the current date and time
        st.text_input("Current Date and Time", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), key="current_datetime", disabled=True)
        
        submitted = st.form_submit_button("Submit")

        if submitted:
            current_datetime = st.session_state.current_datetime  # Get the current date and time
            data = {
                "numberplate": number_plate,
                "name": name,
                "visit_from": visit_from,
                "datetime": current_datetime  # Include current datetime in the data
            }

            # Send POST request to Flask app on Glitch
            url = "https://lpdata.glitch.me/submit_form"  # Replace with your Glitch app URL
            try:
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    st.success("Form submitted successfully!")
                else:
                    st.error("Failed to submit form. Please try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error submitting form: {str(e)}")


def main():
    st.title("License Plate Detection")
    option = st.selectbox("Choose an input method", ("Upload Image", "Live Camera"))
    
    form_submitted = False  # Flag to track form submission
    
    if option == "Upload Image":
        image_path = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if image_path is not None:
            frame = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), 1)
            st.image(frame, channels="BGR")

            # Load class list for YOLO
            try:
                with open("coco1.txt", "r") as my_file:
                    class_list = my_file.read().strip().split("\n")
            except FileNotFoundError:
                st.error("Class list file 'coco1.txt' not found.")
                return

            area = [(0, 0), (0, frame.shape[0]), (frame.shape[1], frame.shape[0]), (frame.shape[1], 0)]

            processed_frame, detected_text = process_image(frame, area, class_list, reader)

            st.image(processed_frame, channels="BGR", caption="Processed Image")

            form_submitted = display_form(detected_text, frame)
            
    elif option == "Live Camera":
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            frame = cv2.imdecode(np.frombuffer(picture.read(), np.uint8), 1)
            st.image(frame, channels="BGR")

            # Load class list for YOLO
            try:
                with open("coco1.txt", "r") as my_file:
                    class_list = my_file.read().strip().split("\n")
            except FileNotFoundError:
                st.error("Class list file 'coco1.txt' not found.")
                return

            area = [(0, 0), (0, frame.shape[0]), (frame.shape[1], frame.shape[0]), (frame.shape[1], 0)]

            processed_frame, detected_text = process_image(frame, area, class_list, reader)

            st.image(processed_frame, channels="BGR", caption="Processed Image")

            form_submitted = display_form(detected_text, frame)
    
    # Display link button if form was successfully submitted
    link_url = "https://lpdata.glitch.me/"
    button_label = "click here "

# Create a button that acts as a link
    if st.button(button_label):
       st.markdown(f"Updated list[{button_label}]({link_url})")
           

if __name__ == '__main__':
    main()
