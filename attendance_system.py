# import cv2
# import numpy as np
# import pickle
# from keras_facenet import FaceNet
# from datetime import datetime
# import tkinter as tk
# from PIL import Image, ImageTk

# # Initialize FaceNet model
# embedder = FaceNet()

# # Function to load face encodings
# def load_encodings(filename='encodings_keras_facenet.pkl'):
#     with open(filename, 'rb') as file:
#         data = pickle.load(file)
#     return data['encodings'], data['names']

# # Function to mark attendance
# def mark_attendance(name, attendance_set):
#     if name not in attendance_set:
#         with open('attendance.csv', 'a') as file:
#             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             file.write(f'{name},{timestamp}\n')
#         attendance_set.add(name)
#         print(f"Attendance marked for {name}")
#     else:
#         print(f"Attendance already marked for {name}")

# # Function to get face embeddings from a frame
# def get_face_encodings(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     faces = embedder.extract(rgb_image, threshold=0.95)
#     face_encodings = [face['embedding'] for face in faces]
#     face_boxes = [face['box'] for face in faces]
#     return face_encodings, face_boxes

# # Function to close the camera prompt
# def close_camera_prompt():
#     print("Closing the camera prompt.")
#     cv2.destroyAllWindows()

# # Function to process video feed
# def process_video_feed():
#     known_encodings, known_names = load_encodings()
#     attendance_set = set()

#     def show_frame():
#         ret, frame = video_capture.read()

#         if ret:
#             face_encodings, face_boxes = get_face_encodings(frame)

#             for face_encoding, box in zip(face_encodings, face_boxes):
#                 matches = []
#                 for known_encoding in known_encodings:
#                     match = np.linalg.norm(known_encoding - face_encoding)
#                     matches.append(match)

#                 best_match_index = np.argmin(matches)
#                 name = "Unknown"
#                 if matches[best_match_index] < 0.6:
#                     name = known_names[best_match_index]
#                     mark_attendance(name, attendance_set)

#                 (startX, startY, endX, endY) = box
                
#                 # Calculate the width and height of the rectangle
#                 rect_width = endX - startX
#                 rect_height = endY - startY
                
#                 # Calculate the center of the face
#                 face_centerX = (startX + endX) // 2
#                 face_centerY = (startY + endY) // 2
                
#                 # Calculate the coordinates of the top-left corner of the rectangle
#                 rect_startX = max(0, face_centerX - rect_width // 2)
#                 rect_startY = max(0, face_centerY - rect_height // 2)
                
#                 # Draw rectangle around the detected face
#                 cv2.rectangle(frame, (rect_startX, rect_startY), (rect_startX + rect_width, rect_startY + rect_height), (0, 255, 0), 2)
#                 cv2.putText(frame, name, (rect_startX, rect_startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame)
#             imgtk = ImageTk.PhotoImage(image=img)
#             panel.imgtk = imgtk
#             panel.config(image=imgtk)
#             panel.after(10, show_frame)
#         else:
#             close_camera_prompt()

#     root = tk.Tk()
#     root.title("Attendance System")

#     video_capture = cv2.VideoCapture(0)

#     panel = tk.Label(root)
#     panel.pack(padx=10, pady=10)

#     show_frame()

#     root.mainloop()

# if __name__ == "__main__":
#     process_video_feed()
import cv2
import numpy as np
import pickle
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk
from keras_facenet import FaceNet

# Initialize FaceNet model
embedder = FaceNet()

# Function to load face encodings
def load_encodings(filename='encodings_keras_facenet.pkl'):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data['encodings'], data['names']

# Function to mark attendance
def mark_attendance(name, attendance_set):
    if name not in attendance_set:
        with open('attendance.csv', 'a') as file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f'{name},{timestamp}\n')
        attendance_set.add(name)
        print(f"Attendance marked for {name}")
    else:
        print(f"Attendance already marked for {name}")

# Function to close the camera prompt
def close_camera_prompt():
    print("Closing the camera prompt.")
    cv2.destroyAllWindows()

# Function to process video feed
def process_video_feed():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Provide the path to your Haar cascade XML file
    known_encodings, known_names = load_encodings()
    attendance_set = set()

    def show_frame():
        ret, frame = video_capture.read()

        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Draw rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Perform face recognition or mark attendance as needed
                face_image = frame[y:y+h, x:x+w]
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_encoding = embedder.extract(rgb_image, threshold=0.95)
                
                if face_encoding:
                    embedding = face_encoding[0]['embedding']  # Access the embedding from the dictionary
                    matches = [np.linalg.norm(np.array(known_encoding) - np.array(embedding)) for known_encoding in known_encodings]
                    best_match_index = np.argmin(matches)
                    min_distance = min(matches)
                    name = known_names[best_match_index]
                    if min_distance < 0.6:
                        mark_attendance(name, attendance_set)

                        # Draw name label
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
            panel.after(10, show_frame)
        else:
            close_camera_prompt()

    root = tk.Tk()
    root.title("Attendance System")

    video_capture = cv2.VideoCapture(0)

    panel = tk.Label(root)
    panel.pack(padx=10, pady=10)

    show_frame()

    root.mainloop()

if __name__ == "__main__":
    process_video_feed()
