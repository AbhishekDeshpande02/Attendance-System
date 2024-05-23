import os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from datetime import datetime
import shutil
import time

# Initialize FaceNet model
embedder = FaceNet()

# Function to load face encodings and names
def load_encodings(filename='encodings_keras_facenet.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data['encodings'], data['names']
    else:
        return [], []

# Function to save face encodings and names
def save_encodings(encodings, names, filename='encodings_keras_facenet.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump({"encodings": encodings, "names": names}, file)

# Function to mark attendance
def mark_attendance(name):
    with open('attendance.csv', 'a') as file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f'{name},{timestamp}\n')
    print(f"Attendance marked for {name}")

# Function to get face embeddings from a frame
def get_face_encodings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = embedder.extract(rgb_image, threshold=0.95)
    face_encodings = [face['embedding'] for face in faces]
    return face_encodings

# Function to capture images from webcam automatically
def capture_images(employee_name, save_path, num_images=20):
    cap = cv2.VideoCapture(0)
    captured_images = []
    img_count = 0

    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_count += 1
        image_path = os.path.join(save_path, f"{employee_name}_{img_count}.jpg")
        cv2.imwrite(image_path, frame)
        captured_images.append(frame)
        print(f"Captured {img_count} / {num_images}")

        # Display the frame (optional)
        cv2.imshow("Capturing Images", frame)
        
        # Wait for 1 second before capturing the next image
        time.sleep(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return captured_images

# Function to add a new user
def add_user():
    dataset_path = 'dataset'
    encodings_file = 'encodings_keras_facenet.pkl'

    known_encodings, known_names = load_encodings(encodings_file)

    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Check if encodings file exists
    if not os.path.exists(encodings_file):
        save_encodings(known_encodings, known_names, encodings_file)

    # Check for deleted users and remove their encodings
    existing_users = set(known_names)
    current_users = set(os.listdir(dataset_path))
    deleted_users = existing_users - current_users
    if deleted_users:
        for user in deleted_users:
            while user in known_names:
                index = known_names.index(user)
                del known_encodings[index]
                del known_names[index]
        save_encodings(known_encodings, known_names, encodings_file)

    # Input the new employee name
    employee_name = input("Enter the new employee's name: ").strip()
    employee_dir = os.path.join(dataset_path, employee_name)

    if not os.path.exists(employee_dir):
        os.makedirs(employee_dir)

    # Capture images for the new employee
    print("Capturing images for the new employee...")
    captured_images = capture_images(employee_name, employee_dir, num_images=20)

    # Encode captured images
    for image in captured_images:
        encodings = get_face_encodings(image)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(employee_name)

    # Save updated encodings
    save_encodings(known_encodings, known_names, encodings_file)
    print("New encodings saved to encodings_keras_facenet.pkl")

    # Mark attendance for the newly added employee
    mark_attendance(employee_name)

# Function to remove an existing user
def remove_user():
    dataset_path = 'dataset'
    encodings_file = 'encodings_keras_facenet.pkl'

    known_encodings, known_names = load_encodings(encodings_file)

    # Input the employee name to be removed
    employee_name = input("Enter the employee's name to be removed: ").strip()

    if employee_name in known_names:
        # Remove the user's directory and its contents
        user_dir = os.path.join(dataset_path, employee_name)
        shutil.rmtree(user_dir)

        # Remove the user's encodings
        while employee_name in known_names:
            index = known_names.index(employee_name)
            del known_encodings[index]
            del known_names[index]

        # Save updated encodings
        save_encodings(known_encodings, known_names, encodings_file)
        print(f"Employee '{employee_name}' has been removed.")
    else:
        print(f"Employee '{employee_name}' not found in the database.")

# Main function
def main():
    while True:
        print("\nOptions:")
        print("1. Add new user")
        print("2. Remove existing user")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            add_user()
        elif choice == '2':
            remove_user()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
