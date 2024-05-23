This is a image processing model for automatic attendance marking system

!!!!!!   THIS REPO CONTAINS ONLY THE MODEL !!!!!!!!!

You can integrate the model with your ui to use it for face recognization

// Steps to check the working of the model :

Step1: Clone the repository into your local device 

Step2: Locate to the repositories directory

Step3: Run the "python3 install requirements.txt" command in your terminal

Step4: After the installations of all the required libraries run the " train_mdoel.py" file this file contains the code for training of the model after running the file the model captures the photos with the help of webcam
and trains itself to identify the user next time. The file automatically creates 2 new directories i.e dataset ( contains folders with naming after the username of the user that contain the images of the users ) and
attendance.csv file for marking the attendance of the user with username, date and timestamp

Step5: After the " train_model.py " runs successfully run the " attendance.py " file this file contains code for marking the attendance of the user containing the data of the user in the dataset folder and the user's face encodings in encodings.pkl file
