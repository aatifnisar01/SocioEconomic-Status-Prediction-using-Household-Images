# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 22:44:10 2023

@author: Aatif
"""

import csv
import os
import random
from PIL import Image, ImageTk
from tkinter import Tk, Label, Radiobutton, Button, IntVar, simpledialog

# Define the folders and their corresponding labels
folders = {'class0': '0', 'class1': '1', 'class2': '2', 'class3': '3', 'class4': '4'}

# Get the surveyor's name
root = Tk()
root.withdraw()
surveyor_name = simpledialog.askstring(title="Surveyor's Name", prompt="What is your name?")
if not surveyor_name:
    exit()
csv_file_path = f'responses_{surveyor_name}.csv'

# Create the CSV file and write the header row
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Class Label', 'Response'])

# Create a list of image paths by randomly selecting one from each folder
image_paths = []
for folder, label in folders.items():
    folder_path = f'C:/Survey/{folder}'
    file_names = os.listdir(folder_path)
    image_path = os.path.join(folder_path, random.choice(file_names))
    image_paths.append(image_path)

# Keep track of the images that have already been shown
shown_images = set()

def show_image(image_count):
    # If we have shown all images, exit the program
    if image_count > 10:
        return

    # Create a window and display the image
    window = Tk()
    window.title('AHI Estimation')

    image_path = random.choice([path for path in image_paths if path not in shown_images])
    shown_images.add(image_path)
    image = Image.open(image_path)
    image = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    label = Label(window, image=photo)
    label.grid(row=0, column=0)

    # Create the radio buttons
    var = IntVar()
    class_labels = list(folders.values())

    Radiobutton(window, text='Lowest Income Level', variable=var, value=0).grid(row=1, column=0, padx=5, pady=5)
    Radiobutton(window, text='Lower-middle Income Level', variable=var, value=1).grid(row=2, column=0, padx=5, pady=5)
    Radiobutton(window, text='Median Income Level', variable=var, value=2).grid(row=3, column=0, padx=10, pady=10)
    Radiobutton(window, text='Upper-middle Income Level', variable=var, value=3).grid(row=4, column=0, padx=10, pady=10)
    Radiobutton(window, text='Highest Income Level', variable=var, value=4).grid(row=5, column=0, padx=10, pady=10)

    # Create the submit button
    submit_button = Button(window, text='Submit', command=lambda: next_image(var.get(), image_path, image_count+1))
    submit_button.grid(row=2, column=2, pady=10)
    
    
    # Function to show the next image based on the selected folder and write response to a CSV file
    def next_image(selected_folder):
        global shown_images
        global image_paths

        # Update the image paths list by randomly selecting one from the selected folder
        selected_index = int(selected_folder)
        selected_folder_path = f'C:/Users/Aatif/OneDrive/Desktop/IITD/archive/HouseInterior/HouseInterior/val/class{selected_index}'
        selected_folder_file_names = os.listdir(selected_folder_path)
        selected_image_path = os.path.join(selected_folder_path, random.choice(selected_folder_file_names))
        image_paths[selected_index] = selected_image_path

        # Write the image name, class label, and response to a CSV file
        image_path = random.choice([path for path in image_paths if path not in shown_images])
        shown_images.add(image_path)
        image = Image.open(image_path)
        image = image.resize((600, 600))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
        image_name = os.path.basename(image_path)
        class_label = folders[os.path.basename(os.path.dirname(image_path))]
        with open('survey_responses.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, class_label, selected_folder])

    next_image(0)
    window.state('zoomed')
    window.mainloop()
    
show_image(1)    

# =============================================================================
#     # Function to get the next image based on the selected folder and write response to the CSV file
#     def next_image(selected_folder, image_path, image_count):
#         # Update the image paths list by randomly selecting one from the selected folder
#         selected_index = int(selected_folder)
#         selected_folder_path = f'C:/Survey/class{selected_index}'
#         selected_folder_file_names = os.listdir(selected_folder_path)
#         selected_image_path = os.path.join(selected_folder_path, random.choice(selected_folder_file_names))
#         image_paths[selected_index] = selected_image_path
# 
#         # Write the image name, class label, and response to the CSV file
#         image_name = os.path.basename(image_path)
#         class_label = folders[os.path.basename(os.path.dirname(image_path))]
#         response = selected_folder
#         with open(csv_file_path, 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([image_name, class_label, response])
#             
#     # Start the event loop
#     window.mainloop()
#     
# # Show the first image
# show_image(1)
# 
# =============================================================================
