# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:57:20 2023

@author: Aatif
"""


import csv
import os
import random
from PIL import Image, ImageTk
from tkinter import Tk, Label, Radiobutton, Button, IntVar, Frame, Entry, messagebox


folders = {'class0': '0', 'class1': '1', 'class2': '2', 'class3': '3', 'class4': '4'}
# Create a list of image paths by randomly selecting one from each folder
image_paths = []
for folder, label in folders.items():
    folder_path = f'D:/data/5_Classes/Interior/valid/{folder}'
    file_names = os.listdir(folder_path)
    image_path = os.path.join(folder_path, random.choice(file_names))
    image_paths.append(image_path)


# Keep track of the images that have already been shown
shown_images = set()

# Create the window and label for displaying the images
window = Tk()
window.title('AHI Estimation')
#window.configure(bg='grey')

# Add some text to the window
welcome_label = Label(window, text='Welcome to our survey!', font=('Arial', 20))
welcome_label.pack(padx=10, pady=10)
instruction_label = Label(window, text='Select the appropriate income level bracket you think this image belongs to:', font=('Arial', 14))
instruction_label.pack(padx=10, pady=10)

# Create a frame to hold the image
image_frame = Frame(window)
image_frame.place(relx=0.01, rely=0.18)

# Create a label for the image
label = Label(image_frame)
label.pack(fill='both', expand=True)

# Create a frame to hold the buttons
button_frame = Frame(window)
button_frame.place(relx=0.58, rely=0.23, relwidth=0.2, relheight=0.6)

# Create a frame to hold the buttons
button_frame1 = Frame(window)
button_frame1.place(relx=0.77, rely=0.43, relwidth=0.2, relheight=0.2)


# Create the radio buttons
var = IntVar()
class_labels = list(folders.values())

# Define the hover effect function
def on_enter(widget):
    widget.config(cursor="hand2")


# Create the radio buttons with hover effect and border
Radiobutton(button_frame, text='Highest Income Level', variable=var, value=4, indicatoron=0, width=20, padx=10, pady=10, bg='#f7f7f7', activebackground='#e6e6e6').pack(padx=10, pady=10)
Radiobutton(button_frame, text='Upper-middle Income Level', variable=var, value=3, indicatoron=0, width=20, padx=10, pady=10, bg='#f7f7f7', activebackground='#e6e6e6').pack(padx=10, pady=10)
Radiobutton(button_frame, text='      Median Income Level      ', variable=var, value=2, indicatoron=0, width=20, padx=10, pady=10, bg='#f7f7f7', activebackground='#e6e6e6').pack(padx=10, pady=10)
Radiobutton(button_frame, text='Lower-middle Income Level', variable=var, value=1, indicatoron=0, width=20, padx=10, pady=10, bg='#f7f7f7', activebackground='#e6e6e6').pack(padx=10, pady=10)
Radiobutton(button_frame, text='      Lowest Income Level     ', variable=var, value=0, indicatoron=0, width=20, padx=10, pady=10, bg='#f7f7f7', activebackground='#e6e6e6').pack(padx=10, pady=10)


# Create a frame to hold the comment box
comment_frame = Frame(window)
comment_frame.place(relx=0.5, rely=0.8, relwidth=0.5, relheight=0.1)

# Create the comment box label and entry
comment_label = Label(comment_frame, text='Why did you select this option?', font=('Arial', 12))
comment_label.pack(padx=17, pady=0, side='left')
comment_entry = Entry(comment_frame)
comment_entry.pack(padx=11, pady=0, side='right', fill='x', expand=True)

# Bind the hover effect function to the mouse entering the widget
for widget in button_frame.winfo_children():
    if isinstance(widget, Radiobutton):
        widget.bind("<Enter>", lambda e, widget=widget: on_enter(widget))

# Create the submit button
submit_button = Button(button_frame1, text='Submit', command=lambda: next_image(var.get()), bg='green', font=('Arial', 16))
submit_button.pack(pady=10)




# Define the maximum number of images to show
MAX_IMAGES = 10

# Define a counter for the number of images shown
num_images_shown = 0




# Function to show the next image based on the selected folder and write response to a CSV file
def next_image(selected_folder):
    global shown_images
    global image_paths
    global num_images_shown
    
    
    if num_images_shown == 0:
        # If this is the first image, set comment to None
        comment = None
    else:
        comment = comment_entry.get()

    if num_images_shown >= MAX_IMAGES:
        # Display a message that the survey has ended
        welcome_label.config(text='Thank you for participating in our survey!')
        instruction_label.config(text='')
        label.config(image='')
        for widget in button_frame.winfo_children():
            widget.destroy()
        submit_button.config(text='Exit', command=window.quit, bg='red')
        return
    
    
    # Update the image paths list by randomly selecting one from the selected folder
    selected_index = int(selected_folder)
    selected_folder_path = f'D:/data/5_Classes/Interior/valid/class{selected_index}'
    selected_folder_file_names = os.listdir(selected_folder_path)
    selected_image_path = os.path.join(selected_folder_path, random.choice(selected_folder_file_names))
    image_paths[selected_index] = selected_image_path

    # Write the image name, class label, response, and comment to a CSV file
    image_path = random.choice([path for path in image_paths if path not in shown_images])
    shown_images.add(image_path)
    image = Image.open(image_path)
    image = image.resize((620, 500))
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo
    image_name = os.path.basename(image_path)
    class_label = folders[os.path.basename(os.path.dirname(image_path))]
    

    
    if comment is None:
        # If this is the first image and no comment has been entered, do not write to CSV file
        pass
    elif not comment:
        # If a comment is required and not entered, show an error message
        messagebox.showerror('Error', 'Please enter a comment before submitting.')
        return
    else:
        # Write the comment to the CSV file
        with open('survey_responses.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, class_label, selected_folder, comment])
        comment_entry.delete(0, 'end')
    
    
    # Update the number of images shown
    num_images_shown += 1



next_image(0)
window.state('zoomed')
window.mainloop()