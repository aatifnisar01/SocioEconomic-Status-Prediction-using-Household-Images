# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:05:33 2023

@author: Aatif
"""


import csv
import os
import random
from PIL import Image, ImageTk
from tkinter import Tk, Label, Radiobutton, Button, IntVar, Frame, Entry, messagebox


folders = {'class0': '0', 'class1': '1', 'class2': '2', 'class3': '3', 'class4': '4'}
# Create a list of image paths by randomly selecting one from each folder

image_paths1 = []
for folder, label in folders.items():
    folder_path = f'C:/AHI Data/Front_Images/valid/{folder}'
    file_names = os.listdir(folder_path)
    image_path = os.path.join(folder_path, random.choice(file_names))
    image_paths1.append(image_path)

image_paths2 = []
for folder, label in folders.items():
    folder_path = f'A:/AHI Data/Cooking_Images/valid/{folder}'
    file_names = os.listdir(folder_path)
    image_path = os.path.join(folder_path, random.choice(file_names))
    image_paths2.append(image_path)
    
image_paths3 = []
for folder, label in folders.items():
    folder_path = f'A:/AHI Data/Interior_Images/valid/{folder}'
    file_names = os.listdir(folder_path)
    image_path = os.path.join(folder_path, random.choice(file_names))
    image_paths3.append(image_path)
    

# Keep track of the images that have already been shown
shown_images = set()

# Create the window and label for displaying the images
window = Tk()
window.title('AHI Estimation')
#window.configure(bg='grey')


# Create a frame to hold the image
image_frame1 = Frame(window)
image_frame1.place(relx=0.01, rely=0.07)

# Create a frame to hold the image
image_frame2 = Frame(window)
image_frame2.place(relx=0.38, rely=0.07)

# Create a frame to hold the image
image_frame3 = Frame(window)
image_frame3.place(relx=0.2, rely=0.55)


# Create a label for the image
label1 = Label(image_frame1)
label1.pack(fill='both', expand=True)
text1 = Label(image_frame1, text='Front Image', bg='#f7f7f7', font=('Arial', 12))
text1.place(relx=0.5, rely=0.1, anchor='center')

# Create a label for the image
label2 = Label(image_frame2)
label2.pack(fill='both', expand=True)
text2 = Label(image_frame2, text='Cooking Image', bg='#f7f7f7', font=('Arial', 12))
text2.place(relx=0.5, rely=0.1, anchor='center')

# Create a label for the image
label3 = Label(image_frame3)
label3.pack(fill='both', expand=True)
text3 = Label(image_frame3, text='Interior Image', bg='#f7f7f7', font=('Arial', 12))
text3.place(relx=0.5, rely=0.1, anchor='center')

# Create a frame to hold the buttons
button_frame = Frame(window)
button_frame.place(relx=0.71, rely=0.23, relwidth=0.2, relheight=0.6)

# Create a frame to hold the buttons
button_frame1 = Frame(window)
button_frame1.place(relx=0.87, rely=0.43, relwidth=0.17, relheight=0.1)


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
comment_frame.place(relx=0.67, rely=0.8, relwidth=0.3, relheight=0.1)

# Create the comment box label and entry
comment_label = Label(comment_frame, text='Why did you select this option?', font=('Arial', 12))
comment_label.pack(padx=17, pady=0, side='top')
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
        label1.config(image='')
        for widget in button_frame.winfo_children():
            widget.destroy()
        submit_button.config(text='Exit', command=window.quit, bg='red')
        return
    
    
    # Update the image paths list by randomly selecting one from the selected folder
    selected_index = int(selected_folder)
    
    selected_folder_path1 = f'C:/AHI Data/Front_Images/valid/class{selected_index}'
    selected_folder_file_names1 = os.listdir(selected_folder_path1)
    selected_image_path1 = os.path.join(selected_folder_path1, random.choice(selected_folder_file_names1))
    image_paths1[selected_index] = selected_image_path1
    
    selected_folder_path2 = f'A:/AHI Data/Cooking_Images/valid/class{selected_index}'
    selected_folder_file_names2 = os.listdir(selected_folder_path2)
    selected_image_path2 = os.path.join(selected_folder_path2, random.choice(selected_folder_file_names2))
    image_paths2[selected_index] = selected_image_path2
    
    selected_folder_path3 = f'A:/AHI Data/Interior_Images/valid/class{selected_index}'
    selected_folder_file_names3 = os.listdir(selected_folder_path3)
    selected_image_path3 = os.path.join(selected_folder_path3, random.choice(selected_folder_file_names3))
    image_paths3[selected_index] = selected_image_path3

    # Write the image name, class label, response, and comment to a CSV file
    image_path1 = random.choice([path for path in image_paths1 if path not in shown_images])
    shown_images.add(image_path1)
    
    # Write the image name, class label, response, and comment to a CSV file
    image_path2 = random.choice([path for path in image_paths2 if path not in shown_images])
    shown_images.add(image_path2)
    
    # Write the image name, class label, response, and comment to a CSV file
    image_path3 = random.choice([path for path in image_paths3 if path not in shown_images])
    shown_images.add(image_path3)
    
    
    
    image1 = Image.open(image_path1)
    image1 = image1.resize((420, 280))
    photo1 = ImageTk.PhotoImage(image1)
    label1.config(image=photo1)
    label1.image = photo1
    image_name = os.path.basename(image_path1)
    class_label = folders[os.path.basename(os.path.dirname(image_path1))]
    
    image2 = Image.open(image_path2)
    image2 = image2.resize((420, 280))
    photo2 = ImageTk.PhotoImage(image2)
    label2.config(image=photo2)
    label2.image = photo2
# =============================================================================
#     image_name = os.path.basename(image_path2)
#     class_label = folders[os.path.basename(os.path.dirname(image_path2))]
# =============================================================================
    
    image3 = Image.open(image_path3)
    image3 = image3.resize((420, 280))
    photo3 = ImageTk.PhotoImage(image3)
    label3.config(image=photo3)
    label3.image = photo3
# =============================================================================
#     image_name = os.path.basename(image_path1)
#     class_label = folders[os.path.basename(os.path.dirname(image_path1))]
# =============================================================================
    
    

    
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