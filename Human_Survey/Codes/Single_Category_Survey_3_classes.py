# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:01:27 2023

@author: Aatif
"""

import os
import tkinter as tk
import random
from PIL import Image, ImageTk
import csv
import tkinter.messagebox as messagebox

class ImageLabelApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder
        self.image_files_by_class = {}  # Dictionary to store images grouped by class
        self.current_class = None
        self.current_image_path = None
        self.displayed_images = []

        self.label_var = tk.StringVar()
        self.label_var.set("Select a class to start!")

        self.label = tk.Label(root, textvariable=self.label_var, font=("Helvetica", 18, "bold"))
        self.label.pack()

        self.actual_label_var = tk.StringVar()
        self.actual_label_var.set("Actual Label: None")

        self.actual_label = tk.Label(root, textvariable=self.actual_label_var, font=("Helvetica", 14))
        self.actual_label.pack()

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.class_options = ["low income", "middle income", "high income"]
        self.class_buttons = []
        self.user_reason_var = tk.StringVar()

        for i, class_name in enumerate(self.class_options):
            button = tk.Button(self.button_frame, text=class_name, font=("Helvetica", 14), command=lambda class_name=class_name: self.submit_guess(class_name))
            button.pack(side=tk.LEFT, padx=3, pady=3)
            self.class_buttons.append(button)

        self.user_reason_label = tk.Label(root, text="Write why you chose this option:", font=("Helvetica", 14))
        self.user_reason_label.pack(pady=3)

        self.user_reason_entry = tk.Entry(root, textvariable=self.user_reason_var, font=("Helvetica", 12))
        self.user_reason_entry.pack(pady=3)

        self.load_images()

    def load_images(self):
        for i, class_name in enumerate(self.class_options):
            class_folder = os.path.join(self.image_folder, f"class{i}")
            image_files = os.listdir(class_folder)
            self.image_files_by_class[class_name] = [os.path.join(class_folder, img_file) for img_file in image_files]

    def get_random_image(self):
        class_name = random.choice(self.class_options)
        images = self.image_files_by_class[class_name]
        if images:
            for _ in range(len(images)):
                image_path = random.choice(images)
                if image_path not in self.displayed_images:
                    self.displayed_images.append(image_path)
                    return image_path
        return None

    def display_next_image(self):
        for button in self.class_buttons:
            button.config(state=tk.NORMAL)  # Enable all class buttons
        self.current_image_path = self.get_random_image()
        if self.current_image_path:
            self.display_image(self.current_image_path)
            self.current_class = os.path.basename(os.path.dirname(self.current_image_path))
            self.actual_label_var.set(f"Actual Label: {self.current_class}")
            self.label_var.set("Select the correct class:")
            self.user_reason_var.set("")  # Clear the user's reason entry

    def display_image(self, image_path):
        self.canvas.delete("all")
        img = Image.open(image_path)
        img = img.resize((400, 400), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def submit_guess(self, user_guess):
        user_reason = self.user_reason_var.get()
        if not user_reason:
            messagebox.showerror("Error", "Please write why you chose this option.")
            return

        if self.current_image_path is not None:
            for button in self.class_buttons:
                button.config(state=tk.DISABLED)  # Disable all class buttons until next image is shown
            actual_label = self.current_class
            self.save_guess_to_csv(self.current_image_path, actual_label, user_guess, user_reason)
            self.display_next_image()

    def save_guess_to_csv(self, image_path, actual_label, user_guess, user_reason):
        csv_file = "guesses.csv"
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"# {user_reason}", image_path, actual_label, user_guess])


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Labeling App")
    image_folder = "D:/valid"  # Replace this with the path to your 'interior' folder
    app = ImageLabelApp(root, image_folder)
    app.display_next_image()  # Display the first random image
    root.mainloop()
