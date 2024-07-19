# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:01:35 2023

@author: Aatif
"""



import tkinter as tk
import os
import random
from PIL import Image, ImageTk

class ImageBrowser:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Browser")
        self.master.geometry("800x600")
        self.master.attributes('-fullscreen', True)
        self.current_image_idx = 0
        #self.master.configure(bg='pink')
        self.image_paths = []
        self.labels = []
        self.image_label = tk.Label(self.master)
        self.label_text = tk.StringVar()
        self.label_text.set("")
        self.label = tk.Label(self.master, textvariable=self.label_text, font=("Arial", 20, "bold"))
        self.forward_button = tk.Button(self.master, text=">>", command=self.next_image, font=("Arial", 20))
        self.backward_button = tk.Button(self.master, text="<<", command=self.previous_image, font=("Arial", 20))
        self.load_images()
        self.display_image()
        self.exit_button = tk.Button(self.master, text="Exit", command=self.master.quit, font=("Arial", 14), bg="#ffcccc", fg="#660000", width=10, height=2)
        self.exit_button.pack(side="bottom", pady=20)

    def load_images(self):
        folder_names = ['class0', 'class1', 'class2', 'class3', 'class4']
        for i, folder in enumerate(folder_names):
            folder_path = f"C:/AHI Data/Front_Images/valid/{folder}/"
            images_in_folder = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    images_in_folder.append(os.path.join(folder_path, filename))
            if i < 5:
                self.image_paths.extend(images_in_folder[:5])
                self.labels.extend([folder]*5)
            else:
                self.image_paths.extend(images_in_folder)
                self.labels.extend([folder]*len(images_in_folder))
        self.labels = self.labels[:45]
        self.image_paths = self.image_paths[:45]
        random_image_paths = []
        for folder in folder_names:
            folder_path = f"C:/AHI Data/Front_Images/valid/{folder}/"
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    if file_path not in self.image_paths:
                        random_image_paths.append((file_path, folder))
        random.shuffle(random_image_paths)
        random_image_paths = random_image_paths[:20]
        for path, label in random_image_paths:
            self.image_paths.append(path)
            self.labels.append(label)

    def display_image(self):
        image_path = self.image_paths[self.current_image_idx]
        image = Image.open(image_path)
        resized_image = image.resize((550, 550), Image.BICUBIC)
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.image_label.pack(side="top", pady=12)
        
        # Set label text based on income level
        income_level = self.labels[self.current_image_idx]
        if income_level == "class0":
            label_text = "Lowest Income Level"
        elif income_level == "class1":
            label_text = "Lower-middle Income Level"
        elif income_level == "class2":
            label_text = "Median Income Level"
        elif income_level == "class3":
            label_text = "Upper-middle Income Level"
        elif income_level == "class4":
            label_text = "Highest Income Level"
        else:
            label_text = ""

        self.label_text.set(f"Label: {label_text}")
        self.label.pack(side="top", pady=10)

        self.backward_button.pack(side="left", padx=20, pady=20)
        self.forward_button.pack(side="right", padx=20, pady=20)


    def next_image(self):
        self.current_image_idx = (self.current_image_idx + 1) % len(self.image_paths)
        self.label_text.set(f"Label: {self.labels[self.current_image_idx]}")
        self.image_label.pack_forget()
        self.label.pack_forget()
        self.display_image()
        
        # Check if all images have been shown
        if self.current_image_idx == len(self.image_paths) - 1:
            self.label_text.set("Thank you for browsing!")
            self.forward_button.config(state="disabled")

    def previous_image(self):
        self.current_image_idx = (self.current_image_idx - 1) % len(self.image_paths)
        self.label_text.set(f"Label: {self.labels[self.current_image_idx]}")
        self.image_label.pack_forget()
        self.label.pack_forget()
        self.display_image()


root = tk.Tk()
app = ImageBrowser(root)
root.mainloop()