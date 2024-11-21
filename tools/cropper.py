import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, BOTH

import numpy as np
from PIL import Image, ImageTk, ImageDraw

from utils.plot import frame
from utils.labels import read_label, write_labels, delete_labels
from utils.augment import crop


class_name = ['bicycle', 'bus', 'car', 'person','motorbike', 'truck']


class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Configure application
        self.title("Image Cropper")

        # Create a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initializing frames
        self.frames = {
            Cropper: Cropper(container, self),
            Menu: Menu(container, self),
        }

        # Position on the frames
        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="news")

        # Set Menu as entrypoint
        self.model = None
        self.show_frame(Menu)

    def show_frame(self, container):
        # Display the frame passes as parameter
        frame = self.frames[container]

        # Resize cropper window, and load the current image.
        if container == Cropper:
            self.resizable(True, True)

        # Resize menu window to given static size
        elif container == Menu:
            self.geometry("220x320")
            self.resizable(False, False)

        frame.tkraise()

    def init_model(self, images_dir, labels_dir):
        # Initialize the model
        self.model = Model(images_dir, labels_dir)

    def get_file(self, index):
        if self.model is not None:
            return self.model[index]
        else:
            raise Exception("Model was not initialized")

    def start(self):
        cropper = self.frames[Cropper]
        cropper.config(0, len(self.model))
        cropper.load_current()


class Cropper(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller

        # Rectangle feature parameters
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None

        # For undo feature remember the original image
        self.image = None
        self.labels = None
        self.index = 0
        self.size = 0

        # For more efficient calculation complexity
        self.framed_image = None

        # Create a grid layout for the entire frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a frame for the image and label
        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0, sticky="nsew")

        # Create a label to display the image
        self.image_label = tk.Label(self.image_frame)  # Set a fixed size for the image
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_tk = None

        # Buttons at the bottom
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=1, column=0, columnspan=5, sticky="we", pady=3, padx=3)  # Span across all columns
        self.button_frame.rowconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)

        # Save button
        self.save_button = tk.Button(self.button_frame, text="Save Cropped Image", command=self.save, state=tk.DISABLED)
        self.save_button.grid(row=0, column=0, sticky="we")

        # Next button
        self.next_button = tk.Button(self.button_frame, text="Next Image", command=self.next)
        self.next_button.grid(row=0, column=1, sticky="we")

        # Undo button
        undo_button = tk.Button(self.button_frame, text="Undo", command=self.undo)
        undo_button.grid(row=0, column=2, sticky="we")

        # Restart button
        restart_button = tk.Button(self.button_frame, text="Restart", command=lambda: controller.show_frame(Menu))
        restart_button.grid(row=0, column=3, sticky="we")

        # Create a counter label to display the current image index
        self.index_label = tk.Label(self.button_frame, text=f"{self.index + 1}/{self.size}")
        self.index_label.grid(row=0, column=4, sticky="we")

        # Add sizegrip to bottom-right corner of frame
        self.sizegrip = ttk.Sizegrip(self.button_frame)
        self.sizegrip.grid(row=0, column=5, sticky="se")

        # Bind mouse events to the image label
        self.image_label.bind("<Button-1>", self.start_rect)
        self.image_label.bind("<B1-Motion>", self.draw_rect)
        self.image_label.bind("<ButtonRelease-1>", self.end_rect)
        self.image_label.bind("<Button-3>", self.delete)
        self.controller.bind("<Control-z>", self.undo, add="+")

    def load_current(self):
        image_path, label_path = self.controller.get_file(self.index)
        if image_path is not None and label_path is not None:

            # Load bayer image and transform to rgb
            self.image = Image.open(image_path)

            # Resize the image frame to fit the image
            self.image_frame.config(width=self.image.width, height=self.image.height)

            # Resize the main window to fit the image and control frame
            self.controller.update_idletasks()
            self.controller.geometry(
                f"{self.image.width}x"
                f"{self.image.height + 10 +
                   self.button_frame.winfo_height()}")

            # Load label into memory
            self.labels = read_label(label_path)

            # Plot new bounding boxes on a copy.
            framed_image = frame(np.array(self.image.copy()), self.labels, class_name)
            self.framed_image = Image.fromarray(framed_image)

            # Visualize cropped image
            self.image_tk = ImageTk.PhotoImage(self.framed_image)
            self.image_label.config(image=self.image_tk)
            self.save_button.config(state=tk.NORMAL)

            # Update the index label with the current image index
            self.index_label.config(text=f"{self.index + 1}/{self.size}")
        else:
            # Handle the end of the directory
            print("End of image directory.")

    def config(self, index, length):
        self.index = index
        self.size = length

    def start_rect(self, event):
        self.rect_start_x = event.x
        self.rect_start_y = event.y

    def draw_rect(self, event):
        self.rect_end_x = event.x
        self.rect_end_y = event.y

        # Reset the canvas
        image = self.framed_image.copy()
        draw = ImageDraw.Draw(image)
        rect_coords = (self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y)

        # Display rectangle
        draw.rectangle(rect_coords, outline="red", width=2)
        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk)

    def end_rect(self, event):
        # Calculate the cropping rectangle
        rect = (min(self.rect_start_x, self.rect_end_x),
                min(self.rect_start_y, self.rect_end_y),
                max(self.rect_start_x, self.rect_end_x),
                max(self.rect_start_y, self.rect_end_y))

        # Crop the labels using the rectangle
        image_array = np.array(self.image)
        _, cropped_labels = crop(image_array, self.labels, rect)

        # Crop the image using the rectangle coordinates
        cropped_image = self.image.crop(rect)

        # Store the cropped image
        self.image = cropped_image
        self.labels = cropped_labels

        # Plot new bounding boxes on the clean image
        framed_image = frame(np.array(self.image.copy()), self.labels, class_name)
        self.framed_image = Image.fromarray(framed_image)

        # Visualize cropped image
        self.image_tk = ImageTk.PhotoImage(self.framed_image)
        self.image_label.config(image=self.image_tk)

    def delete(self, event):
        # Delete label according to the event
        point = (event.x, event.y)
        image_array = np.array(self.image)
        self.labels = delete_labels(self.labels, point, image_array.shape)

        # Plot new bounding boxes on the clean image
        framed_image = frame(image_array, self.labels, class_name)
        self.framed_image = Image.fromarray(framed_image)

        # Visualize cropped image
        self.image_tk = ImageTk.PhotoImage(self.framed_image)
        self.image_label.config(image=self.image_tk)

    def save(self):
        # Save cropped image to file path.
        image_path, label_path = self.controller.get_file(self.index)
        self.image.save(image_path)
        write_labels(label_path, self.labels)

    def next(self):
        self.index += 1
        self.load_current()

    def undo(self, event=None):
        self.load_current()

    def config(self, index=None, size=None):
        self.index = index
        self.size = size


class Menu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller

        # Labels for directory paths
        self.image_dir_label = tk.Label(self, text="No directory selected", anchor="w", wraplength=200, justify="right")
        self.label_dir_label = tk.Label(self, text="No directory selected", anchor="w", wraplength=200, justify="right")

        # Buttons
        self.load_images_button = tk.Button(self, text="Load Images", width=22, command=self.load_image_directory)
        self.load_labels_button = tk.Button(self, text="Load Labels", width=22, command=self.load_label_directory)
        self.settings_button = tk.Button(self, text="Settings", state=tk.DISABLED)  # Placeholder for future settings
        self.start_button = tk.Button(self, text="Start", command=self.start_cropper, state=tk.DISABLED)

        # Layout
        self.load_images_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        self.image_dir_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        self.load_labels_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        self.label_dir_label.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        self.settings_button.grid(row=5, column=0, columnspan=1, sticky="news", padx=10, pady=5)
        self.start_button.grid(row=5, column=1, columnspan=1, padx=10, pady=5, sticky="news")

        # Center the menu frame on the screen
        self.place(relx=0.5, rely=0.5, anchor="center")

        # Variables
        self.images_dir = None
        self.labels_dir = None

    def load_image_directory(self):
        """Open a directory selection dialog and update the image directory label."""
        self.images_dir = filedialog.askdirectory()
        if os.path.isdir(self.images_dir):
            self.image_dir_label.config(text=f"Image Directory: {self.images_dir}")
            self.check_directories()
        else:
            raise Exception("Invalid image directory.")

    def load_label_directory(self):
        """Open a directory selection dialog and update the label directory label."""
        self.labels_dir = filedialog.askdirectory()
        if os.path.isdir(self.labels_dir):
           self.label_dir_label.config(text=f"Label Directory: {self.labels_dir}")
           self.check_directories()
        else:
            raise Exception("Invalid label directory.")

    def check_directories(self):
        """Check if both directories are loaded, enabling the Start button."""
        if self.images_dir and self.labels_dir:
            self.start_button.config(state=tk.NORMAL)  # Enable start button when both dirs are selected

    def start_cropper(self):
        """Initialize the model with the selected directories and switch to the Cropper frame."""
        if self.images_dir and self.labels_dir:
            self.controller.init_model(self.images_dir, self.labels_dir)
            self.controller.show_frame(Cropper)
            self.controller.start()
        else:
            print("Please select both image and label directories.")


class Model:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        # Check if image and labels directory exists
        assert os.path.isdir(self.images_dir) and os.path.isdir(self.labels_dir), \
            f"{images_dir} or {labels_dir} does not exist or is not a directory"

        self.images = os.listdir(self.images_dir)
        self.labels = os.listdir(self.labels_dir)

        # Sort the directories in alphanumeric order
        self.images.sort()
        self.labels.sort()

        # Check if they are both the same length
        assert len(self.images) == len(self.labels), \
            "Directories must have the same length"

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images[index])
        label_path = os.path.join(self.labels_dir, self.labels[index])
        return image_path, label_path


if __name__ == "__main__":
    app = Application()
    app.mainloop()