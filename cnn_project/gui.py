import tkinter as tk
from tkinter import ttk
from cnn_model import CNNModel
from data_loader import DataLoader
from plot_creator import PlotCreator
from object_detection import sliding_window_search
from typing import Union

class CNNRunner:
    def __init__(self) -> None:
        # Create and display entry fields with default values
        self.cars_image_dir = tk.StringVar()
        self.not_cars_image_dir = tk.StringVar()
        self.object_detection_test_image_dir = tk.StringVar()
        self.num_epochs = tk.IntVar()
        self.show_performance_graph = tk.BooleanVar(value=True)
        self.show_incorrect_images = tk.BooleanVar(value=True)
        self.show_sliding_window = tk.BooleanVar(value=True)
        self.save_model_checkbox_var = tk.BooleanVar(value=False)
        self.save_model_entry_var = tk.StringVar()

        # Set default values
        self.cars_image_dir.set("./data/cars/")
        self.not_cars_image_dir.set("./data/not_cars/")
        self.object_detection_test_image_dir.set("./data/object_detection/")
        self.save_model_entry_var.set("./")
        self.num_epochs.set(10)

    def run_cnn(self):
        data_loader = DataLoader(self.cars_image_dir.get(), self.not_cars_image_dir.get())
        cnn_model = CNNModel()

        train_data, train_labels = data_loader.load_images()
        train_data = data_loader.normalize_images_for_cnn(train_data)

        cnn_model.run_model(train_data, train_labels)
        cnn_model.prediction(show_incorrect_indices=True)

        plot_creator = PlotCreator(cnn_model)
        if self.show_performance_graph.get():
            plot_creator.plot_cnn_performance()
        if self.show_incorrect_images.get():
            plot_creator.plot_incorrect_indices()

        # model.load_without_training("model.h5")
        if self.show_sliding_window:
            sliding_window_search(self.object_detection_test_image_dir.get(), cnn_model.cnn)

        if self.save_model_checkbox_var.get():
            cnn_model.save_model(self.save_model_entry_var.get())


class CNNGui(CNNRunner):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("CNN Model Settings")

        self.create_widgets()

    def toggle_save_model_path(self):
        if self.save_model_checkbox_var.get():
            self.save_model_entry.config(state="normal")
        else:
            self.save_model_entry.config(state="disabled")
        
    def create_widgets(self):
        # Create and display labels
        ttk.Label(self.root, text="Cars Image Directory:").grid(row=0, column=0, padx=10, pady=5)
        ttk.Label(self.root, text="Not Cars Image Directory:").grid(row=1, column=0, padx=10, pady=5)
        ttk.Label(self.root, text="Object Detection Test Image Directory:").grid(row=2, column=0, padx=10, pady=5)
        ttk.Label(self.root, text="Number of Epochs:").grid(row=3, column=0, padx=10, pady=5)

        ttk.Entry(self.root, textvariable=self.cars_image_dir, width=30).grid(row=0, column=1, padx=10, pady=5)
        ttk.Entry(self.root, textvariable=self.not_cars_image_dir, width=30).grid(row=1, column=1, padx=10, pady=5)
        ttk.Entry(self.root, textvariable=self.object_detection_test_image_dir, width=30).grid(row=2, column=1, padx=10, pady=5)
        ttk.Entry(self.root, textvariable=self.num_epochs, width=10).grid(row=3, column=1, padx=10, pady=5)
        ttk.Checkbutton(self.root, text="Show CNN Performance Graph", variable=self.show_performance_graph).grid(row=4, columnspan=2, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(self.root, text="Show Incorrectly Predicted Images", variable=self.show_incorrect_images).grid(row=5, columnspan=2, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(self.root, text="Show Sliding Window Search", variable=self.show_sliding_window).grid(row=6, columnspan=2, padx=10, pady=5, sticky="w")

        ttk.Checkbutton(self.root, text="Save CNN model", variable=self.save_model_checkbox_var, command=self.toggle_save_model_path).grid(row=7, columnspan=2, padx=10, pady=5, sticky="w")

        ttk.Label(self.root, text="Save model path:").grid(row=8, columnspan=2, padx=10, pady=5, sticky="w")



        self.save_model_entry = ttk.Entry(self.root, textvariable=self.save_model_entry_var, state="disabled")

        self.toggle_save_model_path()  # Call the function initially to set the state correctly
        self.save_model_entry.grid(row=8, column=1, columnspan=2, padx=10, pady=5, sticky="w")



        # Add a button to start CNN training
        ttk.Button(self.root, text="Start Training", command=self.run_cnn).grid(row=9, columnspan=2, pady=10)
