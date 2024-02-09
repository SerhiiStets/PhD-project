import numpy as np
import matplotlib.pyplot as plt
from cnn_model import CNNModel 

class PlotCreator:

    def __init__(self, model: CNNModel) -> None:
        self.model  = model

    def plot_cnn_performance(self) -> None:
        """Plots loss and accuracy graphs"""
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the loss and validation loss in the first subplot
        axs[0].plot(self.model.history.history.get("loss"), label="training loss")
        axs[0].plot(self.model.history.history.get("val_loss"), label="validation loss")
        axs[0].set_title("Loss")
        axs[0].legend()

        # Plot the accuracy and validation accuracy in the second subplot
        axs[1].plot(self.model.history.history.get("accuracy"), label="training accuracy")
        axs[1].plot(self.model.history.history.get("val_accuracy"), label="validation accuracy")
        axs[1].set_title("Accuracy")
        axs[1].legend()

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()

    def plot_incorrect_indices(self) -> None:
        """Plots all images that were predicted incorrectly"""
        num_images = len(self.model.incorrect_indices)
        # Calculate grid dimensions based on the number of images
        rows = int(np.sqrt(num_images))
        cols = int(np.ceil(num_images / rows))
        # Create a subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12), gridspec_kw={"hspace": 0.5, "wspace": 0.3})
        # Iterate through incorrect indices and display images in the grid
        for i, idx in enumerate(self.model.incorrect_indices):
            # If there are a lot of bad predictions we don't 
            # want to show thousands of photos
            if idx > 30:
                break

            img = self.model.test_data[idx]
            ax = axes[i // cols, i % cols]
            ax.imshow(img)
            ax.set_title(
                f"Actual Label: {self.model.test_labels[idx]}, Predicted Label: {self.model.predicted_test_labels[idx]}"
            )
            ax.axis("off")

        # Adjust layout and show the grid
        plt.tight_layout()
        plt.show()

