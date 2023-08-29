import logging
import tkinter as tk

from gui import CNNGui
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler()],
)


if __name__ == "__main__":
    # TODO grapth of accuracy
    # похибка для трейн і тест графік
    parser = argparse.ArgumentParser(description="CNN Model Settings")
    parser.add_argument("mode", choices=["gui", "cli"], help="Run in GUI or CLI mode")

    args = parser.parse_args()

    if args.mode == "gui":
        root = tk.Tk()
        app = CNNGui(root)
        root.mainloop()
    else:
        car_images_dir: str = "./data/cars/"
        not_car_images_dir: str = "./data/not_cars/"
        object_detection_dir = "./data/object_detection/"
        # run_cnn(car_images_dir, not_car_images_dir, object_detection_dir)

