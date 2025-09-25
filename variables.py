import os
from threading import Thread
from dataset import find_extrema
from utils import OutputSmoother

DATA_DIR = os.path.join(os.getcwd(), "training_data")
print(DATA_DIR)
MODEL_LEFT_PATH = "model_left.pth"
MODEL_RIGHT_PATH = "model_right.pth"
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3

is_eye_tracking_running = False
found_models = os.path.exists(MODEL_LEFT_PATH) and os.path.exists(MODEL_RIGHT_PATH)

eye_x_scale = .6
eye_y_scale = .5
eye_smoothness = 0
blink_smoothness = .1
average_eye_gazes = False

x_max_abs, y_max_abs = find_extrema(DATA_DIR)

eye_tracking_thread: Thread = None

update_graph = False