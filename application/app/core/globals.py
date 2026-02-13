import os
import sys
import queue
from app.core.settings import settings
from multiprocessing import Lock
from app.utilites.get_resource_path import get_resource_path

lock = Lock()

MOUSE_QUEUE = queue.Queue()

ALLOWABLE_ADD_DATA = 1

IS_PRESSED = 0
MAX_QUEUE_SIZE = 100000

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEATURES = [
    "dist",
    "speed",
    "jerk",
    "micro_shake",
    "curvature",
    "angle_vel",
    "low_speed_const_acc"
]

input_size = len(FEATURES) * 3
LAST_EVENT_TS:float = 0.0

MACRO_DETECTOR  = [] 

JsonPath = settings.JsonPath

threshold = settings.threshold
improvement_val_loss_cut = settings.improvement_val_loss_cut
chunk_size = settings.chunk_size

# model
d_model=settings.d_model
num_layers=settings.num_layers
dropout=settings.dropout
batch_size=settings.batch_size
lr=settings.lr
tolerance=settings.tolerance
n_head=settings.n_head

epoch = settings.epoch
patience = settings.patience
weight_decay = settings.weight_decay
dim_feedforward = settings.dim_feedforward

weight_threshold = settings.weight_threshold

CLIP_BOUNDS = settings.CLIP_BOUNDS

GLOBAL_CHANGE = False

LOG_QUEUE = None
CHART_DATA = None
TRAIN_DATA = None
INFERENCE_CHART_VIEW = None
PROCESS_LOCK = None

init_model_path = ""
init_scale_path = ""

save_path = get_resource_path(os.path.join("app", "models", "weights"))
plot_data_path = get_resource_path(os.path.join("app", "models", "weights", "ploat_data.json"))
scaler_path = get_resource_path(os.path.join("app", "models", "weights"))

def init_manager():
    global LOG_QUEUE
    global CHART_DATA
    global TRAIN_DATA
    global INFERENCE_CHART_VIEW
    global PROCESS_LOCK
    
    from multiprocessing import Manager
    manager = Manager()
    LOG_QUEUE = manager.Queue()
    CHART_DATA = manager.Queue()
    TRAIN_DATA = manager.Queue()
    INFERENCE_CHART_VIEW = manager.Value('b', False)
    PROCESS_LOCK = manager.Lock()
    