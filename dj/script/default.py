import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

try:
    print("Detectron2 version:", detectron2.__version__)
except AttributeError:
    print("Detectron2 imported successfully")

