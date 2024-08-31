import os
import sys
import importlib


# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TensorRT-For-YOLO-Series"))

from utils.utils import preproc, vis, BaseEngine


class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80 # your model classes

if __name__ == '__main__':
    pred = Predictor(engine_path='yolov7x.engine')
    pred.get_fps()
    pred.detect_video("video.mp4", conf=0.1, end2end=False)