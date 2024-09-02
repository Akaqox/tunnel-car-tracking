import os
import sys
import importlib
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Polygon
import rasterio.features
import matplotlib.pyplot as plt

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TensorRT-For-YOLO-Series"))

from utils.utils import preproc, rainbow_fill, BaseEngine




tracker = DeepSort(max_age=10)
_COLORS = rainbow_fill(5).astype(np.float32).reshape(-1, 3)

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 5 # your model classes
        self.class_names = ['Car', 'Motorcycle', 'Truck', 'Bus',' Bicycle' ]
        
    def detect_video(self, video_path, conf=0.5, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
        fps = 0
        track_id = None
        import time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)
            if dets is not None:
                bbs = []
                for i in range(dets.shape[0]):
                    if dets[i, 4]  < 0.5:
                        continue
                    if dets[i, 5] != 0.:
                         continue
                    xywh = xyxy_to_xywh(dets[i,:4])
                    coor = (int(dets[i,0]), int(dets[i,1]), xywh[2], xywh[3])

                    if not isValidCar(frame.shape, coor):
                        continue

                    # bbs.append([tuple(xywh), dets[i, 4], dets[i, 5]])
                    bbs.append([coor, dets[i, 4], dets[i, 5]])
                print(bbs)
                tracks = tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
                ids = []
                locations = []
                class_ids = []
                scores = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ids.append(track_id)
                    ltrb = track.to_ltrb()         
                    locations.append(ltrb)
                    class_id = track.get_det_class()
                    class_ids.append(class_id)
                    score = track.get_det_conf()
                    scores.append(score)


                frame = vis(frame, locations, scores, class_ids , conf, ids = ids, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()



def isValidCar(frame_size:tuple, coor:tuple):
    valid = False
    width = coor[2]
    height = coor[3]
    if frame_size[1]*0.25 > width and frame_size[1]*0.08 < width:
        valid = 1
        return valid
    if frame_size[2]*0.25 > height and frame_size[2]*0.08 < height:
        valid = 1
        return valid
    return valid
    
def create_mask(im):
    #draw polygon
    poly = Polygon([(0, im.shape[0]), (int(im.shape[1]*0.438), int(im.shape[0]*0.28)), (int(im.shape[1]*0.563), int(im.shape[0]*0.28)), (im.shape[1], im.shape[0])])
    
    mask = rasterio.features.rasterize([poly], out_shape=(im.shape[0], im.shape[1]))
    return mask

def delete_mask(im, mask):
    for i in range(im.shape[2]):
        im[:,:,i] = im[:,:,i] * mask
    return im

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    mask = create_mask(resized_img)
    resized_img = delete_mask(resized_img, mask)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    image = padded_img
    return image, r


def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])

def xywh_to_xyxy(xywh):
    """
    Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).
    :param xywh: [X, Y, W, H]
    :return: [X1, Y1, X2, Y2]
    """
    if np.array(xywh).ndim > 1 or len(xywh) > 4:
        raise ValueError('xywh format: [x1, y1, width, height]')
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return np.array([int(x1), int(y1), int(x2), int(y2)])

def vis(img, boxes, scores, cls_ids, conf=0.5, ids = [], class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]

        cls_id = int(cls_ids[i])
        score = scores[i]
        if score == None:
            continue
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)

        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        if ids != [] and not len(ids) <= i:
            text = ' ID=' + ids[i]
            cv2.putText(img, text, (x0 , y0 - txt_size[1]*2), font, 0.4, txt_color, thickness=1)
    return img


if __name__ == '__main__':
    pred = Predictor(engine_path='yolov5_vehicle.engine')
    pred.get_fps()
    pred.detect_video("video.mp4", conf=0.5, end2end=False)