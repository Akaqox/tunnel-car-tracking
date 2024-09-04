import os
import sys
import argparse
import importlib
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

tracker_model = 'ByteTrack'
# tracker_model = 'DeepSort'

from utils.utils import preproc, rainbow_fill, BaseEngine
from utils.byte_tracker import BYTETracker # type: ignore

track_thresh = 0.5
match_thresh = 0.8
track_buffer = 20
mot20 = False

args = argparse.Namespace(track_thresh=track_thresh, track_buffer=track_buffer, mot20=mot20, match_thresh = match_thresh)


tracker = DeepSort(max_age=5)
byte_tracker = BYTETracker(args)


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
        in_count = 0
        out_count = 0
        observed = []
        in_counter = []
        out_counter = []

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
            poly = create_mask(frame)
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)
            if dets is not None:
                bbs = []
                bboxes = []
                for i in range(dets.shape[0]):
                    if dets[i, 4]  < conf:
                        continue
                    if dets[i, 5] != 0.:
                         continue
                    xywh = xyxy_to_tlwh(dets[i,:4])
                    coor = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])
                    
                    valid, in_area = isValidCar(frame.shape, coor, poly)

                    if not valid or not in_area:
                        continue
                    bboxes.append([dets[i, :6]])
                    bbs.append([coor, int(dets[i, 4]*1000)/1000, dets[i, 5]])
                if tracker_model == 'ByteTrack':
                        ids, locations, class_ids, scores = trackBT(bboxes, frame)
                        
                else:
                    try:
                        ids, locations, class_ids, scores = trackDS(bbs, frame)
                    except ValueError:
                        print('Library Error :matrix contains invalid numeric entries')
                        continue
                int_coords = np.array(poly.exterior.coords, np.int32)
                int_coords = int_coords.reshape((-1, 1, 2))
                # Draw the interested area with polygon
                cv2.polylines(frame, [int_coords], isClosed=True, color=(0, 255, 0), thickness=2)

                frame = vis(frame, locations, scores, class_ids , conf, ids = ids, class_names=self.class_names)
                
                in_active, out_active, in_count, out_count = count(ids, locations, frame.shape, in_counter, out_counter, observed, in_count, out_count)

                frame = vis_count(frame, in_active, out_active, in_count, out_count)
                
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()


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

    # mask, _ = create_mask(resized_img)
    # resized_img = delete_mask(resized_img, mask)

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
    # plt.imshow(image[0])
    # plt.show()
    return image, r


def trackBT(bboxes, frame):
    ids = [] 
    locations = []
    class_ids = []
    scores = []

    output = np.zeros((len(bboxes), 5))
    for i in range(len(bboxes)):
        class_id = bboxes[i][0][-1]
        for j in range(5):
            output[i][j] = bboxes[i][0][j]

    

    stracks = byte_tracker.update(output, frame.shape, frame.shape)
    for strack in stracks:
        tlbr = strack.tlbr
        score = strack.score

        track_id = str(strack.track_id)
        ids.append(track_id)
        locations.append(tlbr)
        class_ids.append(class_id)
        scores.append(score)
    return ids, locations, class_ids, scores


def trackDS(bbs, frame):
    tracks = tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    ids = [] 
    locations = []
    class_ids = []
    scores = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        score = track.get_det_conf()
        class_id = track.get_det_class()

        track_id = track.track_id
        ids.append(track_id)
        locations.append(ltrb)
        class_ids.append(class_id)
        scores.append(score)
    return ids, locations, class_ids, scores


def isValidCar(frame_size:tuple, coor:tuple, poly):
    valid = False
    isInside = False
    upper_limit = 0.35
    down_limit = 0.05
    width = coor[2]
    height = coor[3]
    center = (coor[0]+ coor[2]/2, coor[1]+ coor[3]/2)
    point = Point(center)
    isInside = poly.contains(point)

    if int(frame_size[0] * upper_limit) > width and int(frame_size[0] * down_limit) < width:
        valid = 1
        return valid, isInside
    if int(frame_size[1] * upper_limit) > height and int(frame_size[1] * down_limit) < height:
        valid = 1
        return valid, isInside
    return valid, isInside
    
def create_mask(im):
    #draw polygon
    poly = Polygon([(int(im.shape[1]*0.05), im.shape[0]*0.9), (int(im.shape[1]*0.3), int(im.shape[0]*0.5)), (int(im.shape[1]*0.7), int(im.shape[0]*0.5)), (int(im.shape[1]*0.95), im.shape[0] * 0.9)])
    return  poly

# def delete_mask(im, mask):
#     for i in range(im.shape[2]):
#         im[:,:,i] = im[:,:,i] * mask
#         im[im == 0.0] = 0.1
#     return im

def count(ids, locs, size, in_counter, out_counter, observed, in_count, out_count):
    in_active = 0
    out_active = 0
    memory_thres = 12
    for id, loc in zip(ids, locs):
       if loc[0] < size[1]/2:
            in_active += 1  # Increment in_active
            if id in observed:
                continue
            else:
                if in_counter.count(id) < memory_thres:
                    in_counter.append(id)
                else:
                    observed.append(id)
                    in_count += 1  # Increment in_count
                    in_counter = list(filter(lambda a: a != id, in_counter))

       else:
            out_active += 1  # Increment out_active
            if id in observed:
                continue
            else:
                if out_counter.count(id) < memory_thres:
                    out_counter.append(id)
                else:
                    observed.append(id)
                    out_counter = list(filter(lambda a: a != id, out_counter))
                    out_count += 1  # Increment out_count

    return in_active, out_active, in_count, out_count

def vis_count(frame, in_active, out_active, in_count, out_count):
    frame = cv2.putText(frame, "TOTAL IN:%d " %in_count, (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
    frame = cv2.putText(frame, "ACTIVE:%d " %in_active, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
    frame = cv2.putText(frame, "TOTAL OUT:%d "%out_count, (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
    frame = cv2.putText(frame, "ACTIVE:%d " %out_active, (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)    
    return frame


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


def xyxy_to_tlwh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    x1 = xyxy[0]
    y1 = xyxy[1] 
    return np.array([int(x1), int(y1), int(w_temp), int(h_temp)])

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


if __name__ == '__main__':
    pred = Predictor(engine_path='yolov5_vehicle.engine')
    pred.get_fps()
    pred.detect_video("video.mp4", conf=0., end2end=False)