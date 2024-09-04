import argparse
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.visualization import vis, vis_count, draw_poly
from utils.utils import xyxy_to_tlwh, count, create_mask, isValidCar
from tensorrt_yolo.utils import preproc, BaseEngine
from bytetrack.byte_tracker import BYTETracker # type: ignore


# tracker_model = 'ByteTrack'
tracker_model = 'DeepSort'

track_thresh = 0.4
match_thresh = 0.7
track_buffer = 20
detection_area_scales =[(0.05, 0.9),(0.3, 0.5),(0.7,0.5),(0.95, 0.9)]
mot20 = False

args = argparse.Namespace(track_thresh=track_thresh, track_buffer=track_buffer, mot20=mot20, match_thresh = match_thresh)
tracker = DeepSort(max_age=5)
byte_tracker = BYTETracker(args)


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
        import time

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            #Calculate and add FPS value to frame
            t1 = time.time()
            data = self.infer(blob)

            poly = create_mask(frame, detection_area_scales)
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            
            #Apply nms to detections
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

                    #Define if detection valid or not
                    valid, in_area = isValidCar(frame.shape, coor, poly)
                    if not valid or not in_area:
                        continue

                    #Append valid detections to lists
                    bboxes.append([dets[i, :6]])
                    bbs.append([coor, int(dets[i, 4]*1000)/1000, dets[i, 5]])

                # Select the model and update tracks
                if tracker_model == 'ByteTrack':
                        ids, locations, class_ids, scores = trackBT(bboxes, frame)
                else:
                    try:
                        ids, locations, class_ids, scores = trackDS(bbs, frame)
                    except ValueError:
                        print('Library Error :matrix contains invalid numeric entries')
                        continue

                #?Counting
                in_active, out_active, in_count, out_count = count(ids, locations, frame.shape, in_counter, out_counter, observed, in_count, out_count)
                
                #Visualizing the frame and polygon
                frame = draw_poly(frame, poly)
                frame = vis(frame, locations, scores, class_ids , conf, ids = ids, class_names=self.class_names)
                frame = vis_count(frame, in_active, out_active, in_count, out_count)

                fps = (fps + (1. / (time.time() - t1))) / 2
                frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
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



if __name__ == '__main__':
    pred = Predictor(engine_path='yolov5_vehicle.engine')
    pred.get_fps()
    pred.detect_video("video.mp4", conf=0., end2end=False)