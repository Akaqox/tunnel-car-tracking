import cv2
from tensorrt_yolo.utils import rainbow_fill
import numpy as np


def vis_count(frame, in_active:int, out_active:int, in_count:int, out_count:int):
    frame = cv2.putText(frame, "TOTAL IN:%d " %in_count, (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
    frame = cv2.putText(frame, "ACTIVE:%d " %in_active, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
    frame = cv2.putText(frame, "TOTAL OUT:%d "%out_count, (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
    frame = cv2.putText(frame, "ACTIVE:%d " %out_active, (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)    
    return frame


def vis(img, boxes:list, scores:list, cls_ids:list, conf=0.5, ids:list = [], class_names:list = None, nc = 5):
    _COLORS = rainbow_fill(nc).astype(np.float32).reshape(-1, 3)
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