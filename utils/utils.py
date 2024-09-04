import numpy as np
from shapely.geometry import Polygon, Point

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


def create_mask(im):
    #draw polygon
    poly = Polygon([(int(im.shape[1]*0.05), im.shape[0]*0.9), (int(im.shape[1]*0.3), int(im.shape[0]*0.5)), (int(im.shape[1]*0.7), int(im.shape[0]*0.5)), (int(im.shape[1]*0.95), im.shape[0] * 0.9)])
    return  poly

# def delete_mask(im, mask):
#     for i in range(im.shape[2]):
#         im[:,:,i] = im[:,:,i] * mask
#         im[im == 0.0] = 0.1
#     return im
def isValidCar(frame_size:tuple, coor:tuple, poly)->bool:
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
    
def count(ids:list, locs:list, size:int, in_counter:list, out_counter:list, observed:list, in_count:int, out_count:int) -> tuple:
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