import torch
import numpy as np

from ops_2d import torch_op_2d
from ops_3d import torch_op_3d

_nms_2d = torch_op_2d.nms
_nms_3d = torch_op_3d.nms


def nms_2d(dets, nms_thresh, max_count=-1):
    """
    dets has to be a tensor
    Args:
        dets(Tensor): [x1, y1, x2, y2, score] mode boxes and score, use absolute coordinates(not support relative coordinates),
            shape is (n, 5)
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    boxes = dets[:, :-1]
    scores = dets[:, -1]
    keep = _nms_2d(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep


def nms_3d(dets, nms_thresh, max_count=-1):
    """
    Args:
        dets(nd-array): [x, y, z, dx, dy, dz, score] mode boxes and score, use absolute coordinates(not support relative coordinates),
            shape is (n, 7), dets has to be a float tensor
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    x, y, z, dx, dy, dz, score = np.split(dets, 7, axis = 1)
    x1, x2 = x - dx / 2, x + dx / 2
    y1, y2 = y - dy / 2, y + dy / 2
    z1, z2 = z - dz / 2, z + dz / 2

    bboxes = np.concatenate([x1, y1, x2, y2, z1, z2, score], axis=1)
    bboxes = torch.from_numpy(bboxes).cuda().float()
    boxes = bboxes[:, :-1]
    scores = bboxes[:, -1]
    # dets(tensor): [x1, y1, x2, y2, z1, z2, score].
    keep = _nms_3d(boxes, scores, nms_thresh)
    keep = keep.cpu().numpy()
    if max_count > 0:
        keep = keep[:max_count]

    return dets[keep]



