import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _triple

from ops_3d import torch_op_3d


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, sampling_ratio):
        # input: [n, c, h, w, d]
        # rois columns: bi, x1, y1, x2, y2, z1, z2
        # output_size: pd,ph,pw
        ctx.save_for_backward(roi)
        ctx.output_size = _triple(output_size)
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = torch_op_3d.roi_align_forward(
            input, roi, output_size[1], output_size[2], output_size[0], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w, d = ctx.input_shape
        grad_input = torch_op_3d.roi_align_backward(
            grad_output,
            rois,
            output_size[1],
            output_size[2],
            output_size[0],
            bs,
            ch,
            h,
            w,
            d,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply

class ROIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): d, h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
    """

    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=0):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        '''
        input: tensor with shape of N*C*D*H*W
        rois: tensor with shape of n*7, columns of [batch_id, z1, y1, x1, z2, y2, x2]
        '''
        # convert data from N*C*D*H*W to N*C*H*W*D
        input = input.permute(0,1,3,4,2)
        # normalize rois coord
        # zyx
        if not isinstance(self.spatial_scale, (tuple, list)):
            self.spatial_scale = torch.tensor([self.spatial_scale, self.spatial_scale, self.spatial_scale])
        rois[:,1:4] = rois[:,1:4]*self.spatial_scale
        rois[:,4:7] = rois[:,4:7]*self.spatial_scale
        # convert rois columns from [batch_id, z1, y1, x1, z2, y2, x2] to [batch_id(0), x1(3), y1(2), x2(6), y2(5), z1(1), z2(4)]
        rois = cat([rois[:,0:1],rois[:,3:4],rois[:,2:3],rois[:,6:7],rois[:,5:6],rois[:,1:2],rois[:,4:5]], dim=1)  
        return roi_align(
            input, rois, self.output_size, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr


class ROIAlignPooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    """

    def __init__(self, output_size, scales, sampling_ratio=2):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(ROIAlignPooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[bboxes]): boxes to be used to perform the pooling operation, bboxes with columns: z1y1x1z2y2x2.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        result_size = [num_rois, num_channels]
        result_size.extend(_triple(self.output_size))
        result_size = tuple(result_size)

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            result_size,
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)

        return result


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


def box_area(box, mode="zyxzyx"):
    if mode == "zyxzyx":
        TO_REMOVE = 1
        area = (box[:, 3] - box[:, 0] + TO_REMOVE) *(box[:, 4] - box[:, 1] + TO_REMOVE) * (box[:, 5] - box[:, 2] + TO_REMOVE)
    elif mode == "zyxd":
        area = box[:, 3] * box[:, 4] * box[:, 5]
    else:
        raise RuntimeError("Should not be here")

    return area


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)