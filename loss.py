def intersection_over_union(Y,Z):
    iou = (torch.sum(torch.min(Y, Z)))/(torch.sum(torch.max(Y, Z)))
    return iou
