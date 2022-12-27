import torch


def tile_and_concatenate(nchw, nch=None, ncw=None, nc=None):
    """
    Expand and concatenate tensors of shape
    `[N, C1, H, W], [N, C2, H], [N, C3, W], [N, C4]`,
    with possible omission,
    into a tensor of shape `[N, C1+C2+C3+C4, H, W]`.

    Parameters
    ----------
    nchw : torch.Tensor or None

    nch : torch.Tensor or None

    ncw : torch.Tensor or None

    nc : torch.Tensor or None

    Returns
    -------
    torch.Tensor
    """
    h, w = nchw.shape[-2:]
    tensor_list = [nchw]
    if nch is not None:
        # rearrange and repeat: n c h -> n c h 1 -> n c h w
        tensor_list.append(nch.unsqueeze(3).expand(-1, -1, -1, w))
    if ncw is not None:
        # rearrange and repeat: n c w -> n c 1 w -> n c h w
        tensor_list.append(ncw.unsqueeze(2).expand(-1, -1, h, -1))
    if nc is not None:
        # rearrange and repeat: n c -> n c 1 1 -> n c h w
        tensor_list.append(nc.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w))
    return torch.cat(tensor_list, dim=1)

