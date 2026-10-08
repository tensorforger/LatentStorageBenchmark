import torch


def pad_frames_last(video: torch.Tensor, step: int, offset: int) -> torch.Tensor:
    """
    Repeat the last frame so that the frame count T satisfies (T - offset) % step == 0.
    args:
        video: [B, C, T, H, W]
    returns:
        video: [B, C, T', H, W], T' >= T
    """
    pad = (offset - video.shape[2]) % step
    if pad == 0:
        return video
    return torch.cat([video, video[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
