import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.model.building_blocks import QueryGRUEncoder, VideoSelfAttentionEncoder, PositionwiseFeedForward,\
    QueryVideoCrossModalEncoder
from src.utils.utils import sliding_window


def forward_train_val(self, batch):
    """ The "Gaussian Alignment Module", use in training.
    Returns:
        loss: single item tensor
    """
    batch = self._prepare_batch(batch)
    sentence_feature, video_feature, attn_weights = self.network_forward(batch)

    def get_gaussian_weight(video_mask, glance_frame):
        """ Get the Gaussian weight of full video feature.
        Args:
            video_mask: (B, L)
            glance_frame: (B)
        Returns:
            weight: (B, L)
        """
        B, L = video_mask.shape

        x = torch.linspace(-1, 1, steps=L, device=self.device).view(1, L).expand(B, L)
        lengths = torch.sum(video_mask, dim=1).to(torch.long)

        # normalize video lengths into range
        sig = lengths / L
        sig = sig.view(B, 1)
        sig *= self.sigma_factor

        # normalize glance frames into range
        u = (glance_frame / L) * 2 - 1
        u = u.view(B, 1)

        weight = torch.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
        weight /= torch.max(weight, dim=1, keepdim=True)[0]  # normalize weight
        weight.masked_fill_(video_mask == 0.0, 0.0)
        return weight

    video_mask = batch["video_mask"]
    glance_frame = batch["glance_frame"]
    weight = get_gaussian_weight(video_mask, glance_frame)  # (B, L)

    # sliding window
    def slice(video_feature, video_mask, weight):
        """ We use the scheme "variational clip frame, fixed stride".
        Args:
            video_feature: (B, L, dim)
            video_mask: (B, L)
            weight: (B, L)
        Returns:
            clips: (B, N, dim)
            clip_masks: (B, N)
            clip_weights: (B, N)
        """
        video_feature = video_feature.masked_fill(video_mask.unsqueeze(2) == 0.0, -torch.inf)
        clips, clip_masks, clip_weights = [], [], []
        for clip_frame in self.clip_frames:
            temp, idx = sliding_window(video_feature, clip_frame, self.stride, dim=1)
            temp = torch.stack([self.pooling(x, dim=1) for x in temp], dim=1)  # (B, N, dim)
            temp_mask = video_mask[:, idx[:, 0]]  # (B, N)
            temp.masked_fill_(temp_mask.unsqueeze(2) == 0.0, 0.0)
            temp_weight = weight[:, torch.div(idx[:, 0] + idx[:, 1], 2.0, rounding_mode='floor').to(torch.long)]  # (B, N)
            clips.append(temp)
            clip_masks.append(temp_mask)
            clip_weights.append(temp_weight)
        clips = torch.cat(clips, dim=1)
        clip_masks = torch.cat(clip_masks, dim=1)
        clip_weights = torch.cat(clip_weights, dim=1)
        return clips, clip_masks, clip_weights

    clips, clip_masks, clip_weights = slice(video_feature, video_mask, weight)
    scores = torch.matmul(clips, sentence_feature.T.unsqueeze(0))  # (B, N, B)

    # loss
    B, N, _ = scores.shape
    label = torch.zeros(B, N, B, device=self.device)
    for i in range(B):
        label[i, :, i] = clip_weights[i, :]
        label[i, :, list(range(i)) + list(range(i + 1, B))] = ((1 - clip_weights[i, :]) / (B - 1)).unsqueeze(1)
    label.masked_fill_(clip_masks.unsqueeze(2) == 0.0, 0.0)

    nce_loss = self.nce_loss(scores.view(B * N, B) / self.temp, label.view(B * N, B))
    nce_loss = torch.sum(nce_loss) / torch.sum(clip_masks)  # masked mean

    attn_loss = F.kl_div(F.log_softmax(attn_weights, dim=1), F.log_softmax(weight, dim=1), reduction="none", log_target=True)
    attn_loss.masked_fill_(video_mask == 0.0, 0.0)
    attn_loss = torch.sum(attn_loss) / torch.sum(video_mask) * 10000

    loss = nce_loss + attn_loss
    return loss