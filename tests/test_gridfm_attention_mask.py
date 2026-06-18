import torch

from fastwam.models.wan22.fastwam import FastWAM


class _DummyVideoExpert:
    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        del video_tokens_per_frame
        return torch.tril(torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device))


def test_gridfm_action_does_not_attend_grid_tokens():
    model = FastWAM.__new__(FastWAM)
    model.video_expert = _DummyVideoExpert()
    model.grid_flow_num_windows = 2

    video_seq_len = 6
    grid_seq_len = 8
    action_seq_len = 4
    mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=2,
        device=torch.device("cpu"),
        grid_seq_len=grid_seq_len,
    )

    grid_start = video_seq_len
    action_start = video_seq_len + grid_seq_len

    assert mask.shape == (video_seq_len + grid_seq_len + action_seq_len,) * 2
    assert not mask[action_start:, grid_start:action_start].any()
    assert mask[action_start:, :video_seq_len].any()
    assert mask[:video_seq_len, grid_start:action_start].any()
    assert mask[grid_start:action_start, :video_seq_len].any()
