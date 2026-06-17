from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from speech_comprehension.codec_editor import (  # noqa: E402
    CodecLatentEditorConfig,
    CodecLatentEditorOutput,
    LatentListenerProxy,
    LatentListenerProxyConfig,
    StreamingCodecLatentEditor,
    codec_editor_loss,
    freeze_module,
    listener_proxy_loss,
)


def test_codec_editor_edits_latents_with_bounded_residual() -> None:
    torch.manual_seed(0)
    config = CodecLatentEditorConfig(
        latent_dim=12,
        num_accents=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        max_residual_scale=0.1,
    )
    model = StreamingCodecLatentEditor(config)
    latents = torch.randn(2, 9, 12)
    accent_ids = torch.tensor([1, 3])

    output = model(latents, accent_ids)

    assert output.edited_latents.shape == latents.shape
    assert output.residual.shape == latents.shape
    assert output.edit_gate.shape == (2, 9, 1)
    assert torch.max(torch.abs(output.edited_latents - latents)) <= config.max_residual_scale


def test_listener_proxy_predicts_confidence_like_score() -> None:
    torch.manual_seed(0)
    proxy = LatentListenerProxy(
        LatentListenerProxyConfig(
            latent_dim=10,
            hidden_dim=20,
            num_layers=1,
            num_heads=4,
        )
    )
    latents = torch.randn(3, 6, 10)
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [False, False, False, False, True, True],
        ]
    )

    score = proxy(latents, padding_mask=padding_mask)
    loss = listener_proxy_loss(score, torch.tensor([0.9, 0.2, 0.5]))

    assert score.shape == (3,)
    assert torch.all(score >= 0.0)
    assert torch.all(score <= 1.0)
    assert loss.requires_grad


def test_codec_editor_loss_rewards_listener_improvement() -> None:
    latents = torch.zeros(2, 5, 8)
    output = CodecLatentEditorOutput(
        edited_latents=torch.full_like(latents, 0.01),
        residual=torch.full_like(latents, 0.02),
        edit_gate=torch.full((2, 5, 1), 0.5),
    )
    better = codec_editor_loss(
        output,
        original_latents=latents,
        original_listener_score=torch.tensor([0.40, 0.60]),
        edited_listener_score=torch.tensor([0.55, 0.70]),
    )
    worse = codec_editor_loss(
        output,
        original_latents=latents,
        original_listener_score=torch.tensor([0.40, 0.60]),
        edited_listener_score=torch.tensor([0.39, 0.59]),
    )

    assert better["listener_preference"] < worse["listener_preference"]
    assert better["minimal_edit"] > 0.0
    assert better["gate_sparsity"] == 0.5


def test_freeze_module_disables_listener_proxy_gradients() -> None:
    proxy = LatentListenerProxy(
        LatentListenerProxyConfig(
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
        )
    )

    freeze_module(proxy)

    assert not proxy.training
    assert all(not parameter.requires_grad for parameter in proxy.parameters())
