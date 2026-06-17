from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from speech_comprehension.phoneme_editor import (  # noqa: E402
    PhonemeEditorConfig,
    StreamingPhonemeEditor,
    phoneme_editor_loss,
)


def test_phoneme_editor_preserves_feature_shape() -> None:
    torch.manual_seed(0)
    config = PhonemeEditorConfig(
        acoustic_dim=16,
        num_phonemes=12,
        num_accents=3,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
    )
    model = StreamingPhonemeEditor(config)
    features = torch.randn(2, 7, 16)
    phonemes = torch.randint(0, 12, (2, 7))
    accents = torch.tensor([0, 2])

    output = model(features, phonemes, accents)

    assert output.edited_features.shape == features.shape
    assert output.residual.shape == features.shape
    assert output.edit_gate.shape == (2, 7, 1)
    assert torch.max(torch.abs(output.edited_features - features)) <= config.max_residual_scale


def test_phoneme_editor_loss_contains_training_terms() -> None:
    torch.manual_seed(0)
    config = PhonemeEditorConfig(
        acoustic_dim=8,
        num_phonemes=6,
        num_accents=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    model = StreamingPhonemeEditor(config)
    features = torch.randn(1, 5, 8)
    phonemes = torch.randint(0, 6, (1, 5))
    accents = torch.tensor([1])
    output = model(features, phonemes, accents)

    losses = phoneme_editor_loss(
        output,
        original_features=features,
        intelligibility_loss=torch.tensor(0.25),
        original_speaker_embedding=torch.randn(1, 4),
        edited_speaker_embedding=torch.randn(1, 4),
        valid_frame_mask=torch.tensor([[True, True, True, False, False]]),
    )

    assert set(losses) == {
        "total",
        "intelligibility",
        "speaker",
        "minimal_edit",
        "gate_sparsity",
        "residual_smoothness",
    }
    assert losses["total"].requires_grad
