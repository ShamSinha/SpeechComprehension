from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise RuntimeError(
        "The phoneme editor requires PyTorch. Install torch before using this module."
    ) from exc


@dataclass(frozen=True)
class PhonemeEditorConfig:
    acoustic_dim: int
    num_phonemes: int
    num_accents: int
    hidden_dim: int = 256
    phoneme_dim: int = 64
    accent_dim: int = 32
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_residual_scale: float = 0.25


@dataclass(frozen=True)
class PhonemeEditorOutput:
    edited_features: Tensor
    residual: Tensor
    edit_gate: Tensor


class StreamingPhonemeEditor(nn.Module):
    """Trainable sparse phoneme-conditioned feature editor.

    This module edits acoustic features, not phoneme labels. A phoneme alignment
    tells the model which sound is being spoken at each frame, and the model
    learns whether that frame needs a small residual repair.
    """

    def __init__(self, config: PhonemeEditorConfig) -> None:
        super().__init__()
        self.config = config
        self.acoustic_projection = nn.Linear(config.acoustic_dim, config.hidden_dim)
        self.phoneme_embedding = nn.Embedding(config.num_phonemes, config.phoneme_dim)
        self.accent_embedding = nn.Embedding(config.num_accents, config.accent_dim)
        self.condition_projection = nn.Linear(
            config.hidden_dim + config.phoneme_dim + config.accent_dim,
            config.hidden_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.editor = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.residual_head = nn.Linear(config.hidden_dim, config.acoustic_dim)
        self.gate_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        acoustic_features: Tensor,
        phoneme_ids: Tensor,
        accent_ids: Tensor,
        padding_mask: Tensor | None = None,
        causal: bool = True,
    ) -> PhonemeEditorOutput:
        if acoustic_features.ndim != 3:
            raise ValueError("acoustic_features must have shape [batch, frames, acoustic_dim]")
        if phoneme_ids.shape != acoustic_features.shape[:2]:
            raise ValueError("phoneme_ids must have shape [batch, frames]")

        batch_size, num_frames, _ = acoustic_features.shape
        acoustic = self.acoustic_projection(acoustic_features)
        phoneme = self.phoneme_embedding(phoneme_ids)
        accent = self.accent_embedding(accent_ids).unsqueeze(1).expand(batch_size, num_frames, -1)
        conditioned = self.condition_projection(torch.cat([acoustic, phoneme, accent], dim=-1))

        attention_mask = _causal_mask(num_frames, acoustic_features.device) if causal else None
        hidden = self.editor(
            conditioned,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )
        residual = torch.tanh(self.residual_head(hidden)) * self.config.max_residual_scale
        edit_gate = torch.sigmoid(self.gate_head(hidden))
        edited_features = acoustic_features + edit_gate * residual
        return PhonemeEditorOutput(
            edited_features=edited_features,
            residual=residual,
            edit_gate=edit_gate,
        )


@dataclass(frozen=True)
class PhonemeEditorLossWeights:
    intelligibility: float = 1.0
    speaker: float = 0.5
    minimal_edit: float = 1.0
    gate_sparsity: float = 0.05
    residual_smoothness: float = 0.02


def phoneme_editor_loss(
    output: PhonemeEditorOutput,
    original_features: Tensor,
    intelligibility_loss: Tensor,
    original_speaker_embedding: Tensor | None = None,
    edited_speaker_embedding: Tensor | None = None,
    valid_frame_mask: Tensor | None = None,
    weights: PhonemeEditorLossWeights = PhonemeEditorLossWeights(),
) -> dict[str, Tensor]:
    minimal_edit = masked_l1(output.edited_features, original_features, valid_frame_mask)
    gate_sparsity = masked_mean(output.edit_gate, valid_frame_mask)
    smoothness = residual_smoothness_loss(output.residual, valid_frame_mask)

    speaker_loss = original_features.new_tensor(0.0)
    if original_speaker_embedding is not None and edited_speaker_embedding is not None:
        speaker_loss = 1.0 - F.cosine_similarity(
            original_speaker_embedding,
            edited_speaker_embedding,
            dim=-1,
        ).mean()

    total = (
        weights.intelligibility * intelligibility_loss
        + weights.speaker * speaker_loss
        + weights.minimal_edit * minimal_edit
        + weights.gate_sparsity * gate_sparsity
        + weights.residual_smoothness * smoothness
    )
    return {
        "total": total,
        "intelligibility": intelligibility_loss,
        "speaker": speaker_loss,
        "minimal_edit": minimal_edit,
        "gate_sparsity": gate_sparsity,
        "residual_smoothness": smoothness,
    }


def masked_l1(left: Tensor, right: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    loss = torch.abs(left - right)
    return masked_mean(loss, valid_frame_mask)


def masked_mean(values: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    if valid_frame_mask is None:
        return values.mean()
    mask = valid_frame_mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def residual_smoothness_loss(residual: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    if residual.shape[1] < 2:
        return residual.new_tensor(0.0)
    delta = residual[:, 1:] - residual[:, :-1]
    mask = valid_frame_mask[:, 1:] if valid_frame_mask is not None else None
    return masked_mean(torch.abs(delta), mask)


def _causal_mask(num_frames: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(num_frames, num_frames, dtype=torch.bool, device=device),
        diagonal=1,
    )
