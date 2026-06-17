from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise RuntimeError(
        "The codec editor requires PyTorch. Install with `pip install -e .[train]`."
    ) from exc


@dataclass(frozen=True)
class CodecLatentEditorConfig:
    latent_dim: int
    num_accents: int = 1
    hidden_dim: int = 256
    accent_dim: int = 32
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_residual_scale: float = 0.20


@dataclass(frozen=True)
class CodecLatentEditorOutput:
    edited_latents: Tensor
    residual: Tensor
    edit_gate: Tensor


class StreamingCodecLatentEditor(nn.Module):
    """Sparse residual editor for continuous codec latents.

    The codec encoder/decoder owns waveform reconstruction. This module only
    learns a small gated residual in latent space:

        z' = z + gate(z, accent) * residual(z, accent)

    The gate is deliberately explicit so "do nothing everywhere" and "rewrite
    everything" are both visible training failure modes.
    """

    def __init__(self, config: CodecLatentEditorConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.accent_embedding = nn.Embedding(config.num_accents, config.accent_dim)
        self.condition_projection = nn.Linear(
            config.hidden_dim + config.accent_dim,
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
        self.residual_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.gate_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        latents: Tensor,
        accent_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
        causal: bool = True,
    ) -> CodecLatentEditorOutput:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, frames, latent_dim]")

        batch_size, num_frames, _ = latents.shape
        if accent_ids is None:
            accent_ids = torch.zeros(batch_size, dtype=torch.long, device=latents.device)
        if accent_ids.shape != (batch_size,):
            raise ValueError("accent_ids must have shape [batch]")

        latent_hidden = self.latent_projection(latents)
        accent_hidden = self.accent_embedding(accent_ids).unsqueeze(1).expand(
            batch_size,
            num_frames,
            -1,
        )
        conditioned = self.condition_projection(torch.cat([latent_hidden, accent_hidden], dim=-1))
        attention_mask = _causal_mask(num_frames, latents.device) if causal else None
        hidden = self.editor(
            conditioned,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )
        residual = torch.tanh(self.residual_head(hidden)) * self.config.max_residual_scale
        edit_gate = torch.sigmoid(self.gate_head(hidden))
        edited_latents = latents + edit_gate * residual
        return CodecLatentEditorOutput(
            edited_latents=edited_latents,
            residual=residual,
            edit_gate=edit_gate,
        )


@dataclass(frozen=True)
class LatentListenerProxyConfig:
    latent_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class LatentListenerProxy(nn.Module):
    """Differentiable proxy for listener/ASR confidence from codec latents.

    Train this head on offline labels such as Whisper confidence or WER-derived
    scores. Then freeze it and use its score as the editor's differentiable
    "edited should be more understandable than original" signal.
    """

    def __init__(self, config: LatentListenerProxyConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, latents: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, frames, latent_dim]")
        hidden = self.encoder(
            self.input_projection(latents),
            src_key_padding_mask=padding_mask,
        )
        pooled = masked_frame_pool(hidden, _valid_mask_from_padding(padding_mask, latents))
        return torch.sigmoid(self.score_head(pooled)).squeeze(-1)


@dataclass(frozen=True)
class CodecEditorLossWeights:
    listener_preference: float = 1.0
    minimal_edit: float = 1.0
    gate_sparsity: float = 0.05
    residual_smoothness: float = 0.02
    speaker: float = 0.5
    listener_margin: float = 0.02


def codec_editor_loss(
    output: CodecLatentEditorOutput,
    original_latents: Tensor,
    original_listener_score: Tensor,
    edited_listener_score: Tensor,
    original_speaker_embedding: Tensor | None = None,
    edited_speaker_embedding: Tensor | None = None,
    valid_frame_mask: Tensor | None = None,
    weights: CodecEditorLossWeights = CodecEditorLossWeights(),
) -> dict[str, Tensor]:
    improvement = edited_listener_score - original_listener_score
    listener_preference = F.relu(weights.listener_margin - improvement).mean()
    minimal_edit = masked_l1(output.edited_latents, original_latents, valid_frame_mask)
    gate_sparsity = masked_mean(output.edit_gate, valid_frame_mask)
    smoothness = residual_smoothness_loss(output.residual, valid_frame_mask)

    speaker_loss = original_latents.new_tensor(0.0)
    if original_speaker_embedding is not None and edited_speaker_embedding is not None:
        speaker_loss = 1.0 - F.cosine_similarity(
            original_speaker_embedding,
            edited_speaker_embedding,
            dim=-1,
        ).mean()

    total = (
        weights.listener_preference * listener_preference
        + weights.minimal_edit * minimal_edit
        + weights.gate_sparsity * gate_sparsity
        + weights.residual_smoothness * smoothness
        + weights.speaker * speaker_loss
    )
    return {
        "total": total,
        "listener_preference": listener_preference,
        "listener_improvement": improvement.mean(),
        "minimal_edit": minimal_edit,
        "gate_sparsity": gate_sparsity,
        "residual_smoothness": smoothness,
        "speaker": speaker_loss,
    }


def listener_proxy_loss(predicted_score: Tensor, target_score: Tensor) -> Tensor:
    """Train the listener proxy from ASR/listener confidence labels in [0, 1]."""

    return F.mse_loss(predicted_score, target_score.to(predicted_score.dtype))


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def masked_l1(left: Tensor, right: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    return masked_mean(torch.abs(left - right), valid_frame_mask)


def masked_mean(values: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    if valid_frame_mask is None:
        return values.mean()
    mask = valid_frame_mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def masked_frame_pool(values: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    if values.ndim != 3:
        raise ValueError("values must have shape [batch, frames, dim]")
    if valid_frame_mask is None:
        return values.mean(dim=1)
    mask = valid_frame_mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denominator


def residual_smoothness_loss(residual: Tensor, valid_frame_mask: Tensor | None = None) -> Tensor:
    if residual.shape[1] < 2:
        return residual.new_tensor(0.0)
    delta = residual[:, 1:] - residual[:, :-1]
    mask = valid_frame_mask[:, 1:] if valid_frame_mask is not None else None
    return masked_mean(torch.abs(delta), mask)


def _valid_mask_from_padding(padding_mask: Tensor | None, latents: Tensor) -> Tensor | None:
    if padding_mask is None:
        return None
    return ~padding_mask.to(device=latents.device)


def _causal_mask(num_frames: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(num_frames, num_frames, dtype=torch.bool, device=device),
        diagonal=1,
    )
