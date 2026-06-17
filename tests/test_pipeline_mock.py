from __future__ import annotations

from pathlib import Path

import numpy as np

from speech_comprehension.asr import ASRResult, ASRSegment, MockASRBackend, WordTiming
from speech_comprehension.audio import read_wav, write_wav, AudioSignal
from speech_comprehension.pipeline import RepairConfig, RepairPipeline


def test_pipeline_accepts_minimal_local_repair(tmp_path: Path) -> None:
    sample_rate = 16000
    seconds = 1.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    samples = 0.04 * np.sin(2.0 * np.pi * 180.0 * t)

    start_i = int(0.38 * sample_rate)
    end_i = int(0.62 * sample_rate)
    samples[start_i:end_i] += 0.0001 * np.sin(
        2.0 * np.pi * 2600.0 * t[start_i:end_i]
    )

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    report_path = tmp_path / "report.json"
    write_wav(input_path, AudioSignal(samples.astype(np.float32), sample_rate))

    backend = MockASRBackend(_mock_world_confidence)
    pipeline = RepairPipeline(
        backend,
        config=RepairConfig(
            confidence_threshold=0.72,
            min_confidence_gain=0.005,
            edit_lambda=0.15,
            max_edit_ratio=0.50,
        ),
    )

    report = pipeline.repair(input_path, output_path, report_path=report_path)

    assert output_path.exists()
    assert report_path.exists()
    assert report.accepted_edits
    assert report.final_confidence > report.original_confidence
    assert report.transcript_similarity == 1.0


def test_pipeline_uses_reference_transcript_as_preservation_target(tmp_path: Path) -> None:
    sample_rate = 16000
    seconds = 1.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    samples = 0.04 * np.sin(2.0 * np.pi * 180.0 * t)

    start_i = int(0.38 * sample_rate)
    end_i = int(0.62 * sample_rate)
    samples[start_i:end_i] += 0.0001 * np.sin(
        2.0 * np.pi * 2600.0 * t[start_i:end_i]
    )

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    write_wav(input_path, AudioSignal(samples.astype(np.float32), sample_rate))

    backend = MockASRBackend(_mock_word_to_world_repair)
    pipeline = RepairPipeline(
        backend,
        config=RepairConfig(
            confidence_threshold=0.72,
            min_confidence_gain=0.005,
            edit_lambda=0.15,
            max_edit_ratio=0.50,
            transcript_similarity_floor=0.98,
        ),
    )

    report = pipeline.repair(
        input_path,
        output_path,
        reference_transcript="the world is changing",
    )

    assert report.accepted_edits
    assert report.reference_transcript == "the world is changing"
    assert report.original_transcript == "the word is changing"
    assert report.final_transcript == "the world is changing"
    assert report.original_wer == 0.25
    assert report.final_wer == 0.0


def _mock_world_confidence(audio_path: str | Path) -> ASRResult:
    signal = read_wav(audio_path)
    confidence = _synthetic_clarity_confidence(signal)

    return ASRResult(
        text="the world is changing",
        segments=[
            ASRSegment(
                text="the world is changing",
                start=0.0,
                end=1.0,
                confidence=(0.90 + confidence + 0.90 + 0.90) / 4.0,
                words=[
                    WordTiming("the", 0.05, 0.20, 0.90),
                    WordTiming("world", 0.38, 0.62, confidence),
                    WordTiming("is", 0.64, 0.74, 0.90),
                    WordTiming("changing", 0.76, 0.95, 0.90),
                ],
            )
        ],
    )


def _mock_word_to_world_repair(audio_path: str | Path) -> ASRResult:
    signal = read_wav(audio_path)
    confidence = _synthetic_clarity_confidence(signal)
    repaired = confidence >= 0.54
    word = "world" if repaired else "word"

    return ASRResult(
        text=f"the {word} is changing",
        segments=[
            ASRSegment(
                text=f"the {word} is changing",
                start=0.0,
                end=1.0,
                confidence=(0.90 + confidence + 0.90 + 0.90) / 4.0,
                words=[
                    WordTiming("the", 0.05, 0.20, 0.90),
                    WordTiming(word, 0.38, 0.62, confidence),
                    WordTiming("is", 0.64, 0.74, 0.90),
                    WordTiming("changing", 0.76, 0.95, 0.90),
                ],
            )
        ],
    )


def _synthetic_clarity_confidence(signal: AudioSignal) -> float:
    start_i, end_i = signal.span_to_samples(0.38, 0.62)
    segment = signal.samples[start_i:end_i]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / signal.sample_rate)
    consonant_band = float(np.mean(spectrum[(freqs >= 2550.0) & (freqs <= 2650.0)]))
    vowel_band = float(np.mean(spectrum[(freqs >= 120.0) & (freqs <= 260.0)]))
    clarity_ratio = consonant_band / max(vowel_band, 1e-8)
    return min(0.95, 0.45 + clarity_ratio * 20.0)
