from __future__ import annotations

from speech_comprehension.text import transcript_similarity, word_error_rate


def test_word_error_rate_counts_word_substitutions() -> None:
    assert word_error_rate("the world is changing", "the word is changing") == 0.25
    assert word_error_rate("the world is changing", "the world is changing") == 0.0


def test_word_error_rate_handles_empty_reference() -> None:
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "extra words") == 1.0


def test_transcript_similarity_remains_character_level() -> None:
    assert transcript_similarity("world", "word") == 0.8
