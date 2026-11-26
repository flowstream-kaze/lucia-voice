"""Shared Kokoro preset selection; interpolates styles, never rendered audio."""

DEFAULT_VOICE = 'bf_emma50_lily50'
BLEND_SOURCES = ('bf_emma', 'bf_lily')


def english_voices(engine):
    voices = {v for v in engine.get_voices() if v.startswith(('af_', 'am_', 'bf_', 'bm_'))}
    if all(name in voices for name in BLEND_SOURCES):
        voices.add(DEFAULT_VOICE)
    return sorted(voices)


def resolve_voice(engine, name):
    if name not in english_voices(engine):
        raise ValueError('Unknown English voice. Use --list-voices to see the bank.')
    if name != DEFAULT_VOICE:
        return name

    import numpy as np
    styles = [np.asarray(engine.get_voice_style(source), dtype=np.float32)
              for source in BLEND_SOURCES]
    first, second = styles
    if first.ndim < 1 or first.shape[0] < 2 or first.shape != second.shape:
        raise ValueError('Blend source styles must have matching, nonempty token-length banks.')
    if not all(np.isfinite(style).all() for style in styles):
        raise ValueError('Blend source styles must be finite.')
    # Pass the full bank back to Kokoro so its normal token-length selection is retained.
    return first * .5 + second * .5
