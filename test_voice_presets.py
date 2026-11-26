"""Standard-library preset tests; no model, numpy installation or Windows needed."""
import math
import sys
import unittest
from unittest.mock import Mock, patch

from voice_presets import DEFAULT_VOICE, english_voices, resolve_voice


class Vector:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)
        self.ndim = 1

    def __mul__(self, weight):
        return Vector([x * weight for x in self.values])

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.values, other.values)])


class PresetTests(unittest.TestCase):
    def engine(self):
        engine = Mock()
        engine.get_voices.return_value = ['bf_emma', 'bf_lily', 'af_heart', 'jf_alpha']
        return engine

    def numpy(self):
        module = Mock()
        module.asarray.side_effect = lambda style, dtype: style
        module.isfinite.side_effect = lambda style: Mock(
            all=lambda: all(math.isfinite(x) for x in style.values))
        return module

    def test_list_adds_alias_only_with_both_sources_and_filters_non_english(self):
        engine = self.engine()
        self.assertEqual(english_voices(engine), ['af_heart', 'bf_emma', DEFAULT_VOICE, 'bf_lily'])
        engine.get_voices.return_value = ['bf_emma']
        self.assertNotIn(DEFAULT_VOICE, english_voices(engine))

    def test_plain_voice_does_not_import_numpy_or_read_styles(self):
        engine = self.engine()
        with patch.dict(sys.modules, {'numpy': None}):
            self.assertEqual(resolve_voice(engine, 'af_heart'), 'af_heart')
        engine.get_voice_style.assert_not_called()

    def test_unknown_or_missing_source_fails_before_style_access(self):
        engine = self.engine()
        for name in ['bf_missing', 'jf_alpha']:
            with self.assertRaises(ValueError):
                resolve_voice(engine, name)
        engine.get_voices.return_value = ['bf_emma']
        with self.assertRaises(ValueError):
            resolve_voice(engine, DEFAULT_VOICE)
        engine.get_voice_style.assert_not_called()

    def test_equal_weight_blend_retains_whole_token_length_bank(self):
        engine = self.engine()
        engine.get_voice_style.side_effect = [Vector([2., 4., 6.]), Vector([8., 10., 12.])]
        with patch.dict(sys.modules, {'numpy': self.numpy()}):
            result = resolve_voice(engine, DEFAULT_VOICE)
        self.assertEqual(result.values, [5., 7., 9.])
        self.assertEqual(result.shape, (3,))
        self.assertEqual([c.args for c in engine.get_voice_style.call_args_list],
                         [('bf_emma',), ('bf_lily',)])

    def test_invalid_banks_rejected(self):
        cases = [([1.], [2.]), ([1., 2.], [3., 4., 5.]),
                 ([float('nan'), 2.], [3., 4.]), ([1., 2.], [float('inf'), 4.])]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                engine = self.engine()
                engine.get_voice_style.side_effect = [Vector(a), Vector(b)]
                with patch.dict(sys.modules, {'numpy': self.numpy()}), self.assertRaises(ValueError):
                    resolve_voice(engine, DEFAULT_VOICE)


if __name__ == '__main__':
    unittest.main()
