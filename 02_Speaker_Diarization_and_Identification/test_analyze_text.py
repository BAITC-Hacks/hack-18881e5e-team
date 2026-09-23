import copy
from pathlib import Path
import unittest
from unittest.mock import patch, mock_open
from analyze_text import batches, load_units, validate


class SourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.units = [
            {'index': 0, 'text': 'Когда?', 'start': 1.2, 'end': 2.5},
            {'index': 1, 'text': 'Завтра.', 'start': 2.5, 'end': 3.8}]
        self.result = {'fragments': [{'first': 0, 'last': 1, 'form': 'dialogue',
            'communication_type': 'question_answer', 'evidence': 'Когда?'}]}

    def test_preserve_text_times_and_unconfirmed_status(self):
        actual = validate(self.result, self.units)[0]
        self.assertEqual(actual['source_segments'], self.units)
        self.assertEqual((actual['start'], actual['end']), (1.2, 3.8))
        self.assertEqual(actual['text'], 'Когда?\nЗавтра.')
        self.assertEqual(actual['status'], 'needs_review')

    def test_reject_omissions_overlaps_out_of_bounds(self):
        for a, b in [(1, 1), (0, 0), (0, 2), (-1, 1)]:
            result = copy.deepcopy(self.result)
            result['fragments'][0].update(first=a, last=b)
            with self.assertRaises(ValueError):
                validate(result, self.units)
        result = copy.deepcopy(self.result)
        result['fragments'] *= 2
        with self.assertRaises(ValueError):
            validate(result, self.units)

    def test_reject_fabricated_quote(self):
        self.result['fragments'][0]['evidence'] = 'Меня зовут Иван.'
        actual = validate(self.result, self.units)[0]
        self.assertEqual(actual['form'], 'uncertain')
        self.assertEqual(actual['evidence'], '')
        self.assertFalse(actual['evidence_verified'])

    def test_txt_lossless_unicode_offsets(self):
        text = '  Привет!\r\n— Когда?  Завтра.\r\n'
        with patch.object(Path, 'open', mock_open(read_data=text)):
            units = load_units(Path('sample.txt'))
        self.assertEqual(''.join(u['text'] for u in units), text)
        for unit in units:
            self.assertEqual(text[unit['char_start']:unit['char_end']], unit['text'])

    def test_batches_no_loss_and_oversize_rejected(self):
        self.assertEqual([u for b in batches(self.units, 60) for u in b], self.units)
        with self.assertRaises(ValueError):
            list(batches(self.units, 10))


if __name__ == '__main__':
    unittest.main()
