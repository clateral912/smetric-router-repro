import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from run_repeated_matrix import aggregate


class RepeatAggregationTests(unittest.TestCase):
    def fixture(self):
        plan = {'scenarios': ['po127', 'po64', 'pd64'], 'cases': ['cache_aware'], 'runs': []}
        runs = []
        for scenario, setting, goodputs in (('po127', 'po', [30, 60, 90]),
                                            ('po64', 'po', [0, 10, 20]),
                                            ('pd64', 'pd', [100, 120, 140])):
            for repeat, goodput in enumerate(goodputs, 1):
                identity = {'scenario': scenario, 'setting': setting, 'case': 'cache_aware', 'repeat': repeat}
                plan['runs'].append(identity)
                result = dict.fromkeys(('offered', 'served', 'passing'), 100)
                result.update(goodput_ktok_s=goodput, passing_pct=goodput,
                              ttft_s=dict.fromkeys(('mean', 'p50', 'p90', 'p95', 'p99'), repeat),
                              tpot_ms=None if setting == 'po' else dict.fromkeys(
                                  ('mean', 'p50', 'p90', 'p95', 'p99'), repeat * 10))
                runs.append(identity | {'result': result})
        return plan, runs

    def test_sample_variability_separates_settings_and_preserves_missing_tpot(self):
        plan, runs = self.fixture()
        po127, po, pd = aggregate(plan, runs)['groups']
        self.assertEqual(po127['metrics']['goodput_ktok_s']['mean'], 60)
        self.assertEqual(po127['metrics']['goodput_ktok_s']['sample_stdev'], 30)
        self.assertEqual(po['metrics']['goodput_ktok_s'], {
            'n': 3, 'values': [0, 10, 20], 'mean': 10,
            'sample_stdev': 10, 'min': 0, 'max': 20})
        self.assertEqual(pd['metrics']['goodput_ktok_s']['mean'], 120)
        self.assertEqual(pd['metrics']['goodput_ktok_s']['sample_stdev'], 20)
        self.assertIsNone(po['metrics']['tpot_ms.mean']['mean'])
        self.assertEqual(po['metrics']['tpot_ms.mean']['n'], 0)
        self.assertEqual(pd['metrics']['tpot_ms.mean']['values'], [10, 20, 30])

    def test_missing_repeat_cannot_be_reported_as_complete(self):
        plan, runs = self.fixture()
        with self.assertRaisesRegex(ValueError, 'planned matrix'):
            aggregate(plan, runs[:-1])

    def test_duplicate_repeat_cannot_replace_a_missing_repeat(self):
        plan, runs = self.fixture()
        runs[-1] = runs[-2]
        with self.assertRaisesRegex(ValueError, 'planned matrix'):
            aggregate(plan, runs)


if __name__ == '__main__':
    unittest.main()
