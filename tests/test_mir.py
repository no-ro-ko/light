"""Песочница «Мир»: законы, проверка законов, проверка самих проверок."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.mir import curious, life, patches, probes, world as W  # noqa: E402
from light.mir.mlp import PatchMLP  # noqa: E402
from light.mir.diary import Diary  # noqa: E402
from light.mir.seg import SegCortex  # noqa: E402


def run(g, acts, ph0=0):
    g = g[None].copy()
    ph = np.array([ph0])
    for a in acts:
        g = W.step(g, np.array([a]), ph)
        ph = 1 - ph
    return g[0]


def scene(*cells, agent=(10, 5, W.AGENT_R)):
    g = W.empty_world()
    for y, x, k in cells:
        g[y, x] = k
    g[agent[0], agent[1]] = agent[2]
    return g


class TestWorld(unittest.TestCase):
    def test_gravity_and_rest(self):
        g = scene((2, 10, W.SAND), (2, 12, W.STONE))
        out = run(g, [W.NOOP] * 12)
        self.assertEqual(out[10, 10], W.SAND)
        self.assertEqual(out[10, 12], W.STONE)
        self.assertTrue((run(g, [0] * 40) == run(g, [0] * 41)).all())

    def test_sand_flattens_and_water_settles(self):
        g = scene(*[(y, 10, W.SAND) for y in range(5, 10)])
        out = run(g, [0] * 20)
        self.assertEqual(int((out[10] == W.SAND).sum()), 5)
        g = scene(*[(y, 10, W.WATER) for y in range(2, 6)])
        self.assertTrue((run(g, [0] * 60) == run(g, [0] * 61)).all())

    def test_fire_burns_dry_wood_only(self):
        g = scene((9, 8, W.FIRE), (9, 9, W.WOOD), (9, 10, W.WOOD))
        self.assertEqual(run(g, [0])[9, 9], W.FIRE)
        self.assertEqual(run(g, [0])[9, 8], W.AIR)
        wet = scene((9, 8, W.FIRE), (9, 9, W.WOOD), (8, 9, W.WATER))
        self.assertEqual(run(wet, [0])[9, 9], W.WOOD)

    def test_fire_melts_ice(self):
        g = scene((9, 8, W.FIRE), (9, 9, W.ICE))
        self.assertEqual(run(g, [0])[9, 9], W.WATER)

    def test_agent_actions(self):
        self.assertEqual(run(scene(), [W.RIGHT])[10, 6], W.AGENT_R)
        bump = run(scene((10, 6, W.GROUND), agent=(10, 5, W.AGENT_L)), [W.RIGHT])
        self.assertEqual((bump[10, 5], bump[10, 6]), (W.AGENT_R, W.GROUND))
        up = run(scene((10, 7, W.GROUND), agent=(10, 6, W.AGENT_R)), [W.JUMP, W.RIGHT])
        self.assertEqual(up[9, 7], W.AGENT_R)
        self.assertEqual(run(scene((10, 6, W.SAND)), [W.ACT])[10, 6], W.AIR)
        self.assertEqual(run(scene((10, 6, W.WOOD)), [W.ACT])[10, 6], W.FIRE)
        self.assertEqual(run(scene((10, 6, W.WOOD), (9, 6, W.WATER)), [W.ACT])[10, 6], W.WOOD)
        push = run(scene((10, 6, W.STONE)), [W.ACT])
        self.assertEqual((push[10, 6], push[10, 7]), (W.AIR, W.STONE))
        self.assertEqual(run(scene((10, 6, W.STONE), (10, 7, W.GROUND)), [W.ACT])[10, 6], W.STONE)
        self.assertEqual(run(scene(), [W.PUT_WATER])[10, 6], W.WATER)

    def test_invariants_on_random_lives(self):
        rng = np.random.default_rng(1)
        env = W.Sandbox(150, seed=3)
        for _ in range(40):
            a = rng.integers(0, W.NA, 150)
            g0, _, g1 = env.step(a)
            self.assertTrue(((g1 >= W.AGENT_L).reshape(150, -1).sum(1) == 1).all())
            self.assertTrue((g1[:, 0] == W.GROUND).all() and (g1[:, :, 0] == W.GROUND).all())
            quiet = (~np.isin(a, [W.ACT, W.PUT_SAND, W.PUT_WATER, W.PUT_STONE])
                     & ~(g0 == W.FIRE).any((1, 2)) & ~(g0 == W.ICE).any((1, 2)))
            for k in (W.STONE, W.SAND, W.WATER, W.WOOD, W.ICE):
                same = (g0 == k).sum((1, 2)) == (g1 == k).sum((1, 2))
                self.assertTrue(same[quiet].all())

    def test_locality_radius_two(self):
        rng = np.random.default_rng(7)
        n = 2500
        env = W.Sandbox(n, seed=21)
        for _ in range(6):
            env.step(rng.integers(0, W.NA, n))
        g, ph, a = env.g.copy(), rng.integers(0, 2, n), rng.integers(0, W.NA, n)
        base = W.step(g, a, ph)
        sel = np.nonzero((base != g).reshape(n, -1).any(1))[0]
        Y, X = np.mgrid[0:W.H, 0:W.W]
        inner = (Y > 0) & (Y < W.H - 1) & (X > 0) & (X < W.W - 1)
        g2 = g[sel].copy()
        ty, tx = np.zeros(len(sel), int), np.zeros(len(sel), int)
        for k, i in enumerate(sel):
            ys, xs = np.nonzero(base[i] != g[i])
            j = rng.integers(len(ys))
            ty[k], tx[k] = ys[j], xs[j]
            far = inner & (np.maximum(abs(Y - ty[k]), abs(X - tx[k])) > 2) & (g[i] < W.AGENT_L)
            flip = far & (rng.random((W.H, W.W)) < .35)
            g2[k][flip] = rng.choice([0, 1, 2, 3, 4, 5, 6, 7], int(flip.sum()))
        out = W.step(g2, a[sel], ph[sel])
        k = np.arange(len(sel))
        self.assertTrue((out[k, ty, tx] == base[sel, ty, tx]).all())


class TestPatches(unittest.TestCase):
    def test_dedupe_is_exact(self):
        env = W.Sandbox(60, seed=2)
        rng = np.random.default_rng(0)
        g0s, a_s, p_s, g1s = [], [], [], []
        for _ in range(8):
            a = rng.integers(0, W.NA, 60)
            g0, ph, g1 = env.step(a)
            g0s.append(g0), a_s.append(a), p_s.append(ph), g1s.append(g1)
        S = patches.Samples.collect(np.concatenate(g0s), np.concatenate(a_s), np.concatenate(p_s), np.concatenate(g1s))
        self.assertEqual(int(S.w.sum()), 480 * W.H * W.W)
        self.assertEqual(len({(r.tobytes(), int(y)) for r, y in zip(S.X, S.y)}), len(S))


class TestProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pr = probes.build(seed=99, per_mode=6)

    def test_reference_models(self):
        self.assertGreaterEqual(probes.evaluate(probes.RuleModel(), self.pr)["macro"], .99)
        self.assertLessEqual(probes.evaluate(probes.RuleModel(0), self.pr)["macro"], .40)
        self.assertAlmostEqual(probes.evaluate(probes.UniformModel(), self.pr)["macro"], .5)

    def test_each_law_is_needed_by_its_family(self):
        for bit, fams in ((W.R_FALL, ["падение", "я падаю"]), (W.R_SLIDE, ["осыпание песка"]),
                          (W.R_BURN, ["горение"]), (W.R_PUSH, ["толкаю"]), (W.R_MOVE, ["иду", "упираюсь"])):
            r = probes.evaluate(probes.RuleModel(W.R_ALL & ~bit), self.pr)
            for f in fams:
                self.assertLess(r["families"][f], .7, (bit, f))

    def test_pairs_differ_and_are_real(self):
        self.assertTrue((self.pr.G1T != self.pr.G1I).reshape(len(self.pr), -1).any(1).all())
        again = W.step(self.pr.G0, self.pr.A, self.pr.PH)
        self.assertTrue((again == self.pr.G1T).all())


class TestLearners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = life.wander(1500, seed=5, cache=False)
        cls.pr = probes.build(seed=99, per_mode=6)

    def test_pooled_features(self):
        g = W.empty_world()
        g[5, 5], g[5, 6], g[4, 6] = W.WOOD, W.FIRE, W.WATER   # огонь справа, вода — по диагонали
        X = patches.patches(g[None], np.array([0]), np.array([0]))[0, 5, 5]
        self.assertEqual(X[patches.POOL4 + W.FIRE], 1)
        self.assertEqual(X[patches.POOL4 + W.WATER], 0)      # вода по диагонали — не «по сторонам»
        self.assertEqual(X[patches.POOL8 + W.WATER], 1)      # но среди восьми соседей

    def test_cortex_learns_gravity_fast(self):
        S0, A, PH, S1 = (x[:400] for x in self.stream)
        m = SegCortex(tol=0.8).fit(S0, A, PH, S1)
        r = probes.evaluate(m, self.pr)
        self.assertGreaterEqual(r["families"]["падение"], .9)
        self.assertGreater(r["macro"], .5)
        self.assertIn("песок → воздух", m.rules_text(top=60, min_weight=1))

    def test_cortex_probabilities_and_sleep(self):
        S0, A, PH, S1 = (x[:800] for x in self.stream)
        m = SegCortex(tol=0.8, consolidate=True).fit(S0, A, PH, S1)
        lp = m.logp(self.pr.G0[:20], self.pr.A[:20], self.pr.PH[:20])
        self.assertTrue(np.allclose(np.exp(lp).sum(-1), 1, atol=1e-3))
        self.assertGreater(len(m.rules), 0)
        self.assertTrue(all(r["n_pos"] >= 2 for r in m.rules))
        self.assertTrue(m.laws_text())

    def test_incremental_equals_batch_knowledge(self):
        S0, A, PH, S1 = (x[:600] for x in self.stream)
        a = SegCortex(tol=0.8).fit(S0, A, PH, S1)
        b = SegCortex(tol=0.8)
        for i in range(0, 600, 200):
            b.partial_fit(S0[i:i + 200], A[i:i + 200], PH[i:i + 200], S1[i:i + 200])
        self.assertAlmostEqual(probes.evaluate(a, self.pr)["macro"], probes.evaluate(b, self.pr)["macro"], delta=.12)

    def test_mlp_runs_and_normalises(self):
        S0, A, PH, S1 = (x[:200] for x in self.stream)
        m = PatchMLP(width=16, steps=120, eval_every=60).fit(S0, A, PH, S1)
        lp = m.logp(self.pr.G0[:10], self.pr.A[:10], self.pr.PH[:10])
        self.assertTrue(np.allclose(np.exp(lp).sum(-1), 1, atol=1e-3))
        self.assertGreater(probes.evaluate(m, self.pr)["macro"], .3)

    def test_curious_day_runs(self):
        res = curious.live("curious", {"tol": .8}, seed=3, T_max=100, mode="dev", checkpoints=[100])
        self.assertEqual(len(res), 1)
        self.assertIn("dev", res[0])
        self.assertEqual(set(res[0]["second_witness_at"]), set(W.RULE_NAMES.values()))


class TestDiary(unittest.TestCase):
    DIR = ["осыпание песка", "растекание воды"]

    @classmethod
    def setUpClass(cls):
        cls.S0, cls.A, cls.PH, cls.S1 = life.wander(4000, seed=7, cache=False)
        cls.pr = probes.build(seed=99, per_mode=8)
        cls.prB = probes.Probes(**{**cls.pr.__dict__, "PH": 1 - cls.pr.PH})

    def _direction(self, m, pr):
        r = probes.evaluate(m, pr)
        return sum(r["families"][f] for f in self.DIR) / 2

    def test_cortex_forgetting_clock_and_rebuild(self):
        S0, A, PH, S1 = (x[:600] for x in (self.S0, self.A, self.PH, self.S1))
        m = SegCortex(tol=0.8, half_life=300)
        m.partial_fit(S0[:300], A[:300], PH[:300], S1[:300])
        w0 = m.w.sum()
        m.partial_fit(S0[300:600], A[300:600], PH[300:600], S1[300:600])
        self.assertEqual(m.clock, 600)
        self.assertEqual(int(m.last.max()), 600)
        base = SegCortex(tol=0.8).fit(S0, A, PH, S1)
        before = probes.evaluate(base, self.pr)["macro"]
        base.rebuild()
        self.assertAlmostEqual(before, probes.evaluate(base, self.pr)["macro"], delta=0.04)
        self.assertGreater(w0, 0)

    def test_diary_does_not_disturb_a_stable_world(self):
        n = 1500
        S0, A, PH, S1 = (x[:n] for x in (self.S0, self.A, self.PH, self.S1))
        plain = SegCortex(tol=0.8)
        plain.fit(S0[:n - 600], A[:n - 600], PH[:n - 600], S1[:n - 600])
        d = Diary(SegCortex(tol=0.8)).fit(S0, A, PH, S1, warm=600, chunk=200)
        for i in range(n - 600, n, 200):
            plain.partial_fit(S0[i:i + 200], A[i:i + 200], PH[i:i + 200], S1[i:i + 200])
        plain.rebuild()
        d.sleep()
        a, b = probes.evaluate(plain, self.pr)["macro"], probes.evaluate(d, self.pr)["macro"]
        self.assertAlmostEqual(a, b, delta=0.03)
        self.assertLessEqual(d.stats()["active_leaves"], 3)

    def test_diary_switches_after_a_law_change(self):
        n_a = 2500
        S0, A, PH, S1 = self.S0, self.A, self.PH, self.S1
        plain = SegCortex(tol=0.8).fit(S0[:n_a - 600], A[:n_a - 600], PH[:n_a - 600], S1[:n_a - 600])
        d = Diary(SegCortex(tol=0.8)).fit(S0[:n_a], A[:n_a], PH[:n_a], S1[:n_a], warm=600, chunk=200)
        for i in range(n_a - 600, n_a, 200):
            plain.partial_fit(S0[i:i + 200], A[i:i + 200], PH[i:i + 200], S1[i:i + 200])
        for i in range(n_a, n_a + 1200, 100):          # мир B: метроном наоборот
            ch = (S0[i:i + 100], A[i:i + 100], 1 - PH[i:i + 100], S1[i:i + 100])
            plain.partial_fit(*ch)
            d.partial_fit(*ch)
        plain.rebuild()
        d.sleep()
        self.assertGreater(self._direction(d, self.prB), self._direction(plain, self.prB) + 0.10)
        self.assertGreater(d.stats()["active_leaves"], 0)

    def test_consolidation_retires_stale_evidence(self):
        n_a = 2500
        S0, A, PH, S1 = self.S0, self.A, self.PH, self.S1
        d = Diary(SegCortex(tol=0.8), consolidate=True).fit(S0[:n_a], A[:n_a], PH[:n_a], S1[:n_a], warm=600, chunk=200)
        w_before = d.cortex.w.sum()
        for i in range(n_a, n_a + 1200, 100):
            d.partial_fit(S0[i:i + 100], A[i:i + 100], 1 - PH[i:i + 100], S1[i:i + 100])
        d.sleep()
        self.assertGreater(d.info.get("retired_rows", 0), 0)
        self.assertGreater(w_before, 0)


if __name__ == "__main__":
    unittest.main()
