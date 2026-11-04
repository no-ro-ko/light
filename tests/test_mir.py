"""Песочница «Мир»: законы, проверка законов, проверка самих проверок."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.mir import patches, probes, world as W  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
