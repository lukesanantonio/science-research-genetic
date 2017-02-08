import parse_program
import unittest


class GeneticProgramTest(unittest.TestCase):
    def test_pick_random_node(self):
        found = []
        l = [1, [2, [3, [4, [5, [6, [7, [8, 9]]]]]]]]
        LEN = 8
        # This may hang under weird circumstances
        for i in range(100000):
            node, i = parse_program.pick_random_node(l)
            self.assertNotEqual(0, i)
            if node not in found:
                found.append(node)

            if len(found) == LEN:
                break

        self.assertEqual(LEN, len(found))
