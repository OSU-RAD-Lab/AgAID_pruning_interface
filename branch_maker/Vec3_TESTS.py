import unittest
import math
import numpy as np
from scipy.spatial.transform import Rotation
from Vec3 import Vec3

class TestVec3(unittest.TestCase):

    def setUp(self):
        # Initialize some Vec3 instances for testing
        self.vec1 = Vec3(1, 2, 3)
        self.vec2 = Vec3(4, 5, 6)
        self.zero_vec = Vec3.zero()

    def test_addition(self):
        result = self.vec1 + self.vec2
        self.assertEqual(result, Vec3(5, 7, 9))

    def test_subtraction(self):
        result = self.vec2 - self.vec1
        self.assertEqual(result, Vec3(3, 3, 3))

    def test_negation(self):
        result = -self.vec1
        self.assertEqual(result, Vec3(-1, -2, -3))

    def test_multiplication(self):
        result = self.vec1 * 2
        self.assertEqual(result, Vec3(2, 4, 6))

    def test_division(self):
        result = self.vec2 / 2
        self.assertEqual(result, Vec3(2, 2.5, 3))

    def test_getitem(self):
        self.assertEqual(self.vec1[0], 1)
        self.assertEqual(self.vec2[1], 5)
        with self.assertRaises(IndexError):
            value = self.vec1[3]

    def test_len(self):
        self.assertEqual(len(self.vec1), 3)

    def test_iter(self):
        result = list(iter(self.vec1))
        self.assertEqual(result, [1, 2, 3])

    def test_contains(self):
        self.assertTrue(2 in self.vec1)
        self.assertFalse(4 in self.vec1)

    def test_reversed(self):
        result = list(reversed(self.vec1))
        self.assertEqual(result, [3, 2, 1])

    def test_magnitude(self):
        result = self.vec1.magnitude()
        self.assertAlmostEqual(result, math.sqrt(14), places=5)

    def test_normalize_or_zero(self):
        result = self.zero_vec.normalize_or_zero()
        self.assertEqual(result, Vec3.zero())

    def test_normalize(self):
        with self.assertRaises(ValueError):
            self.zero_vec.normalize()

    def test_cross(self):
        result = self.vec1.cross(self.vec2)
        self.assertEqual(result, Vec3(-3, 6, -3))

    def test_dot(self):
        result = self.vec1.dot(self.vec2)
        self.assertEqual(result, 32)

    def test_practically_the_same_as(self):
        self.assertTrue(self.vec1.practically_the_same_as(Vec3(1.000001, 2.0000001, 3.0000001)))
        self.assertFalse(self.vec1.practically_the_same_as(self.vec2))

    def test_lerp(self):
        result = self.vec1.lerp(self.vec2, 0.5)
        self.assertEqual(result, Vec3(2.5, 3.5, 4.5))

    def test_transform(self):
        rotation = Rotation.from_euler('xyz', [45, 30, 60], degrees=True)
        transformed_vec = self.vec1.transform(rotation)
        goal = Vec3(1.92927, 1.92738, 2.56186)
        self.assertTrue(transformed_vec.practically_the_same_as(goal), f"Got {transformed_vec} expected {goal}")
        # Add assertions based on expected transformed vector values

    def test_to_arbitrary_perpendicular(self):
        result = self.vec1.to_arbitrary_perpendicular()
        # Add assertions based on expected perpendicular vector values
        self.assertAlmostEqual(result.dot(self.vec1), 0)

    # def test_get_rotation_to(self):
    #     result = Vec3(1,0,1).get_rotation_to(Vec3(0,1,0))
    #     goal = Rotation.from_rotvec(np.array(math.tau/2 * Vec3(-1, 0, 1).normalize()))
    #     # Add assertions based on expected Rotation values
    #     self.assertTrue(result.approx_equal(goal, 1e-5), f"Got {result.as_euler('xyz', True)} expected {goal.as_euler('xyz', True)}")

    # def test_to_rotation_using_parallel_transport(self):
    #     rotation = Rotation.from_euler('xyz', [45, 30, 60], degrees=True)
    #     result = self.vec2.to_rotation_using_parallel_transport(rotation, self.vec1)
    #     # Add assertions based on expected Rotation values

    def test_get_tangent_to_and_from(self):
        result = Vec3(0,0,0).get_tangent_to_and_from(Vec3(1,0,1),Vec3(1,0,-1))
        goal = Vec3(0,0,1)
        self.assertTrue(result.practically_the_same_as(goal), f"Got {result} expected {goal}")

    def test_rotate_around(self):
        result = Vec3(0,1,1).rotate_around(Vec3(1,0,1), math.tau/8).rotate_around(Vec3(1,0,1), math.tau/8).rotate_around(Vec3(1,0,1), math.tau/4)
        goal = Vec3(1,-1,0)
        self.assertTrue(result.practically_the_same_as(goal), f"Got {result} expected {goal}")

if __name__ == '__main__':
    unittest.main()