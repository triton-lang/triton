import unittest
from third_party.metal.backend import driver

class TestMetalDriver(unittest.TestCase):
    def test_get_device_properties(self):
        props = driver.get_device_properties(0)
        self.assertIsInstance(props, dict)
        self.assertIn("name", props)
        self.assertIsInstance(props["name"], str)
        self.assertGreater(len(props["name"]), 0)

if __name__ == '__main__':
    unittest.main()
