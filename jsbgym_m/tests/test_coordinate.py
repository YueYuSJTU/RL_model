import sys
import unittest
import numpy as np
sys.path.insert(0, "D:/Work_File/RL/jsbgym")
from jsbgym_m.coordinate import GPS_utils

class TestGPSUtils(unittest.TestCase):

    def setUp(self):
        self.gps_utils = GPS_utils('ft')
        self.gps_utils.setENUorigin(37.7749, -122.4194, 10)  # 设置原点为旧金山

    def test_geo2ecef(self):
        lat, lon, height = 37.7749, -122.4194, 10
        ecef = self.gps_utils.geo2ecef(lat, lon, height)
        x, y, z = ecef.flatten()
        self.assertIsInstance(ecef, np.ndarray)
        self.assertEqual(ecef.shape, (3, 1))
        self.assertAlmostEqual(x, -2706179.0841, places=3)
        self.assertAlmostEqual(y, -4261066.1616, places=3)
        self.assertAlmostEqual(z, 3885731.6156, places=3)

    def test_ecef2enu(self):
        x, y, z = self.gps_utils.geo2ecef(37.7749, -122.4194, 10).flatten()
        enu = self.gps_utils.ecef2enu(x, y, z)
        x_enu, y_enu, z_enu = enu.flatten()
        self.assertIsInstance(enu, np.ndarray)
        self.assertEqual(enu.shape, (3, 1))
        self.assertAlmostEqual(x_enu, 0, places=3)
        self.assertAlmostEqual(y_enu, 0, places=3)
        self.assertAlmostEqual(z_enu, 0, places=3)

    def test_geo2enu(self):
        lat, lon, height = 37.7749, -122.4194, 10
        enu = self.gps_utils.geo2enu(lat, lon, height)
        self.assertIsInstance(enu, np.ndarray)
        self.assertEqual(enu.shape, (3, 1))

    def test_ecef2geo(self):
        x, y, z = self.gps_utils.geo2ecef(37.7749, -122.4194, 10).flatten()
        geo = self.gps_utils.ecef2geo(x, y, z)
        self.assertIsInstance(geo, np.ndarray)
        self.assertEqual(geo.shape, (3, 1))

    def test_enu2ecef(self):
        x, y, z = 0, 0, 0
        ecef = self.gps_utils.enu2ecef(x, y, z)
        self.assertIsInstance(ecef, np.ndarray)
        self.assertEqual(ecef.shape, (3, 1))

    def test_enu2geo(self):
        x, y, z = 0, 0, 0
        geo = self.gps_utils.enu2geo(x, y, z)
        self.assertIsInstance(geo, np.ndarray)
        self.assertEqual(geo.shape, (3, 1))

if __name__ == '__main__':
    # unittest.main()
    trans = GPS_utils()
    trans.setENUorigin(37.7749, -122.4194, 10)
    lat, lon, height = 36.7749, -121.4194, 10
    x, y, z = trans.geo2ecef(lat, lon, height).flatten()
    print("ECEF:", x, y, z)
    x_enu, y_enu, z_enu = trans.ecef2enu(x, y, z).flatten()
    print("ENU:", x_enu, y_enu, z_enu)
    a, b, c = trans.enu2ecef(2000, 8000, 10)
    print("enu2ECEF:", a, b, c)
    lat, lon, height = trans.ecef2geo(a, b, c).flatten()
    print("ECEF2GEO:", lat, lon, height)
    unittest.main()
    # pass test !!