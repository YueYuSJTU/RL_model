import sys
import unittest
import numpy as np
sys.path.insert(0, "D:/Work_File/RL/jsbgym")
from jsbgym_m.coordinate import GPS_utils

class TestGPSUtils(unittest.TestCase):

    def setUp(self):
        self.gps_utils = GPS_utils('m')
        self.gps_utils.setENUorigin(np.array([37.7749]), np.array([-122.4194]), np.array([10]))  # 设置原点为旧金山

        self.ft_gps = GPS_utils('ft')
        self.ft_gps.setENUorigin(np.array([37.7749]), np.array([-122.4194]), np.array([10/0.3048]))  # 设置原点为旧金山

    def test_geo2ecef(self):
        lat, lon, height = 37.7749, -122.4194, 10
        ecef = self.gps_utils.geo2ecef(lat, lon, height)
        x, y, z = ecef.flatten()
        self.assertIsInstance(ecef, np.ndarray)
        self.assertEqual(ecef.shape, (3, 1))
        self.assertAlmostEqual(x, -2706179.0841, places=3)
        self.assertAlmostEqual(y, -4261066.1616, places=3)
        self.assertAlmostEqual(z, 3885731.6156, places=3)

        lat, lon, height = 37.7749, -122.4194, 32.8084
        ecef = self.ft_gps.geo2ecef(lat, lon, height)
        x, y, z = ecef.flatten()
        self.assertIsInstance(ecef, np.ndarray)
        self.assertEqual(ecef.shape, (3, 1))
        self.assertAlmostEqual(x, -2706179.0841/0.3048, places=3)
        self.assertAlmostEqual(y, -4261066.1616/0.3048, places=3)
        self.assertAlmostEqual(z, 3885731.6156/0.3048, places=3)

    def test_ecef2enu(self):
        x, y, z = self.gps_utils.geo2ecef(37.7749, -122.4194, 10).flatten()
        enu = self.gps_utils.ecef2enu(x, y, z)
        x_enu, y_enu, z_enu = enu.flatten()
        self.assertIsInstance(enu, np.ndarray)
        self.assertEqual(enu.shape, (3, 1))
        self.assertAlmostEqual(x_enu, 0, places=3)
        self.assertAlmostEqual(y_enu, 0, places=3)
        self.assertAlmostEqual(z_enu, 0, places=3)

        x, y, z = self.ft_gps.geo2ecef(37.7749, -122.4194, 32.8084).flatten()
        # print("debug:ft xyz", x, y, z)
        enu = self.ft_gps.ecef2enu(x, y, z)
        x_enu, y_enu, z_enu = enu.flatten()
        # print("debug:", x_enu, y_enu, z_enu)
        self.assertIsInstance(enu, np.ndarray)
        self.assertEqual(enu.shape, (3, 1))
        self.assertAlmostEqual(x_enu, 0, places=3)
        self.assertAlmostEqual(y_enu, 0, places=3)
        self.assertAlmostEqual(z_enu, 0, places=3)

    def test_geo2enu(self):
        lat, lon, height = 37.7749, -122.5194, 10
        enu = self.gps_utils.geo2enu(lat, lon, height)
        self.assertIsInstance(enu, np.ndarray)
        self.assertEqual(enu.shape, (3, 1))
        x_enu, y_enu, z_enu = enu.flatten()
        self.assertAlmostEqual(x_enu, -8810.0348533538, places=3)
        self.assertAlmostEqual(y_enu, 4.7094950234, places=3)
        self.assertAlmostEqual(z_enu, -6.0769395753, places=3)

        lat, lon, height = 37.7749, -122.5194, 10/0.3048
        enu = self.ft_gps.geo2enu(lat, lon, height)
        self.assertIsInstance(enu, np.ndarray)
        self.assertEqual(enu.shape, (3, 1))
        x_enu, y_enu, z_enu = enu.flatten()
        self.assertAlmostEqual(x_enu, -8810.0348533538 / 0.3048, places=3)
        self.assertAlmostEqual(y_enu, 4.7094950234 / 0.3048, places=3)
        self.assertAlmostEqual(z_enu, -6.0769395753 / 0.3048, places=3)

    def test_ecef2geo(self):
        x, y, z = -2492335.3220309215 , 4699403.146501685 , 3269053.8583770543
        geo = self.gps_utils.ecef2geo(x, y, z)
        self.assertIsInstance(geo, np.ndarray)
        self.assertEqual(geo.shape, (3, 1))
        lat, lon, height = geo.flatten()
        self.assertAlmostEqual(lat, 31.7483709921, places=3)
        self.assertAlmostEqual(lon, 117.9393085632, places=3)
        self.assertAlmostEqual(height, -128628.7493826561, places=3)

        x, y, z = -2492335.3220309215 / 0.3048, 4699403.146501685 / 0.3048, 3269053.8583770543 / 0.3048
        geo = self.ft_gps.ecef2geo(x, y, z)
        self.assertIsInstance(geo, np.ndarray)
        self.assertEqual(geo.shape, (3, 1))
        lat, lon, height = geo.flatten()
        self.assertAlmostEqual(lat, 31.7483709921, places=3)
        self.assertAlmostEqual(lon, 117.9393085632, places=3)
        self.assertAlmostEqual(height, -128628.7493826561 / 0.3048, places=3)

    def test_enu2ecef(self):
        x, y, z = 2000, 8000, 0
        ecef = self.gps_utils.enu2ecef(x, y, z)
        self.assertIsInstance(ecef, np.ndarray)
        self.assertEqual(ecef.shape, (3, 1))
        x_ecef, y_ecef, z_ecef = ecef.flatten()
        self.assertAlmostEqual(x_ecef, -2701863.578333677, places=3)
        self.assertAlmostEqual(y_ecef, -4258001.65842185, places=3)
        self.assertAlmostEqual(z_ecef, 3892055.0031314525, places=3)

        x, y, z = 2000 / 0.3048, 8000 / 0.3048, 0
        ecef = self.ft_gps.enu2ecef(x, y, z)
        self.assertIsInstance(ecef, np.ndarray)
        self.assertEqual(ecef.shape, (3, 1))
        x_ecef, y_ecef, z_ecef = ecef.flatten()
        self.assertAlmostEqual(x_ecef, -2701863.578333677 / 0.3048, places=3)
        self.assertAlmostEqual(y_ecef, -4258001.65842185 / 0.3048, places=3)
        self.assertAlmostEqual(z_ecef, 3892055.0031314525 / 0.3048, places=3)

    def test_enu2geo(self):
        x, y, z = 2000, 8000, 5000
        geo = self.gps_utils.enu2geo(x, y, z)
        self.assertIsInstance(geo, np.ndarray)
        self.assertEqual(geo.shape, (3, 1))
        lat, lon, height = geo.flatten()
        self.assertAlmostEqual(lat, 37.8469177167, places=7)
        self.assertAlmostEqual(lon, -122.396694361 , places=7)
        self.assertAlmostEqual(height, 5015.3408790808, places=7)

        x, y, z = 2000 / 0.3048, 8000 / 0.3048, 5000 / 0.3048
        geo = self.ft_gps.enu2geo(x, y, z)
        self.assertIsInstance(geo, np.ndarray)
        self.assertEqual(geo.shape, (3, 1))
        lat, lon, height = geo.flatten()
        self.assertAlmostEqual(lat, 37.8469177167, places=7)
        self.assertAlmostEqual(lon, -122.396694361, places=7)
        self.assertAlmostEqual(height, 5015.3408790808 / 0.3048, places=7)

if __name__ == '__main__':
    # unittest.main()
    # trans = GPS_utils()
    # trans.setENUorigin(37.7749, -122.4194, 10)
    # lat, lon, height = 36.7749, -121.4194, 10
    # x, y, z = trans.geo2ecef(lat, lon, height).flatten()
    # print("ECEF:", x, y, z)
    # x_enu, y_enu, z_enu = trans.ecef2enu(x, y, z).flatten()
    # print("ENU:", x_enu, y_enu, z_enu)
    # a, b, c = trans.enu2ecef(2000, 8000, 10).flatten()
    # print("enu2ECEF:", a, b, c)
    # lat, lon, height = trans.ecef2geo(a, b, c).flatten()
    # print("ECEF2GEO:", lat, lon, height)
    unittest.main()
    # pass test !!