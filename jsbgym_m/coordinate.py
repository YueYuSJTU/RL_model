'''
MIT License
Copyright (c) 2019 Michail Kalaitzakis
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

用在此项目中，这个转换至少能保证1e-3m的精度
'''

import numpy as np

class GPS_utils:
	'''
		Contains the algorithms to convert a gps signal (longitude, latitude, height)
		to a local cartesian ENU system and vice versa
		
		Use setENUorigin(lat, lon, height) to set the local ENU coordinate system origin
		Use geo2enu(lat, lon, height) to get the position in the local ENU system
		Use enu2geo(x_enu, y_enu, z_enu) to get the latitude, longitude and height
	'''
	
	def __init__(self, unit = "m"):
		# Geodetic System WGS 84 axes
		self.a  = 6378137.0
		self.b  = 6356752.314245
		self.a2 = self.a * self.a
		self.b2 = self.b * self.b
		self.e2 = 1.0 - (self.b2 / self.a2)
		self.e  = self.e2 / (1.0 - self.e2)
		
		# Local ENU Origin
		self.latZero = None
		self.lonZero = None
		self.hgtZero = None
		self.xZero = None
		self.yZero = None
		self.zZero = None
		self.R = np.asmatrix(np.eye(3))
		
		self.unit = unit
		if unit not in ["m", "ft"]:
			raise ValueError("unit must be 'm' or 'ft'")
		self.unit_conversion = 1.0 if unit == "m" else 0.3048  # 1 ft = 0.3048 m

	def setENUorigin(self, lat, lon, height):
		# Save origin lat, lon, height
		self.latZero = lat
		self.lonZero = lon
		self.hgtZero = height * self.unit_conversion
		
		# Get origin ECEF X,Y,Z
		origin = self.geo2ecef(self.latZero, self.lonZero, height, fix=True)
		# origin must be in meters
		self.xZero = origin.item(0)
		self.yZero = origin.item(1)
		self.zZero = origin.item(2)
		self.oZero = np.array([[self.xZero], [self.yZero], [self.zZero]])
		
		# Build rotation matrix
		phi = np.deg2rad(self.latZero)
		lmd = np.deg2rad(self.lonZero)
		
		cPhi = np.cos(phi)
		cLmd = np.cos(lmd)
		sPhi = np.sin(phi)
		sLmd = np.sin(lmd)
		
		self.R[0, 0] = -sLmd
		self.R[0, 1] =  cLmd
		self.R[0, 2] =  0.0
		self.R[1, 0] = -sPhi * cLmd
		self.R[1, 1] = -sPhi * sLmd
		self.R[1, 2] =  cPhi
		self.R[2, 0] =  cPhi * cLmd
		self.R[2, 1] =  cPhi * sLmd
		self.R[2, 2] =  sPhi
	
	def geo2ecef(self, lat, lon, height, fix = False):
		# param: fix: if True, return the result in m
		phi = np.deg2rad(lat)
		lmd = np.deg2rad(lon)
		
		cPhi = np.cos(phi)
		cLmd = np.cos(lmd)
		sPhi = np.sin(phi)
		sLmd = np.sin(lmd)
		
		N = self.a / np.sqrt(1.0 - self.e2 * sPhi * sPhi)
		
		height = height * self.unit_conversion
		x = (N + height) * cPhi * cLmd
		y = (N + height) * cPhi * sLmd
		z = ((self.b2 / self.a2) * N + height) * sPhi

		result = np.array([[x], [y], [z]])
		if not fix:
			return result / self.unit_conversion
		
		return result
	
	def ecef2enu(self, x, y, z):
		x, y, z = x * self.unit_conversion, y * self.unit_conversion, z * self.unit_conversion
		ecef = np.array([[x], [y], [z]])
		
		return np.asarray(self.R * (ecef - self.oZero)) / self.unit_conversion
	
	def geo2enu(self, lat, lon, height):
		ecef = self.geo2ecef(lat, lon, height)
		
		return self.ecef2enu(ecef.item(0), ecef.item(1), ecef.item(2))
	
	def ecef2geo(self, x, y, z):
		x, y, z = x * self.unit_conversion, y * self.unit_conversion, z * self.unit_conversion
		p = np.sqrt(x*x + y*y)
		q = np.arctan2(self.a * z, self.b * p)
		
		sq = np.sin(q)
		cq = np.cos(q)
		
		sq3 = sq * sq * sq
		cq3 = cq * cq * cq
		
		phi = np.arctan2(z + self.e * self.b * sq3, p - self.e2 * self.a * cq3)
		lmd = np.arctan2(y, x)
		v = self.a / np.sqrt(1.0 - self.e2 * np.sin(phi) * np.sin(phi))

		lat = np.rad2deg(phi)
		lon = np.rad2deg(lmd)		
		h = ((p / np.cos(phi)) - v) / self.unit_conversion
		
		return np.array([[lat], [lon], [h]]) 
		
	def enu2ecef(self, x, y, z):
		x, y, z = x * self.unit_conversion, y * self.unit_conversion, z * self.unit_conversion
		lmd = np.deg2rad(self.latZero)
		phi = np.deg2rad(self.lonZero)
		
		cPhi = np.cos(phi)
		cLmd = np.cos(lmd)
		sPhi = np.sin(phi)
		sLmd = np.sin(lmd)
		
		N = self.a / np.sqrt(1.0 - self.e2 * sLmd * sLmd)
		
		x0 = (self.hgtZero + N) * cLmd * cPhi
		y0 = (self.hgtZero + N) * cLmd * sPhi
		z0 = (self.hgtZero + (1.0 - self.e2) * N) * sLmd
		
		xd = -sPhi * x - cPhi * sLmd * y + cLmd * cPhi * z
		yd =  cPhi * x - sPhi * sLmd * y + cLmd * sPhi * z
		zd =  cLmd * y + sLmd * z
		
		return np.array([[x0+xd], [y0+yd], [z0+zd]]) / self.unit_conversion
	
	def enu2geo(self, x, y, z):
		ecef = self.enu2ecef(x, y, z)
		
		return self.ecef2geo(ecef.item(0), ecef.item(1), ecef.item(2))
	

class GPS_NED(GPS_utils):
	'''
		Contains the algorithms to convert a gps signal (longitude, latitude, height)
		to a local cartesian NED system and vice versa
		
		Use setNEDorigin(lat, lon, height) to set the local NED coordinate system origin
		Use geo2ned(lat, lon, height) to get the position in the local NED system
		Use ned2geo(x_ned, y_ned, z_ned) to get the latitude, longitude and height
	'''
	
	def __init__(self, unit = "m"):
		super().__init__(unit)
	
	def setNEDorigin(self, lat, lon, height):
		self.setENUorigin(lat, lon, height)
	
	def geo2ned(self, lat, lon, height):
		x,y,z = self.geo2enu(lat, lon, height)
		return self.enu2ned(x, y, z)
	
	def ned2geo(self, x, y, z):
		x,y,z = self.ned2enu(x, y, z)
		return self.enu2geo(x, y, z)
	
	def ned2enu(self, x, y, z):
		return y, x, -z
	
	def enu2ned(self, x, y, z):
		return y, x, -z
	
	def ecef2ned(self, x, y, z):
		x,y,z = self.ecef2enu(x, y, z)
		return self.enu2ned(x, y, z)
	
	def ned2ecef(self, x, y, z):
		x,y,z = self.ned2enu(x, y, z)
		return self.enu2ecef(x, y, z)