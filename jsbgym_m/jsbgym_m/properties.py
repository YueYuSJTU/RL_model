import math
import collections
from jsbgym_m import utils
from pyquaternion import Quaternion

class BoundedProperty(
    collections.namedtuple("BoundedProperty", ["name", "description", "min", "max"])
):
    def get_legal_name(self):
        return utils.AttributeFormatter.translate(self.name)


class Property(collections.namedtuple("Property", ["name", "description"])):
    def get_legal_name(self):
        return utils.AttributeFormatter.translate(self.name)


# position and attitude
altitude_sl_ft = BoundedProperty(
    "position/h-sl-ft", "altitude above mean sea level [ft]", -1400, 85000
)
pitch_rad = BoundedProperty(
    "attitude/pitch-rad", "pitch [rad]", -0.5 * math.pi, 0.5 * math.pi
)
roll_rad = BoundedProperty("attitude/roll-rad", "roll [rad]", -math.pi, math.pi)
psi_rad = BoundedProperty("attitude/psi-rad", "heading [rad]", -math.pi, math.pi)
heading_deg = BoundedProperty("attitude/psi-deg", "heading [deg]", 0, 360)
sideslip_deg = BoundedProperty("aero/beta-deg", "sideslip [deg]", -180, +180)
lat_geod_deg = BoundedProperty(
    "position/lat-geod-deg", "geocentric latitude [deg]", -90, 90
)
altitude_geod_ft = BoundedProperty(
    "position/geod-alt-ft", "altitude above ground level [ft]", 0, 85000
)
lng_geoc_deg = BoundedProperty(
    "position/long-gc-deg", "geodesic longitude [deg]", -180, 180
)
ecef_x_ft = Property("position/ecef-x-ft", "ECEF x-coordinate [ft]")
ecef_y_ft = Property("position/ecef-y-ft", "ECEF y-coordinate [ft]")
ecef_z_ft = Property("position/ecef-z-ft", "ECEF z-coordinate [ft]")
dist_travel_m = Property(
    "position/distance-from-start-mag-mt",
    "distance travelled from starting position [m]",
)
dist_travel_lon_m = Property(
    "position/distance-from-start-lon-mt",
    "logarithmic distance travelled from starting position [m]",
)
dist_travel_lat_m = Property(
    "position/distance-from-start-lat-mt",
    "lateral distance travelled from starting position [m]",
)
alpha_deg = BoundedProperty("aero/alpha-deg", "angle of attack [deg]", -180, +180)
beta_deg = BoundedProperty("aero/beta-deg", "sideslip [deg]", -180, +180)

# velocities
vtrue_fps = BoundedProperty(
    "velocities/vtrue-fps", "true airspeed [ft/s]", 0, 2200
)
u_fps = BoundedProperty(
    "velocities/u-fps", "body frame x-axis velocity [ft/s]", -2200, 2200
)
v_fps = BoundedProperty(
    "velocities/v-fps", "body frame y-axis velocity [ft/s]", -2200, 2200
)
w_fps = BoundedProperty(
    "velocities/w-fps", "body frame z-axis velocity [ft/s]", -2200, 2200
)
v_north_fps = BoundedProperty(
    "velocities/v-north-fps", "velocity true north [ft/s]", float("-inf"), float("+inf")
)
v_east_fps = BoundedProperty(
    "velocities/v-east-fps", "velocity east [ft/s]", float("-inf"), float("+inf")
)
v_down_fps = BoundedProperty(
    "velocities/v-down-fps", "velocity downwards [ft/s]", float("-inf"), float("+inf")
)
p_radps = BoundedProperty(
    "velocities/p-rad_sec", "roll rate [rad/s]", -2 * math.pi, 2 * math.pi
)
q_radps = BoundedProperty(
    "velocities/q-rad_sec", "pitch rate [rad/s]", -2 * math.pi, 2 * math.pi
)
r_radps = BoundedProperty(
    "velocities/r-rad_sec", "yaw rate [rad/s]", -2 * math.pi, 2 * math.pi
)
altitude_rate_fps = Property("velocities/h-dot-fps", "Rate of altitude change [ft/s]")

# accelerations
ax_fps2 = BoundedProperty(
    "accelerations/udot-ft_sec2", "body frame x-axis acceleration [ft/s^2]", float("-inf"), float("+inf")
)
ay_fps2 = BoundedProperty(
    "accelerations/vdot-ft_sec2", "body frame y-axis acceleration [ft/s^2]", float("-inf"), float("+inf")
)
az_fps2 = BoundedProperty(
    "accelerations/wdot-ft_sec2", "body frame z-axis acceleration [ft/s^2]", float("-inf"), float("+inf")
)
aroll_radps2 = BoundedProperty(
    "accelerations/pdot-rad_sec2", "roll acceleration [rad/s^2]", float("-inf"), float("+inf")
)
apitch_radps2 = BoundedProperty(
    "accelerations/qdot-rad_sec2", "pitch acceleration [rad/s^2]", float("-inf"), float("+inf")
)
ayaw_radps2 = BoundedProperty(
    "accelerations/rdot-rad_sec2", "yaw acceleration [rad/s^2]", float("-inf"), float("+inf")
)

# controls state
aileron_left = BoundedProperty(
    "fcs/left-aileron-pos-norm", "left aileron position, normalised", -1, 1
)
aileron_right = BoundedProperty(
    "fcs/right-aileron-pos-norm", "right aileron position, normalised", -1, 1
)
elevator = BoundedProperty(
    "fcs/elevator-pos-norm", "elevator position, normalised", -1, 1
)
rudder = BoundedProperty("fcs/rudder-pos-norm", "rudder position, normalised", -1, 1)
throttle = BoundedProperty(
    "fcs/throttle-pos-norm", "throttle position, normalised", 0, 1
)
throttle_Aug = BoundedProperty(
    "fcs/throttle-pos-norm", "throttle position, normalised, with Augmentation", 0, 2
)
gear = BoundedProperty("gear/gear-pos-norm", "landing gear position, normalised", 0, 1)

# engines
engine_running = Property("propulsion/engine/set-running", "engine running (0/1 bool)")
all_engine_running = Property(
    "propulsion/set-running", "set engine running (-1 for all engines)"
)
# engine_thrust_lbs = Property("propulsion/engine/thrust-lbs", "engine thrust [lb]")
engine_thrust_lbs = BoundedProperty(
    "propulsion/engine/thrust-lbs", "engine thrust [lb]", float("-inf"), float("+inf")
)
total_fuel = BoundedProperty(
    "propulsion/total-fuel-lbs", "total fuel on board [lb]", 0, 10000
)

# controls command
aileron_cmd = BoundedProperty(
    "fcs/aileron-cmd-norm", "aileron commanded position, normalised", -1.0, 1.0
)
elevator_cmd = BoundedProperty(
    "fcs/elevator-cmd-norm", "elevator commanded position, normalised", -1.0, 1.0
)
rudder_cmd = BoundedProperty(
    "fcs/rudder-cmd-norm", "rudder commanded position, normalised", -1.0, 1.0
)
throttle_cmd = BoundedProperty(
    "fcs/throttle-cmd-norm", "throttle commanded position, normalised", 0.0, 1.0
)
mixture_cmd = BoundedProperty(
    "fcs/mixture-cmd-norm", "engine mixture setting, normalised", 0.0, 1.0
)
throttle_1_cmd = BoundedProperty(
    "fcs/throttle-cmd-norm[1]", "throttle 1 commanded position, normalised", 0.0, 1.0
)
mixture_1_cmd = BoundedProperty(
    "fcs/mixture-cmd-norm[1]", "engine mixture 1 setting, normalised", 0.0, 1.0
)
gear_all_cmd = BoundedProperty(
    "gear/gear-cmd-norm", "all landing gear commanded position, normalised", 0, 1
)
teflap_position_norm = BoundedProperty(
    "fcs/tef-pos-norm", "trailing edge flap position, normalised", 0, 1
)
leflap_position_norm = BoundedProperty(
    "fcs/lef-pos-norm", "leading edge flap position, normalised", 0, 1
)
left_dht_rad = BoundedProperty(
    "fcs/dht-left-pos-rad", "left differential horizontal tail angle [rad]", -math.pi, math.pi
)
right_dht_rad = BoundedProperty(
    "fcs/dht-right-pos-rad", "right differential horizontal tail angle [rad]", -math.pi, math.pi
)
f16_engine_n2 = BoundedProperty(
    "propulsion/engine[0]/n2", "F16 engine N2 speed [rpm]", 0, 100000
)
starter_cmd = Property("propulsion/starter-cmd", "F16 engine starter command, bool")
cutoff_cmd = Property("propulsion/cutoff-cmd", "F16 engine cutoff command, bool")

# simulation
sim_dt = Property("simulation/dt", "JSBSim simulation timestep [s]")
sim_time_s = Property("simulation/sim-time-sec", "Simulation time [s]")

# initial conditions
initial_altitude_ft = Property("ic/h-sl-ft", "initial altitude MSL [ft]")
initial_terrain_altitude_ft = Property(
    "ic/terrain-elevation-ft", "initial terrain alt [ft]"
)
initial_longitude_geoc_deg = Property(
    "ic/long-gc-deg", "initial geocentric longitude [deg]"
)
initial_latitude_geod_deg = Property(
    "ic/lat-geod-deg", "initial geodesic latitude [deg]"
)
initial_u_fps = Property(
    "ic/u-fps", "body frame x-axis velocity; positive forward [ft/s]"
)
initial_v_fps = Property(
    "ic/v-fps", "body frame y-axis velocity; positive right [ft/s]"
)
initial_w_fps = Property("ic/w-fps", "body frame z-axis velocity; positive down [ft/s]")
initial_p_radps = Property("ic/p-rad_sec", "roll rate [rad/s]")
initial_q_radps = Property("ic/q-rad_sec", "pitch rate [rad/s]")
initial_r_radps = Property("ic/r-rad_sec", "yaw rate [rad/s]")
initial_roc_fpm = Property("ic/roc-fpm", "initial rate of climb [ft/min]")
initial_heading_deg = Property("ic/psi-true-deg", "initial (true) heading [deg]")


class Vector2(object):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def heading_deg(self):
        """Calculate heading in degrees of vector from origin"""
        heading_rad = math.atan2(self.x, self.y)
        heading_deg_normalised = (math.degrees(heading_rad) + 360) % 360
        return heading_deg_normalised

    @staticmethod
    def from_sim(sim: "simulation.Simulation") -> "Vector2":
        return Vector2(sim[v_east_fps], sim[v_north_fps])
    
    def Norm(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

class Vector3(object):
    def __init__(self, x: float, y: float, z: float):
        # 确保x, y, z是标量
        def to_scalar(val):
            if hasattr(val, 'item'):
                return val.item()  # 处理NumPy数组
            return float(val)  # 处理其他类型
            
        self.x = to_scalar(x)
        self.y = to_scalar(y)
        self.z = to_scalar(z)
    
    def Norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def get_xyz(self):
        return self.x, self.y, self.z
    
    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __str__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"
    
    @staticmethod
    def cal_angle(v1: "Vector3", v2: "Vector3") -> float:
        return math.acos((v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)/(v1.Norm()*v2.Norm()))
    
    @staticmethod
    def Eular2Vector3(psi: float, theta: float) -> "Vector3":
        x = math.cos(psi) * math.cos(theta)
        y = math.sin(psi) * math.cos(theta)
        z = math.sin(theta)
        return Vector3(x, y, z)
    
    def project_to_plane(self, plane: str) -> "Vector2":
        if plane == "xy":
            return Vector3(self.x, self.y, 0)
        elif plane == "yz":
            return Vector3(0, self.y, self.z)
        elif plane == "xz":
            return Vector3(self.x, 0, self.z)
        else:
            raise ValueError("Invalid plane. Choose from 'xy', 'yz', or 'xz'.")

def Eular2Quaternion(psi: float, theta: float, phi: float) -> "Quaternion":
    q = Quaternion(axis=[0, 0, 1], angle=psi) * Quaternion(axis=[0, 1, 0], angle=theta) * Quaternion(axis=[1, 0, 0], angle=phi)
    # cy = math.cos(psi * 0.5)
    # sy = math.sin(psi * 0.5)
    # cp = math.cos(theta * 0.5)
    # sp = math.sin(theta * 0.5)
    # cr = math.cos(phi * 0.5)
    # sr = math.sin(phi * 0.5)
    # p = Quaternion(w=cr * cp * cy + sr * sp * sy,
    #                x=sr * cp * cy - cr * sp * sy,
    #                y=cr * sp * cy + sr * cp * sy,
    #                z=cr * cp * sy - sr * sp * cy)
    return q

class GeodeticPosition(object):
    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat = latitude_deg
        self.lon = longitude_deg

    def heading_deg_to(self, destination: "GeodeticPosition") -> float:
        """Determines heading in degrees of course between self and destination"""
        difference_vector = destination - self
        return difference_vector.heading_deg()

    @staticmethod
    def from_sim(sim: "simulation.Simulation") -> "GeodeticPosition":
        """Return a GeodeticPosition object with lat and lon from simulation"""
        lat_deg = sim[lat_geod_deg]
        lon_deg = sim[lng_geoc_deg]
        return GeodeticPosition(lat_deg, lon_deg)

    def __sub__(self, other) -> Vector2:
        """Returns difference between two coords as (delta_lat, delta_long)"""
        return Vector2(self.lon - other.lon, self.lat - other.lat)
