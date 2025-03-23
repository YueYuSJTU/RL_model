import unittest
import math
import numpy as np
import sys
from pyquaternion import Quaternion
import os
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file_path))  # src目录
project_root = os.path.dirname(src_dir)  # 项目根目录
sys.path.insert(0, project_root)
import jsbgym_m.properties as prp
from jsbgym_m import rewards, utils
from jsbgym_m.assessors import Assessor, AssessorImpl
from jsbgym_m.aircraft import Aircraft, f16
from jsbgym_m.task_tracking import TrackingTask, Opponent
from jsbgym_m.tasks import Shaping
from jsbgym_m.coordinate import GPS_NED
from jsbgym_m.tests.stubs import SimStub, TransitioningSimStub


class TestTrackingTask(unittest.TestCase):
    default_shaping = Shaping.STANDARD
    default_episode_time_s = 6.0
    default_step_frequency_hz = 5.0
    default_aircraft = f16
    default_steps_remaining_non_terminal = 10
    default_positive_rewards = True

    PERFECT_POSITIVE_REWARD = 1.0
    MIDDLING_POSITIVE_REWARD = 0.5
    TERRIBLE_POSITIVE_REWARD = 0.0

    def setUp(self):
        self.task = self.make_task()
        sim = SimStub.make_valid_state_stub(self.task)
        # 初始化task的新一轮属性
        _ = self.task.observe_first_state(sim)

        self.dummy_action = np.asarray(
            [0 for _ in range(len(self.task.action_variables))]
        )

    def make_task(
        self,
        shaping_type: Shaping = default_shaping,
        episode_time_s: float = default_episode_time_s,
        step_frequency_hz: float = default_step_frequency_hz,
        aircraft: Aircraft = default_aircraft,
        positive_rewards: bool = True,
    ) -> TrackingTask:
        return TrackingTask(
            shaping_type=shaping_type,
            step_frequency_hz=step_frequency_hz,
            aircraft=aircraft,
            episode_time_s=episode_time_s,
            positive_rewards=positive_rewards,
        )

    def get_initial_sim_with_state(
        self,
        task: TrackingTask = None,
        time_terminal=False,
        self_x=0,
        self_y=0,
        self_alt=5000,
        oppo_x=3000,
        oppo_y=3000,
        oppo_alt=-500,
        oppo_heading=0,
        heading_deg=0,
        roll_rad=0.0,
        pitch_rad=0.0,
        psi_rad=0.0,
        speed=800,
        v_north_fps=0,
        v_east_fps=0,
        v_down_fps=0,
    ) -> SimStub:
        if task is None:
            task = self.task
        sim = SimStub.make_valid_state_stub(task)
        task.observe_first_state(sim)

        sim = self.modify_sim_to_state_(
            sim, task, time_terminal, self_x, self_y, self_alt, 
            oppo_x, oppo_y, oppo_alt, oppo_heading, heading_deg, roll_rad, pitch_rad,psi_rad,
            speed, v_north_fps, v_east_fps, v_down_fps
        )
        
        # 重置task的last_state属性
        state = task.State(*(sim[prop] for prop in task.state_variables))
        task.last_state = state
        return sim

    def modify_sim_to_state_(
        self,
        sim: SimStub,
        task: TrackingTask = None,
        steps_terminal=False,
        self_x=0,
        self_y=0,
        self_alt=5000,
        oppo_x=3000,
        oppo_y=3000,
        oppo_alt=-500,
        oppo_heading=0,
        heading_deg=0,
        roll_rad=0.0,
        pitch_rad=0.0,
        psi_rad=0.0,
        speed=800,
        v_north_fps=0,
        v_east_fps=0,
        v_down_fps=0,
    ) -> SimStub:
        if task is None:
            task = self.task

        if steps_terminal:
            sim[self.task.steps_left] = 0
        else:
            sim[self.task.steps_left] = self.default_steps_remaining_non_terminal

        # 设置自身飞机状态
        sim[task.ned_Xposition_ft] = self_x
        sim[task.ned_Yposition_ft] = self_y
        sim[prp.altitude_sl_ft] = self_alt
        sim[prp.heading_deg] = heading_deg
        sim[prp.roll_rad] = roll_rad
        sim[prp.pitch_rad] = pitch_rad
        sim[prp.psi_rad] = psi_rad
        
        # 根据heading设置速度分量
        heading_rad = math.radians(heading_deg)
        sim[prp.v_north_fps] = speed * math.cos(heading_rad)
        sim[prp.v_east_fps] = speed * math.sin(heading_rad)
        sim[prp.v_down_fps] = 0
        if v_down_fps != 0:
            sim[prp.v_down_fps] = v_down_fps
            sim[prp.v_north_fps] = v_north_fps
            sim[prp.v_east_fps] = v_east_fps
        
        # 设置对手飞机状态
        sim[task.oppo_x_ft] = oppo_x
        sim[task.oppo_y_ft] = oppo_y
        sim[task.oppo_altitude_sl_ft] = oppo_alt
        sim[task.oppo_heading_deg] = oppo_heading
        
        # 更新计算属性
        # task._update_custom_properties(sim)
        # task._update_extra_properties(sim)
        
        return sim

    def test_init_shaping_standard(self):
        task = self.make_task(shaping_type=Shaping.STANDARD)

        self.assertIsInstance(task.assessor, Assessor)
        self.assertEqual(1, len(task.assessor.base_components))
        self.assertEqual(0, len(task.assessor.potential_components))
        self.assertFalse(task.assessor.potential_components)  # 断言为空

    def test_get_initial_conditions_contains_all_props(self):
        ics = self.task.get_initial_conditions()

        self.assertIsInstance(ics, dict)
        for prop, value in self.task.base_initial_conditions.items():
            self.assertAlmostEqual(value, ics[prop])

        expected_ic_properties = [
            prp.initial_u_fps,
            prp.initial_v_fps,
            prp.initial_w_fps,
            prp.initial_p_radps,
            prp.initial_q_radps,
            prp.initial_r_radps,
            prp.initial_roc_fpm,
            prp.initial_heading_deg,
        ]
        for prop in expected_ic_properties:
            self.assertIn(
                prop,
                ics.keys(),
                msg=f"expected TrackingTask to set value for property {prop} but not found in ICs",
            )

    def test_observe_first_state_returns_valid_state(self):
        sim = SimStub.make_valid_state_stub(self.task)

        first_state = self.task.observe_first_state(sim)

        self.assertEqual(len(first_state), len(self.task.state_variables))
        self.assertIsInstance(first_state, tuple)

    def test_task_first_observation_inputs_controls(self):
        dummy_sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(dummy_sim)

        # 检查引擎状态符合预期
        self.assertAlmostEqual(self.task.THROTTLE_CMD, dummy_sim[prp.throttle_cmd])
        self.assertAlmostEqual(self.task.MIXTURE_CMD, dummy_sim[prp.mixture_cmd])
        self.assertAlmostEqual(1.0, dummy_sim[prp.engine_running])

    def test_task_step_correct_return_types(self):
        sim = SimStub.make_valid_state_stub(self.task)
        steps = 1
        _ = self.task.observe_first_state(sim)

        state, reward, is_terminal, truncated, info = self.task.task_step(
            sim, self.dummy_action, steps
        )

        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), len(self.task.state_variables))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_terminal, bool)
        self.assertIsInstance(info, dict)

    def test_task_step_returns_non_terminal_time_less_than_max(self):
        sim = self.get_initial_sim_with_state(self.task, time_terminal=False)
        _ = self.task.observe_first_state(sim)
        non_terminal_steps_left = 2
        sim[self.task.steps_left] = non_terminal_steps_left

        _, _, is_terminal, _, _ = self.task.task_step(sim, self.dummy_action, 1)

        self.assertFalse(is_terminal)

    def test_task_step_returns_terminal_steps_zero(self):
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)
        sim[self.task.steps_left] = 0
        steps = 1

        _, _, is_terminal, _, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertTrue(is_terminal)

    def test_distance_calculation(self):
        # 测试距离计算是否正确
        self_x, self_y, self_alt = 100, 200, 5000
        oppo_x, oppo_y, oppo_alt = 600, 800, -500
        
        sim = self.get_initial_sim_with_state(
            self.task, 
            time_terminal=False,
            self_x=self_x, 
            self_y=self_y, 
            self_alt=self_alt,
            oppo_x=oppo_x, 
            oppo_y=oppo_y, 
            oppo_alt=oppo_alt
        )
        self.task._update_extra_properties(sim)
        # print(f"Debug: ned_X: {sim[self.task.ned_Xposition_ft]}")
        
        # 计算预期的欧几里得距离
        expected_distance = math.sqrt(
            (oppo_x - self_x) ** 2 + 
            (oppo_y - self_y) ** 2 + 
            (oppo_alt - self_alt) ** 2
        )
        
        # 验证计算的距离是否正确
        self.assertAlmostEqual(
            expected_distance, 
            sim[self.task.distance_oppo_ft], 
            delta=0.1,
            msg="距离计算不正确"
        )

    def test_track_angle_calculation(self):
        # 测试跟踪角度计算是否正确
        self_x, self_y, self_alt = 0, 0, 5000
        oppo_x, oppo_y, oppo_alt = 1000, 1000, 5000  # 目标在正东方向
        sim = self.get_initial_sim_with_state(
            self.task,
            time_terminal=False,
            self_x=self_x,
            self_y=self_y,
            self_alt=self_alt,
            oppo_x=oppo_x,
            oppo_y=oppo_y,
            oppo_alt=oppo_alt
        )
        self.task._update_extra_properties(sim)
        excepted_angle = math.radians(45)
        self.assertAlmostEqual(
            excepted_angle,
            sim[self.task.track_angle_rad],
            msg="跟踪角度计算不正确"
        )
        # print(f"Debug: oppo_x: {oppo_x}, oppo_y: {oppo_y}, oppo_alt: {oppo_alt}")
        # print(f"Debug: oppo_heading: {sim[self.task.oppo_heading_deg]}, oppo_pitch: {sim[self.task.oppo_pitch_rad]}, oppo_roll: {sim[self.task.oppo_roll_rad]}")
        self.assertAlmostEqual(
            excepted_angle,
            sim[self.task.adverse_angle_rad]
        )


        oppo_x, oppo_y, oppo_alt = 1000, 2000, 10000
        heading_deg = 0 
        
        sim = self.get_initial_sim_with_state(
            self.task,
            time_terminal=False,
            self_x=self_x,
            self_y=self_y,
            self_alt=self_alt,
            oppo_x=oppo_x,
            oppo_y=oppo_y,
            oppo_alt=oppo_alt,
            heading_deg=heading_deg
        )
        self.task._update_extra_properties(sim)
        expected_angle = math.acos(1 / math.sqrt(30))
        
        # 验证计算的角度是否接近预期值
        self.assertAlmostEqual(
            expected_angle,
            sim[self.task.track_angle_rad],
            msg="跟踪角度计算不正确"
        )

        # 飞机有5度pitch的情形
        pitch_rad = math.radians(5)
        oppo_x, oppo_y, oppo_alt = 1000, 0, 5000
        sim = self.get_initial_sim_with_state(
            self.task,
            time_terminal=False,
            self_x=self_x,
            self_y=self_y,
            self_alt=self_alt,
            oppo_x=oppo_x,
            oppo_y=oppo_y,
            oppo_alt=oppo_alt,
            pitch_rad=pitch_rad
        )
        self.task._update_extra_properties(sim)
        self.assertAlmostEqual(
            pitch_rad,
            sim[self.task.track_angle_rad],
            msg="跟踪角度计算不正确"
        )

    def test_pointMass_calculation(self):
        # 测试点质量方位角计算是否正确
        self_x, self_y, self_alt = 0, 0, 5000
        oppo_x, oppo_y, oppo_alt = 1000, 0, 5000  # 目标在正北方向
        
        sim = self.get_initial_sim_with_state(
            self.task, 
            time_terminal=False,
            self_x=self_x, 
            self_y=self_y, 
            self_alt=self_alt,
            oppo_x=oppo_x, 
            oppo_y=oppo_y, 
            oppo_alt=oppo_alt
        )
        self.task._update_extra_properties(sim)
        
        # 目标在正东方向，预期方位角为90度(π/2)
        expected_bearing = math.radians(0)
        excepted_elevation = math.radians(0)
        
        # 验证计算的方位角是否接近预期值
        self.assertAlmostEqual(
            expected_bearing, 
            sim[self.task.bearing_pointMass_rad], 
            msg="点质量方位角计算不正确"
        )
        self.assertAlmostEqual(
            excepted_elevation,
            sim[self.task.elevation_pointMass_rad],
            msg="点质量仰角计算不正确"
        )

        expected_bearing = math.radians(35.8)
        expected_elevation = math.radians(14.5)
        r = 1000

        oppo_x = r * math.cos(expected_elevation) * math.cos(expected_bearing)
        oppo_y = r * math.cos(expected_elevation) * math.sin(expected_bearing)
        oppo_alt = self_alt + r * math.sin(expected_elevation)
        # 期望psi和pitch不影响
        psi_rad = math.radians(10)
        pitch_rad = math.radians(5)
        sim = self.get_initial_sim_with_state(
            self.task,
            time_terminal=False,
            self_x=self_x,
            self_y=self_y,
            self_alt=self_alt,
            oppo_x=oppo_x,
            oppo_y=oppo_y,
            oppo_alt=oppo_alt,
            psi_rad=psi_rad,
            roll_rad=0,
            pitch_rad=pitch_rad
        )
        self.task._update_extra_properties(sim)

        self.assertAlmostEqual(
            expected_bearing, 
            sim[self.task.bearing_pointMass_rad], 
            msg="点质量方位角计算不正确"
        )
        self.assertAlmostEqual(
            math.degrees(expected_elevation),
            math.degrees(sim[self.task.elevation_pointMass_rad]),
            msg="点质量仰角计算不正确"
        )
        

    def test_bearing_accountingRollPitch_calculation(self):
        # 测试考虑滚转和俯仰的方位角计算
        self_x, self_y, self_alt = 0, 0, 5000
        oppo_x, oppo_y, oppo_alt = 1000, 0, 5000
        roll_rad = math.radians(0)  # 30度滚转
        pitch_rad = math.radians(10)  # 10度俯仰
        
        sim = self.get_initial_sim_with_state(
            self.task,
            time_terminal=False,
            self_x=self_x,
            self_y=self_y,
            self_alt=self_alt,
            oppo_x=oppo_x,
            oppo_y=oppo_y,
            oppo_alt=oppo_alt,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad
        )
        self.task._update_extra_properties(sim)
        
        self.assertAlmostEqual(
            pitch_rad,
            sim[self.task.elevation_accountingRollPitch_rad],
            msg="考虑滚转和俯仰的方位角计算不正确"
        )

        psi_rad = math.radians(30)
        pitch_rad = math.radians(40)
        roll_rad = math.radians(60)     # 应当无影响
        expected_bearing = math.radians(30)
        expected_elevation = math.radians(40)
        r = 1000

        oppo_x = r * math.cos(expected_elevation) * math.cos(expected_bearing)
        oppo_y = r * math.cos(expected_elevation) * math.sin(expected_bearing)
        oppo_alt = self_alt + r * math.sin(expected_elevation)

        sim = self.get_initial_sim_with_state(
            self.task,
            time_terminal=False,
            self_x=self_x,
            self_y=self_y,
            self_alt=self_alt,
            oppo_x=oppo_x,
            oppo_y=oppo_y,
            oppo_alt=oppo_alt,
            psi_rad=psi_rad,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad
        )
        self.task._update_extra_properties(sim)

        self.assertAlmostEqual(
            0,
            sim[self.task.bearing_accountingRollPitch_rad],
        )
        self.assertAlmostEqual(
            0,
            sim[self.task.elevation_accountingRollPitch_rad],
        )
        
    def test_opponent_state_update(self):
        # 测试对手飞机状态更新
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)
        
        # 记录初始状态
        initial_x = sim[self.task.oppo_x_ft]
        initial_y = sim[self.task.oppo_y_ft]
        
        # 模拟一步后状态更新
        steps = 1
        _, _, _, _, _ = self.task.task_step(sim, self.dummy_action, steps)
        
        # 对手应该移动了
        self.assertNotEqual(initial_x, sim[self.task.oppo_x_ft])
        self.assertNotEqual(initial_y, sim[self.task.oppo_y_ft])

    def test_opponent_reset(self):
        # 测试对手重置功能
        opponent = Opponent()
        
        # 记录初始状态
        opponent.reset()
        initial_point = opponent.init_point
        initial_direction = opponent.init_direction
        initial_speed = opponent.init_speed
        
        # 再次重置
        opponent.reset()
        
        # 重置后应该有新的随机值
        self.assertNotEqual(initial_point, opponent.init_point)
        self.assertNotEqual(initial_direction, opponent.init_direction)
        self.assertNotEqual(initial_speed, opponent.init_speed)

    def test_opponent_step(self):
        # 测试对手step方法
        opponent = Opponent()
        opponent.reset()
        
        # 记录初始状态
        initial_state = opponent.get_state()
        
        # 模拟一步
        frequency = 5.0
        new_state = opponent.step(frequency)
        
        # 位置应该发生变化
        self.assertNotEqual(initial_state["x_position_ft"], new_state["x_position_ft"])
        self.assertNotEqual(initial_state["y_position_ft"], new_state["y_position_ft"])
        
        # 验证位置变化符合运动方向和速度
        time = 1 / frequency
        expected_x = initial_state["x_position_ft"] + initial_state["u_fps"] * time * math.cos(math.radians(initial_state["heading_deg"]))
        expected_y = initial_state["y_position_ft"] + initial_state["u_fps"] * time * math.sin(math.radians(initial_state["heading_deg"]))
        
        self.assertAlmostEqual(expected_x, new_state["x_position_ft"], delta=0.1)
        self.assertAlmostEqual(expected_y, new_state["y_position_ft"], delta=0.1)

    def test_quaternion_rotation(self):
        # 测试四元数旋转计算
        # 这是对bearing_accountingRollPitch_rad和elevation_accountingRollPitch_rad计算的底层验证
        
        # 创建一个相对位置向量
        dlt_x, dlt_y, dlt_z = 1000, 500, 200
        R = Quaternion(0, dlt_x, dlt_y, dlt_z)
        
        # 创建一个旋转四元数 (假设绕Z轴旋转45度)
        Q = Quaternion(axis=[0, 0, 1], radians=math.radians(45))
        
        # 执行旋转
        Rb = Q.inverse * R * Q
        rbx, rby, rbz = Rb.vector
        
        # 验证旋转后的向量与原向量不同
        self.assertNotEqual(dlt_x, rbx)
        self.assertNotEqual(dlt_y, rby)
        self.assertEqual(dlt_z, rbz)  # z轴旋转不应改变z分量
        
        # 计算方位角和仰角
        bearing = math.atan2(rby, rbx)
        elevation = math.atan2(rbz, math.sqrt(rbx**2 + rby**2))
        
        # 验证计算的角度是合理的
        self.assertTrue(-math.pi <= bearing <= math.pi)
        self.assertTrue(-math.pi/2 <= elevation <= math.pi/2)

if __name__ == "__main__":
    unittest.main()
