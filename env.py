

from dotenv import load_dotenv  # Add this
from openai import OpenAI
import os
import random
from models import EdgeObservation, EdgeAction, EdgeReward

class EdgeEnv:
    def __init__(self):
        self.task_name = "task_stable_edge"
        self.max_steps = 15 # Increased slightly for better trajectory data
        self.current_step = 0
        self._state = self._get_initial_state()

    def _get_initial_state(self) -> EdgeObservation:
        if self.task_name == "task_stable_edge":
            return EdgeObservation(ram_usage=45.0, battery_level=100.0, latency=20.0, active_processes=5)
        elif self.task_name == "task_fluctuating_memory":
            return EdgeObservation(ram_usage=65.0, battery_level=85.0, latency=35.0, active_processes=12)
        elif self.task_name == "task_low_battery_spike":
            return EdgeObservation(ram_usage=80.0, battery_level=20.0, latency=45.0, active_processes=18)
        return EdgeObservation(ram_usage=50.0, battery_level=100.0, latency=20.0, active_processes=5)

    def set_task(self, task_name: str):
        self.task_name = task_name
        self.reset()

    def reset(self) -> EdgeObservation:
        self.current_step = 0
        self._state = self._get_initial_state()
        return self._state

    def state(self) -> EdgeObservation:
        return self._state

    def step(self, action: EdgeAction) -> tuple[EdgeObservation, EdgeReward, bool, dict]:
        self.current_step += 1
        
        # Action execution with slight stochastic variances (real-world simulation)
        if action.kill_process and self._state.active_processes > 0:
            self._state.active_processes -= 1
            self._state.ram_usage -= random.uniform(10.0, 15.0)
            
        if action.throttle_cpu:
            self._state.battery_level += random.uniform(1.0, 3.0)
            self._state.latency += random.uniform(10.0, 20.0)
            
        if action.route_to_cloud:
            self._state.ram_usage -= random.uniform(20.0, 30.0)
            self._state.latency += random.uniform(40.0, 60.0)
            self._state.battery_level -= random.uniform(2.0, 5.0)

        # Natural environmental drift
        self._state.battery_level -= random.uniform(1.0, 3.0)
        self._state.ram_usage += random.uniform(3.0, 8.0)
        # Latency naturally recovers slightly if not pushed
        if not action.throttle_cpu and not action.route_to_cloud:
            self._state.latency = max(20.0, self._state.latency - 5.0)

        # Clamp values
        self._state.battery_level = max(0.0, min(100.0, self._state.battery_level))
        self._state.ram_usage = max(0.0, min(100.0, self._state.ram_usage))
        self._state.latency = max(5.0, min(500.0, self._state.latency))

        # Terminal conditions
        done = self.current_step >= self.max_steps or self._state.battery_level <= 0.0 or self._state.ram_usage >= 100.0
        
        # Calculate shaped reward
        final_score = self._grade_task()

        return self._state, EdgeReward(score=final_score), done, {}

    def _grade_task(self) -> float:
        # Failsafe: System crash = 0 score
        if self._state.ram_usage >= 100.0 or self._state.battery_level <= 0.0:
            return 0.0

        # Continuous reward shaping based on task priorities
        ram_health = 1.0 - (self._state.ram_usage / 100.0)
        battery_health = self._state.battery_level / 100.0
        latency_health = max(0.0, 1.0 - (self._state.latency / 200.0)) # Latency over 200ms is 0 health

        if self.task_name == "task_stable_edge":
            # 100% focus on RAM
            score = ram_health
        
        elif self.task_name == "task_fluctuating_memory":
            # 70% focus on RAM, 30% focus on keeping latency down
            score = (ram_health * 0.7) + (latency_health * 0.3)
            
        elif self.task_name == "task_low_battery_spike":
            # Hard: Must balance all three constraints
            score = (battery_health * 0.5) + (ram_health * 0.3) + (latency_health * 0.2)
            
        return max(0.0, min(1.0, score))