import argparse
import json
import logging
import random
import re
import time
from typing import Tuple

import numpy as np

from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        super(Agent, self).__init__(params, port, start_malmo, agent_index)

        # Experiment Parameters
        self.experiment_id: str = 'subskill_nav1'
        self.reward_from_success = 0 # old value: 20
        self.supported_actions = [
            'move 1',
            'turn -1',
            'turn 1'
        ]

    def _restart_world(self, is_train: bool) -> None:
        del is_train

        self._initialize_malmo_communication()
                
        # Set agent to random location and yaw, slightly lowered pitch to help see blocks near feet
        x = random.randint(-4, 4) + 0.5
        y = 227.0
        z = random.randint(-4, 4) + 0.5
        pitch = 25.0 
        yaw = random.randint(0, 3) * 90
        
        mission_file = './agents/domains/subskill_nav1.xml'
        with open(mission_file, 'r') as f:
            logging.debug('Agent[' + str(self.agent_index) + ']: Loading mission from %s.', mission_file)
            mission_xml = f.read()

            success = False
            while not success:
                mission = self._load_mission_from_xml(mission_xml)
                mission.startAtWithPitchAndYaw(x, y, z, pitch, yaw)
                
                self._load_mission_from_missionspec(mission)
                success = self._wait_for_mission_to_begin()

            self.game_running = True

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool, bool]:  # returns: reward, terminal, state, timeout, success
        del world_state

        if reward > 0:
            # Reached goal successfully.
            return self.reward_from_success, True, state, False, True

        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if reward < -5:
            return -1, True, state, True, False

        return -1, False, state, False, False
