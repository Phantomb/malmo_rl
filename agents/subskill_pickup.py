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
        self.experiment_id: str = 'subskill_pickup'
        self.reward_from_success = 0 # old value: 20
        self.supported_actions = [
            'move 1',
            'turn -1',
            'turn 1'
        ]

    def _restart_world(self, is_train: bool) -> None:
        del is_train

        self._initialize_malmo_communication()

        mission_file = './agents/domains/subskill_pickup.xml'
        with open(mission_file, 'r') as f:
            logging.debug('Agent[' + str(self.agent_index) + ']: Loading mission from %s.', mission_file)
            mission_xml = f.read()

            success = False
            while not success:
                mission = self._load_mission_from_xml(mission_xml)
                self._load_mission_from_missionspec(mission)
                success = self._wait_for_mission_to_begin()

            self.game_running = True

        self.touching_block: bool = False

        while True:
            # Ensure that we don't start directly next to the block that is centered on x=0.5 z=0.5
            # Possible x and z range from -3.5 to +4.5
            x = random.randint(-4, 4) + 0.5
            y = 227.0
            z = random.randint(-4, 4) + 0.5
            if (x < -0.5 or x > 1.5) or (z < -0.5 or z > 1.5):
                break

        turn_direction = random.randint(0, 3) * 90
        self.agent_host.sendCommand('chat /tp Agent ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(turn_direction) + ' 0.0')

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool, bool]:  # returns: reward, terminal, state, timeout, success
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        grid = observations.get(u'floor3x3', 0)
        yaw = super(Agent, self)._get_direction_from_yaw(observations.get(u'Yaw', 0))

        # Check if the agent was already touching the block
        if self.touching_block:
            # Check if the agent is facing the block
            if(((grid[10] == u'gold_block' and yaw == 'north')  or 
                (grid[14] == u'gold_block' and yaw == 'east' )  or 
                (grid[16] == u'gold_block' and yaw == 'south')  or
                (grid[12] == u'gold_block' and yaw == 'west' )) and
                action_command == 'move 1'):
                # If the agent executed dummy action 'move 1', the agent succeeded
                return self.reward_from_success, True, state, False, True

        self.touching_block:bool = (grid[10] == u'gold_block' or
                                    grid[14] == u'gold_block' or
                                    grid[16] == u'gold_block' or
                                    grid[12] == u'gold_block')
                
        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if reward < -5:
            return -1, True, state, True, False

        return -1, False, state, False, False