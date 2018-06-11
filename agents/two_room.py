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
        self.experiment_id: str = 'two_room'
        self.reward_from_success = 0

        # QQQ todo: add subskill_attack & single_room
        self.supported_actions = [
            'move 1',
            'turn -1',
            'turn 1',
            'attack 1',
            'subskill_attack',
            'single_room',
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

        mission_file = './agents/domains/two_room.xml'
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

    def perform_action(self, action_command: str, is_train: bool) -> Tuple[float, bool, np.ndarray, bool, bool]:
        # overload super's perform_action() to check for subskill and attack actions

        ### IF we have already selected a subskill, continue executing until K = 5

        ### ELSE, Allow new subskill selection and execution:
        ### SUBSKILL EVOCATION: For 5 timesteps
        # QQ todo if subskill is selected, then activate that policy
        if action_command == 'subskill_attack':
            return super(Agent,self).perform_action('', is_train)
        elif action_command == 'single_room':
            return super(Agent,self).perform_action('', is_train)

        ### ATOMIC ACTIONS:
        elif action_command == 'attack 1':
            # Only allow an attack if we target the zombiepigman
            world_state = self.prev_state #self.agent_host.peekWorldState()
            los = json.loads(world_state.observations[-1].text).get(u'LineOfSight', 0)
            if los[u'type'] == "PigZombie" and los[u'inRange']:

                # if only discretemovement (see attack xml):
                return super(Agent,self).perform_action(action_command, is_train)
                # else if hybrid movement: (see attack xml)

            else:
                action_command = 'jump 1' # if it's not allowed, send 'jump 1' as a no-op action, so it still 'costs' the agent a command
                return super(Agent,self).perform_action(action_command, is_train)

        else: return super(Agent,self).perform_action(action_command, is_train)
    
    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool, bool]:  # returns: reward, terminal, state, timeout, success.
        self.prev_state = world_state        

        if action_command == 'attack 1':
            # Attacked the NPC successfuly (1st subskill), teleport agent to other room

            y = 55.0
            yaw = (float)(random.randint(0, 3) * 90)
            while True:
                # Ensure that we don't start right on the block.
                x = random.randint(0, 6) + 0.5
                z = random.randint(0, 6) + 0.5
                if x != 2.5 or z != 5.5:
                    break
            self.agent_host.sendCommand('chat /tp Cristina ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(yaw) + ' 0.0')

        if reward > 0:
            # Reached goal block successfully. (2nd subskill)
            return self.reward_from_success, True, state, False, True

        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if reward < -5:
            return -1, True, state, True, False

        return -1, False, state, False, False
