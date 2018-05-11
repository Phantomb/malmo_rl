import argparse
import json
import logging
import random
import re
import time
from typing import Tuple

import numpy as np

from agents.agent import Agent as BaseAgent

# The attack subskill trains an agent to attack a (dynamic) target (a zombie pigman)
class Agent(BaseAgent):
    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        super(Agent, self).__init__(params, port, start_malmo, agent_index)

        # Experiment Parameters
        self.experiment_id: str = 'subskill_attack'
        self.reward_from_success = 0 # old value: 20
        self.supported_actions = [
            'move 1',
            'turn -1',
            'turn 1',
            'attack 1'
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
        
        
        mission_file = './agents/domains/subskill_attack.xml'
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
        # overload super's perform_action() to check for attack actions

        if action_command == 'attack 1':
            # Only allow an attack if we target the zombiepigman
            world_state = self.prev_state #self.agent_host.peekWorldState()
            los = json.loads(world_state.observations[-1].text).get(u'LineOfSight', 0)
            if los[u'type'] == "PigZombie" and los[u'inRange']:
                logging.error('Not an error: The agent can hit in the proper situation. Remove this message if it works.')

                # original super.perform_action code pasted here and edited since we need to execute two consecutive commands
                number_of_attempts = 0
                logging.debug('Agent[' + str(self.agent_index) + ']: received command ' + action_command)
                while True:
                    self.agent_host.sendCommand(action_command)
                    self.agent_host.sendCommand('attack 0')
                    reward, terminal, state, world_state, action_succeeded = self._get_new_state(False)
                    if action_succeeded:
                        return self._manual_reward_and_terminal(action_command, reward, terminal, state, world_state)

                    number_of_attempts += 1
                    time.sleep(self.tick_time / 1000.0)
                    if number_of_attempts >= 100:
                        logging.error('Agent[' + str(self.agent_index) + ']: Failed to send action.')
                        self.game_running = False
                        return 0, True, np.empty(0), True, False

            else:
                action_command = 'jump 1' # if it's not allowed, send 'jump 1' as a no-op action, so it still 'costs' the agent a command
                return super(Agent,self).perform_action(action_command, is_train)

        else: return super(Agent,self).perform_action(action_command, is_train)

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool, bool]:  # returns: reward, terminal, state, timeout, success
        self.prev_state = world_state
        
        if reward > 0:
            # Attacked target successfully.
            return self.reward_from_success, True, state, False, True

        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if reward < -5:
            return -1, True, state, True, False

        return -1, False, state, False, False