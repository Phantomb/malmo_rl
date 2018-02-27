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
        self.experiment_id: str = 'subskills'

        # self.reward_from_timeout_regex = re.compile(
        #    '<Reward.*description.*=.*\"command_quota_reached\".*reward.*=.*\"(.*[0-9]*)\".*/>', re.I)
        # self.reward_for_sending_command_regex = re.compile('<RewardForSendingCommand.*reward="(.*)"/>', re.I)

        self.supported_actions = [
            'move 1',
            'turn -1',
            'turn 1'
        ]

        # Experiment Parameters
        self.reward_from_success = 20
        self.max_number_of_steps = 30 # episode length is 30 for single DSNs, 60 for 2-room, 100 for 3-room
        
        # Global Variables
        self.number_of_steps = 0

    def _restart_world(self, is_train: bool) -> None:
        del is_train

        self._initialize_malmo_communication()

        mission_file = './agents/domains/subskills.xml'
        with open(mission_file, 'r') as f:
            logging.debug('Agent[' + str(self.agent_index) + ']: Loading mission from %s.', mission_file)
            mission_xml = f.read()

            success = False
            while not success:
                mission = self._load_mission_from_xml(mission_xml)
                self._load_mission_from_missionspec(mission)
                success = self._wait_for_mission_to_begin()

            self.game_running = True

        # Set agent to random location, which depends on the current experiment
        if self.params.experiment == "nav1":
            x = random.randint(-4,4) + 0.5
            y = 227.0
            z = random.randint(-4,4) + 0.5
        elif self.params.experiment == "pickup":
            self.touching_block = False
            while True:
                # Ensure that we don't start directly next to the block that is centered on x=0.5 z=26.5
                x = random.randint(-4, 4) + 0.5
                y = 227.0
                z = random.randint(22, 30) + 0.5
                if (x < -1.5 or x > 1.5) or (z < 25.5 or z > 27.5):
                    break
        # QQQ TODO: Other subskill domains
        else:
            logging.error("The following provided experiment type is not recognised:", self.params.experiment)

        # logging.debug('Agent[' + str(self.agent_index) + ']:  restarting at position x: ' + str(x) + ' z: ' + str(z))
        self.agent_host.sendCommand('chat /tp Cristina ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' -180.0 0.0')
        time.sleep(2)
        # Generate random int between 0 and 3
        turn_direction = random.randint(0, 3)
        self.agent_host.sendCommand('turn ' + str(turn_direction))

        self.number_of_steps = 0

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool, bool]:  # returns: reward, terminal, state, timeout, success
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        grid = observations.get(u'floor3x3', 0)

        self.number_of_steps += 1

        # Check if we have succeeded in finding the goal
        # Skills_nav1:
        if self.params.experiment == "nav1":
            if(grid[4] == u'gold_block'):
                return self.reward_from_success, True, state, False, True
        # Skills_pickup:
        elif self.params.experiment == "pickup":
            if (grid[10] == u'gold_block' or
                        grid[14] == u'gold_block' or
                        grid[16] == u'gold_block' or
                        grid[12] == u'gold_block'):
                # If the agent was already in the correct location (touching the block) and we execute dummy action 'move 1', the agent succeeded
                if self.touching_block and action_command == 'move 1':
                    self.touching_block = False
                    return self.reward_from_success, True, state, False, True
                self.touching_block = True
            else:
                self.touching_block = False
        # QQQ TODO: Other subskill domains
        else:
            logging.error("The following provided experiment type is not recognised:", self.params.experiment)


        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if self.number_of_steps % self.max_number_of_steps == 0:
            return -1, True, state, True, False

        return -1, False, state, False, False


    # TODO: Determine the parameters, settings and conditions based on the skills we're currently training or testing
    # This contains the reward, the maximum number of steps, the initialisation (position, yaw, items it has), the checking for success.