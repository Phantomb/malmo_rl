import argparse

from parameters.dqn import parser as dqn_parser

parser: argparse = dqn_parser
# override default available actions with added skills
parser.add_argument('--available_actions', default=['move 1', 'turn -1', 'turn 1', 'skill_nav1', 'skill_nav2', 'skill_pickup', 'skill_place', 'skill_break'], nargs='+',
                    help='Space separated list of available actions. E.g. "\'move 1\' \'turn -1\'..."')

parser.add_argument('--experiment', default='tworoom', help='Choose from the \'tworoom\' or \'threeroom\' domain')