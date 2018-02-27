#!/usr/bin/env bash

cd C:\Github_Repos\tessler_Malmo_fork

python main.py dqn subskills --double_dqn --normalize_reward --verbose_prints

python main.py dqn subskills --double_dqn --normalize_reward --verbose_prints --malmo_ports 10001 --save_name skills_nav1

python main.py dqn single_room --double_dqn --normalize_reward --verbose_prints --malmo_ports 10001

# subskill pickup
python main.py qr_dqn subskills --number_of_atoms 200 --number_of_agents 1 --malmo_ports 10000 --retain_rgb --save_name skills_pickup_qr --double_dqn --verbose_prints --experiment pickup --success_replay_memory