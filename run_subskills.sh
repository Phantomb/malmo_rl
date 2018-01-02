#!/usr/bin/env bash

cd C:\Github_Repos\tessler_Malmo_fork

python main.py dqn subskills --double_dqn --normalize_reward --verbose_prints

python main.py dqn subskills --double_dqn --normalize_reward --verbose_prints --malmo_ports 10001
