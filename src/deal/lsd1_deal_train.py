#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> lsd1.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/10
@Desc   ：
==================================================
"""
from src.configs import lsd1_ideal_paarms
from src.train import mult_train_net, save_mods_logs
from src.utils import plot4dfs

if __name__ == '__main__':
    mods, logs = mult_train_net(**lsd1_ideal_paarms)

    path = '../../data/output/lsd1/ideal_train'

    plot4dfs(*logs, path=path)
    save_mods_logs(mods, logs, path)