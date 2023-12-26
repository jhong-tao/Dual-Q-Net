#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> hsd1_pre_train.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/10
@Desc   ：
==================================================
"""
from src.configs import hsd1_pre_paarms
from src.train import mult_train_net, save_mods_logs
from src.utils import plot4dfs

if __name__ == '__main__':
    mods, logs = mult_train_net(**hsd1_pre_paarms)

    path = '../../data/output/hsd1/pre_train'

    plot4dfs(*logs, path=path)
    save_mods_logs(mods, logs, path)