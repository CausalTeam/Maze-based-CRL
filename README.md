# Maze-based-CRL
## 简介
Maze based causal reinforcement learning (Maze-based-CRL)是一种因果强化学习算法，将因果知识融入到演员-评论家模型中。通过使用因果洞察来进行更智能的行动选择和状态评估，从而提高战略准确性，简化学习过程。Maze-based-CRL中对因果关系的关注有助于更准确地预测未来的行动和决策。Maze-based-CRL的方法论在克服由稀疏奖励和复杂因果结构在强化学习中带来的挑战方面证明是有效的。它提高了学习效率和战略精度，同时也改善了决策的透明度和可解释性。
## 环境安装
推荐使用conda安装虚拟环境，推荐使用ubuntu系统，在命令行中运行：
```bash
conda create -n mbcrl python=3.9
conda activate mbcrl
pip install torch torchvision torchaudio
pip install gymnasium
```
## 迷宫环境介绍
本研究中的实验环境为迷宫环境，包括由网格构成的迷宫，学习代理在其中导航以避开限制并收集奖励。在此基础上，模拟了三种不同的实验设置：小型、中型和大型。此环境的基本特征包括：
#### 状态表示
状态以网格中的坐标 (x, y) 表示。起点在 (0, 0)，终点在 (size−1, size−1)。移动通过x（向下）或y（向右）的增量来量化。为简化，状态表示为 (x, y)，但代理能够在所有四个基本方向评估限制概率和奖励大小。
#### 行动空间
代理能够在网格中向上、向下、向左或向右移动。
#### 动态元素
迷宫包括作为动态元素的限制和奖励。与特定状态转换相关的限制具有终止游戏的概率。奖励同样与某些转换相关联，在穿越时提供积极的强化。
#### 奖励结构
奖励配置如下：

(a)达到目标：奖励 2。

(b)每一步：因路径效率而受到 -0.01 的惩罚。

(c)与奖励相关的状态转换：奖励与转换的奖励大小相匹配。

(d)与限制相关的状态转换：如果导致游戏终止，惩罚 -2。
## 运行
首先，修改main.py中的最后两行，调整环境设置以及训练策略

```python
env = MazeEnv(size=8, num_restrictions=8, num_rewards=8, seed=1, max_steps=64)
train_and_evaluate_rl_model(env, num_collect_episodes=100, num_train_episodes=1000, eval_interval=10, prior_prob=0.5, prior_func=prior_function, batch_size=64, learning_rate=0.00001, if_render=False, if_pretrain=True)
```

其中，MazeEnv的size对应环境大小，建议环境大小的平方与num_restrictions（限制数量）、num_rewards（奖励数量）保持一致。train_and_evaluate_rl_model传递的参数建议保持默认。

其次，运行：
```bash
python main.py
```
