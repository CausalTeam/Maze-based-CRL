import numpy as np
import random
import tkinter as tk
import time
import gymnasium as gym
from gymnasium import spaces

class MazeEnv(gym.Env):
    def __init__(self, size=3, num_restrictions=0, num_rewards=0, seed=1, max_steps=100):
        super().__init__()
        random.seed(seed)
        self.max_steps=max_steps
        self.size = size
        self.maze = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start
        self.time = 0
        self.restrictions = self._generate_restrictions(num_restrictions)
        self.rewards = self._generate_rewards(num_rewards)
        self.sleep_time = 0.1
        print("Restrictions:", self.restrictions)
        print("Rewards:", self.rewards)
        self.triggered_rewards = set()  # 跟踪已触发的奖励

        self.action_space = spaces.Discrete(4)  # 上, 下, 左, 右
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # 创建 tkinter 窗口和画布
        self.window = tk.Tk()
        self.window.title("Maze Visualization")
        self.canvas_size = 400
        self.cell_size = self.canvas_size / self.size
        self.canvas = tk.Canvas(self.window, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

    def get_state(self):
        actions= [0, 1, 2, 3]  # 上, 下, 左, 右
        ret=[self.state[0],self.state[1],0,0,0,0]
        for action in actions:
            if (self.state, action) in self.restrictions:
                ret[action+2]=-self.restrictions[(self.state, action)]*10
            if (self.state, action) in self.rewards and (self.state, action) not in self.triggered_rewards:
                ret[action+2]=self.rewards[(self.state, action)]*10
        return ret

    def _get_next_state_no_restriction(self, state, action):
        x = state[0]
        y = state[1]
        if action == 0:  # 上
            x = x - 1
        elif action == 1:  # 下
            x = x + 1
        elif action == 2:  # 左
            y = y - 1
        elif action == 3:  # 右
            y = y + 1
        return x, y

    def _get_reverse_action(self, action):
        # 反向动作映射
        reverse_actions = {0: 1, 1: 0, 2: 3, 3: 2}
        return reverse_actions.get(action)


    def _generate_restrictions(self, num_restrictions):
        restrictions = {}
        while len(restrictions) < num_restrictions * 2:
            from_state = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            action = random.randint(0, 3)
            to_state = self._get_next_state_no_restriction(from_state, action)

            if self._is_valid_state(to_state):  # 检查 to_state 是否有效
                probability = random.uniform(0.2, 0.4)
                if (from_state, action) not in restrictions:
                    restrictions[(from_state, action)] = probability

                reverse_action = self._get_reverse_action(action)
                if reverse_action is not None and (to_state, reverse_action) not in restrictions:
                    restrictions[(to_state, reverse_action)] = probability

        return restrictions

    def _generate_rewards(self, num_rewards):
        rewards = {}
        while len(rewards) < num_rewards * 2:
            from_state = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            action = random.randint(0, 3)
            to_state = self._get_next_state_no_restriction(from_state, action)

            if self._is_valid_state(to_state):  # 检查 to_state 是否有效
                cur_reward = random.uniform(0.01, 0.2)
                if (from_state, action) not in self.restrictions and (from_state, action) not in rewards:
                    rewards[(from_state, action)] = cur_reward

                    reverse_action = self._get_reverse_action(action)
                    if reverse_action is not None and (to_state, reverse_action) not in self.restrictions:
                        rewards[(to_state, reverse_action)] = cur_reward

        return rewards

    def _is_valid_state(self, state):
        x, y = state
        return 0 <= x < self.size and 0 <= y < self.size

    def if_goal(self):
        return self.state == self.goal

    def reset(self):
        self.state = self.start
        self.time = 0
        self.triggered_rewards = set()  # 跟踪已触发的奖励
        return self.get_state()

    def step(self, action):
        # 定义动作
        # 0: 上, 1: 下, 2: 左, 3: 右
        x, y = self.state
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 下
            x = min(self.size - 1, x + 1)
        elif action == 2:  # 左
            y = max(0, y - 1)
        elif action == 3:  # 右
            y = min(self.size - 1, y + 1)

        self.last_state = self.state
        self.state = (x, y)



        # 检查是否触发限制条件
        restriction = (self.last_state, action)
        # print(restriction)
        # print(self.restrictions)
        if restriction in self.restrictions:
            if random.random() < self.restrictions[restriction]:
                # 触发限制条件
                return self.get_state(), -2, True, {}

                # 检查是否到达终点
        done = self.state == self.goal

        # 设置奖励
        reward = 2 if done else -0.01

        self.time += 1

        if (self.last_state, action) in self.rewards and (self.last_state, action) not in self.triggered_rewards:
            reward += self.rewards[(self.last_state, action)]
            self.triggered_rewards.add((self.last_state, action))
            self.triggered_rewards.add((self.state, self._get_reverse_action(action)))

        if self.time >= self.max_steps:
            done = True

        # if self.last_state == self.state:
        #     reward -= 0.5

        return self.get_state(), reward, done, {}


    def render(self):
        self.canvas.delete("all")  # 清除之前的绘制

        # 绘制迷宫的每个格子
        for i in range(self.size):
            for j in range(self.size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")

        # 标记限制和奖励
        for (from_state, action), prob in self.restrictions.items():
            self._draw_edge_dot(from_state, action, "red")
            self._draw_text_near_dot(from_state, action, str(round(prob, 2)), "red")

        for (from_state, action), reward in self.rewards.items():
            self._draw_edge_dot(from_state, action, "green")
            self._draw_text_near_dot(from_state, action, str(round(reward, 2)), "green")

        # 标记智能体和终点
        agent_x, agent_y = self.state
        goal_x, goal_y = self.goal
        self._draw_cell(agent_x, agent_y, "blue")
        self._draw_cell(goal_x, goal_y, "yellow")
        time.sleep(self.sleep_time)

        self.window.update()

    def _draw_text_near_dot(self, state, action, text, color):
        x, y = state
        cx, cy = (y + 0.5) * self.cell_size, (x + 0.5) * self.cell_size  # 格子中心

        if action == 0:  # 上
            cy -= self.cell_size / 2
        elif action == 1:  # 下
            cy += self.cell_size / 2
        elif action == 2:  # 左
            cx -= self.cell_size / 2
        elif action == 3:  # 右
            cx += self.cell_size / 2

        offset = self.cell_size / 5
        self.canvas.create_text(cx + offset, cy + offset, text=text, fill=color)

    def _draw_edge_dot(self, state, action, color):
        x, y = state
        cx, cy = (y + 0.5) * self.cell_size, (x + 0.5) * self.cell_size  # 格子中心

        if action == 0:  # 上
            cy -= self.cell_size / 2
        elif action == 1:  # 下
            cy += self.cell_size / 2
        elif action == 2:  # 左
            cx -= self.cell_size / 2
        elif action == 3:  # 右
            cx += self.cell_size / 2

        dot_size = self.cell_size / 10
        self.canvas.create_oval(cx - dot_size, cy - dot_size, cx + dot_size, cy + dot_size, fill=color)

    def _draw_cell(self, x, y, color):
        x1 = y * self.cell_size
        y1 = x * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def simulate_step(self, state, action):
        x, y = state
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 下
            x = min(self.size - 1, x + 1)
        elif action == 2:  # 左
            y = max(0, y - 1)
        elif action == 3:  # 右
            y = min(self.size - 1, y + 1)

        simulated_state = (x, y)
        done = simulated_state == self.goal

        reward = 2 if done else -0.01
        # 将 state 转换为元组
        state_tuple = (x, y)

        # 使用元组作为键来检查奖励
        if (state_tuple, action) in self.rewards:
            reward += self.rewards[(state_tuple, action)]

        # 检查限制条件
        if (state_tuple, action) in self.restrictions:
            if random.random() < self.restrictions[(state_tuple, action)]:
                # 触发限制条件
                return simulated_state, -2

        return simulated_state, reward
