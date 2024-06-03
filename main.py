import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from env import MazeEnv
import random
from torch.utils.data import DataLoader, TensorDataset

# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 新增的隐藏层
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x1 = torch.relu(self.fc2(x1))  # 通过新增的隐藏层
        action_prob = torch.softmax(self.actor(x1), dim=-1)
        x2 = torch.relu(self.fc3(x1))
        state_value = self.critic(x2)
        return action_prob, state_value



def collect_data(env, num_episodes, prior_prob, prior_func=None, policy=None):
    actions = [0, 1, 2, 3]  # 上, 下, 左, 右
    data = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # 根据概率使用先验函数
            if np.random.rand() < prior_prob and prior_func is not None:
                action = prior_func(env)
            else:
                action = policy(env) if policy else random.choice(actions)  # 使用策略或随机动作

            next_state, reward, done, _ = env.step(action)
            data.append((state, action, reward, next_state, done))
            state = next_state
    return data


def prior_function(env):
    current_state = env.state
    actions = [0, 1, 2, 3]  # 上, 下, 左, 右
    probabilities = [0.05, 0.45, 0.05, 0.45]  # 初始概率，倾向于向右和向下移动
    if current_state[0] == 0:
        probabilities[0] = 0.0  # 如果在最上面一行，则不会向上移动
    if current_state[0] == env.size - 1:
        probabilities[1] = 0.0  # 如果在最下面一行，则不会向下移动
    if current_state[1] == 0:
        probabilities[2] = 0.0  # 如果在最左边一列，则不会向左移动
    if current_state[1] == env.size - 1:
        probabilities[3] = 0.0  # 如果在最右边一列，则不会向右移动
    #print(probabilities)
    # 调整动作概率以避免限制和寻找奖励
    for i, action in enumerate(actions):
        next_state = env._get_next_state_no_restriction(current_state, action)
        restriction_key = (current_state, action)
        reward_key = (current_state, action)

        # 如果动作会导致触发限制，则根据限制概率减小该动作的概率
        if restriction_key in env.restrictions:
            restriction_prob = env.restrictions[restriction_key]
            probabilities[i] *= (1 - restriction_prob * 3) * 0.5

        # 如果动作会导致获得奖励，则根据奖励大小增加该动作的概率
        if reward_key in env.rewards and reward_key not in env.triggered_rewards:
            reward_value = env.rewards[reward_key]
            probabilities[i] += reward_value * 5
            probabilities[i] *= (1 + reward_value*3)

    # 根据调整后的概率选择动作
    total_prob = sum(probabilities)
    probabilities = [p / total_prob for p in probabilities]  # 归一化概率
    action = random.choices(actions, weights=probabilities, k=1)[0]
    return action

def prior_function2(env):
    return random.choices([0, 1, 2, 3],k=1)[0]

def evaluate_model(env, model, device, num_episodes=5, if_render=False):
    total_reward = 0.0
    goal_times = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # 将状态转换为长度为 2 的一维数组
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, _ = model(state_tensor)

            # 使用概率分布随机选择动作
            distribution = Categorical(action_probs)
            action = distribution.sample().item()

            next_state, reward, done, _ = env.step(action)
            if if_render:
                env.render()
            state = next_state  # 更新状态
            total_reward += reward
        if env.if_goal():
            goal_times += 1

    return total_reward / num_episodes, goal_times / num_episodes



def train_offline(env, model, optimizer, device, data, discount_factor=0.99, batch_size=64):
    # 转换数据为张量
    states, actions, rewards, next_states, dones = zip(*data)
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, device=device)
    rewards_tensor = torch.tensor(rewards, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones_tensor = torch.tensor([float(done) for done in dones], device=device)

    # 创建TensorDataset和DataLoader
    dataset = TensorDataset(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0  # 累积损失
    total_reward = 0.0  # 累积奖励
    total_batches = 0  # 批次计数

    # 训练循环
    for state_batch, action_batch, reward_batch, next_state_batch, done_batch in data_loader:
        # 前向传播
        action_probs, values = model(state_batch)
        _, next_values = model(next_state_batch)
        distribution = Categorical(action_probs)

        # TD目标和TD误差
        td_targets = reward_batch + discount_factor * next_values * (1 - done_batch)
        td_errors = td_targets - values.squeeze()

        # Actor和Critic的损失
        actor_loss = -distribution.log_prob(action_batch) * td_errors.detach()
        critic_loss = td_errors.pow(2).mean()
        # print(actor_loss)
        # print(critic_loss)
        # 反向传播和优化
        loss = actor_loss.mean() + critic_loss * 0.1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reward += reward_batch.sum().item()  # 累加奖励
        total_batches += 1

    average_loss = total_loss / total_batches
    average_reward = total_reward / total_batches

    return average_loss, average_reward

def generate_critic_pretraining_data(env_size, num_samples=10000):
    # 生成状态和对应的目标值
    states_int = np.random.randint(0, env_size, (num_samples, 2))
    states_float = np.random.rand(num_samples, 4) * 2 - 1  # 生成随机浮点数
    states = np.hstack((states_int, states_float))  # 水平堆叠两个数组

    targets = (states[:, 0] + states[:, 1]) / env_size
    return states, targets

def pretrain_critic(fc1, fc2, critic, states, targets, optimizer, device, batch_size=256, epochs=1):
    # 转换为张量
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

    # 创建数据加载器
    dataset = TensorDataset(states_tensor, targets_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for state_batch, target_batch in data_loader:
            # 前向传播
            tmp = torch.relu(fc1(state_batch))
            tmp = torch.relu(fc2(tmp))
            predicted_values = critic(tmp)


            # 计算损失
            loss = torch.nn.functional.mse_loss(predicted_values, target_batch)*1
            total_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")


def train_and_evaluate_rl_model(env, num_collect_episodes, num_train_episodes, eval_interval, learning_rate=0.0001, batch_size=256, discount_factor=0.85, prior_prob=0.5, prior_func=None, if_render=False, if_pretrain=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(state_size=6, action_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    if if_pretrain:
        states, targets = generate_critic_pretraining_data(env.size,2000000)

        for param in model.actor.parameters():
            param.requires_grad = False
        pretrain_critic(model.fc1, model.fc2, model.critic, states, targets, optimizer, device, batch_size=256)

        for param in model.actor.parameters():
            param.requires_grad = True


    cur_prior_prob = prior_prob
    for episode in range(1, num_train_episodes + 1):
        # 收集数据，使用先验概率和先验函数
        data = collect_data(env, num_collect_episodes, cur_prior_prob, prior_func)
        cur_prior_prob = max(0, cur_prior_prob*0.997)

        # 离线训练
        average_loss, average_reward=train_offline(env, model, optimizer, device, data, discount_factor=discount_factor, batch_size=batch_size)

        scheduler.step()

        # 每隔一定的episodes进行评估
        if episode % eval_interval == 0:
            eval_reward,goal_rate = evaluate_model(env, model, device, num_episodes=100, if_render=if_render)

            print(f"Evaluation after episode {episode}: Average Reward = {eval_reward}, Goal Rate = {goal_rate}")

if __name__ == "__main__":
    env = MazeEnv(size=8, num_restrictions=8, num_rewards=8, seed=1, max_steps=64)
    train_and_evaluate_rl_model(env, num_collect_episodes=100, num_train_episodes=1000, eval_interval=10, prior_prob=0.5, prior_func=prior_function, batch_size=64,learning_rate=0.00001, if_render=False, if_pretrain=True)
