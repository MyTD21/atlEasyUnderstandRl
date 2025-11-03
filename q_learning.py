import numpy as np
import random

# 1. 定义网格世界环境
class GridWorld:
    def __init__(self):
        # 网格大小：5x5 (行, 列)
        self.rows = 5
        self.cols = 5
        # 起点(0,0)，终点(4,4)，障碍物(2,2)、(3,1)
        self.start = (0, 0)
        self.end = (4, 4)
        self.obstacles = {(2, 2), (3, 1)}
        # 当前位置
        self.current_pos = self.start

    def reset(self):
        """重置环境到起点"""
        self.current_pos = self.start
        return self.current_pos

    def step(self, action):
        """执行动作，返回新状态、奖励、是否结束"""
        row, col = self.current_pos
        # 动作：0(上), 1(右), 2(下), 3(左)
        if action == 0:
            new_row = max(row - 1, 0)
            new_col = col
        elif action == 1:
            new_row = row
            new_col = min(col + 1, self.cols - 1)
        elif action == 2:
            new_row = min(row + 1, self.rows - 1)
            new_col = col
        else:  # action == 3
            new_row = row
            new_col = max(col - 1, 0)

        new_pos = (new_row, new_col)

        # 奖励机制：到达终点+100，撞障碍物-50，其他0
        if new_pos == self.end:
            reward = 100
            done = True
        elif new_pos in self.obstacles:
            reward = -50
            done = False  # 撞到障碍物不结束，但扣分
        else:
            reward = 0
            done = False

        self.current_pos = new_pos
        return new_pos, reward, done

    def render(self):
        """可视化当前网格状态"""
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                if (i, j) == self.current_pos:
                    row.append("A")  # 智能体
                elif (i, j) == self.end:
                    row.append("T")  # 终点
                elif (i, j) in self.obstacles:
                    row.append("X")  # 障碍物
                else:
                    row.append(".")  # 空地
            print(" ".join(row))
        print("-" * 15)


# 2. 定义Q-Learning智能体
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子（未来奖励的权重）
        self.epsilon = epsilon  # 探索率（epsilon-greedy策略）
        # 初始化Q表：状态->动作的价值 (rows x cols x 4动作)
        self.q_table = np.zeros((env.rows, env.cols, 4))

    def choose_action(self, state):
        """epsilon-greedy策略选择动作：平衡探索和利用"""
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.choice(range(4))
        else:
            # 利用：选择当前Q值最高的动作
            row, col = state
            return np.argmax(self.q_table[row, col])

    def learn(self, state, action, reward, next_state):
        """更新Q表（贝尔曼方程）"""
        row, col = state
        next_row, next_col = next_state
        # 当前Q值
        current_q = self.q_table[row, col, action]
        # 下一状态的最大Q值
        max_next_q = np.max(self.q_table[next_row, next_col])
        # 贝尔曼更新公式：Q(s,a) = Q(s,a) + alpha*[r + gamma*maxQ(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[row, col, action] = new_q

    def train(self, episodes=1000):
        """训练智能体"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                #self.env.render()
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            #import pdb; pdb.set_trace()
            # 每100轮打印一次训练进度
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

    def test(self):
        """测试训练好的智能体（完全利用，无探索）"""
        state = self.env.reset()
        done = False
        print(self.q_table)
        print("测试最优路径：")
        self.env.render()
        while not done:
            action = np.argmax(self.q_table[state[0], state[1]])  # 只选最优动作
            next_state, _, done = self.env.step(action)
            state = next_state
            self.env.render()


# 3. 运行主程序
if __name__ == "__main__":
    # 创建环境和智能体
    env = GridWorld()
    agent = QLearningAgent(env)
    # 训练
    agent.train(episodes=1000)
    # 测试最优路径
    agent.test()
