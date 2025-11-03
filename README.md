# 强化学习介绍
## 核心要素
### 智能体(Agent)
执行动作，做出决策的主体;
### 环境(Environment)
智能体所处的外部场景;会对智能体的动作做出反馈;
### 状态(State)
环境在某个时刻的具体情况;
### 动作(Action)
智能体在某个状态下可以采取的行为;
### 奖励(Reward)
环境对智能体动作的即时反馈;奖励是即时、短期的信号，代表当前动作的直接结果;
### 策略(Policy)
智能体的决策逻辑，即什么状态下该做什么动作;
### 价值(Value)
对状态或动作的长期回报的预估;价值是长远的评估，代表从当前状态出发，未来可能获得的所有奖励的总和;

## 定义 & 核心问题
### 定义
- 强化学习是**智能体**通过与**环境**试错**交互**，从**反馈**中学习最优策略，以**最大化价值**为目的的机器学习方法;

### 核心问题
- 信用分配问题：​将最终的，全局的奖励信号，合理地"分配"或"追溯"到导致这个结果的一系列动作上;
- 探索与利用权衡：平衡尝试新动作和依赖已知有效动作;

## 主要分类
### 按照算法目标分类
- 基于价值，学习 状态-动作对的价值（例如Q函数），通过价值指导动作选择（如 Q-Learning、DQN）;
- 基于策略，直接学习策略函数（状态到动作的映射），优化策略以最大化奖励（如 REINFORCE、PPO，TRPO）;
- Actor-Critic，结合两者，Actor负责执行策略，Critic负责评价策略价值，（如，A2C, DDPG, SAC）;

### 按照环境模型
- 无模型，Q-Learning、SARSA、PPO;
- 有模型，MBPO，PETS，World Models;

# Q-Learing
## 代码说明
- 智能体，QLearningAgent，用于学习最优动作策略；
- 智能体核心功能，初始化，动作选择，更新Q表，训练，测试；
- 环境：网格GridWorld ；5x5 网格环境，有起点，终点，障碍物等信息；有动作和初始化等操作；

## 训练
      def train(self, episodes=1000):
        """训练智能体"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state) # epsilon-greedy策略，平衡探索和当前Q值最高的选择；
                next_state, reward, done = self.env.step(action) # 执行动作，返回下一状态，奖励和是否结束；
                self.learn(state, action, reward, next_state) # 核心是贝尔曼更新公式；
                state = next_state
                total_reward += reward

      def learn(self, state, action, reward, next_state):
        current_q = self.q_table[row, col, action] # 当前Q值
        max_next_q = np.max(self.q_table[next_row, next_col]) # 下一状态的最大Q值
        # 贝尔曼更新公式：Q(s,a) = Q(s,a) + alpha*[r + gamma*maxQ(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[row, col, action] = new_q

      # Q(s,a) = Q(s,a) + alpha*[r + gamma*maxQ(s',a') - Q(s,a)]理解：
      #   r + gamma*maxQ(s',a') 是 TD 目标, 即，执行当前操作后，智能体期望获得积累奖励。包括即时奖励和未来奖励；
      #   r + gamma*maxQ(s',a') - Q(s,a) 是TD误差，即期望积累奖励和当前奖励的误差；
      #   Q(s,a) + alpha*[r + gamma*maxQ(s',a') - Q(s,a)]，当前数值，加上更新幅度 * TD误差；
      #   alpha类似于学习率的概念，代表参数更新的幅度；
      #   gamma折扣因子，要对未来的重视程度，gamma接近1时，更关注长远利益，反之更关注眼前利益；


## QA
- 为什么叫Q-Learing？这里的Q是Quality，价值的意思；
- q_table[next_row, next_col]不是确定的吗？为啥要加max？因为q_table[next_row, next_col]包含不同的动作，要取4个动作里收益最高的；
- 如何理解TD？Temporal Difference，即时间序列中连续的两个时刻（当前步 t 和下一步 t+1）的价值估计之差；

# DPO
## 介绍
- DPO（direct preference optimization，直接偏好优化）是一种模型训练方法，是对已经训练好的模型的一种参数微调方式；
- 定义：直接用人类偏好（chosen vs rejected）训练模型，通过最大化策略模型和参考模型对人类偏好的**相对倾向性**，使得模型更加倾向于人类的偏好；
- policy_model和ref_model在强化训练开始前一般采用同一模型；ref_model训练中冻结参数，仅更新policy_model，最终也使用policy_model进行推理；
- 偏好数据集，包含prompt，chosen，rejected；
## 损失函数
      def dpo_loss(policy_model, ref_model, batch, beta=0.1):
          # 策略模型（当前训练的模型）计算log概率
          chosen_logits = policy_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
          rejected_logits = policy_model(...).logits

          with torch.no_grad():    # 参考模型（通常是预训练模型快照）计算log概率（冻结参数）
              ref_chosen_logits = ref_model(...).logits
              ref_rejected_logits = ref_model(...).logits

          def get_sequence_log_prob(logits, input_ids, attention_mask): # 计算生成该序列的 “联合概率” 的对数；
                ...  

          # 计算策略模型和参考模型的log概率差
          policy_chosen_logprob = get_sequence_log_prob(chosen_logits, batch["chosen_input_ids"], batch["chosen_attention_mask"])
          policy_rejected_logprob = get_sequence_log_prob(rejected_logits, batch["rejected_input_ids"], batch["rejected_attention_mask"])
          ref_chosen_logprob = get_sequence_log_prob(ref_chosen_logits, batch["chosen_input_ids"], batch["chosen_attention_mask"])
          ref_rejected_logprob = get_sequence_log_prob(ref_rejected_logits, batch["rejected_input_ids"], batch["rejected_attention_mask"])

          logits = (policy_chosen_logprob - policy_rejected_logprob) - (ref_chosen_logprob - ref_rejected_logprob)
          loss = -F.logsigmoid(beta * logits).mean()  # 负对数似然，确保chosen概率更高
          return loss

      # 损失函数解释：通过对比 “策略模型” 和 “参考模型” 对人类偏好回答（chosen）与非偏好回答（rejected）的 “相对倾向性”，迫使策略模型更明显地偏好 chosen；
      # 为什么要有参考模型？
            a 避免无意义膨胀，防止过于偏向chosen；b 防止遗忘初始能力；



