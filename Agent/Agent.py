import torch
import torch.nn as nn
from Agent.RL_network import CNN_FNN,CNN_dueling
from Memory.Memory import Memory
from Memory.PreMemory import preMemory
import numpy as np

class Agent():
    def __init__(self,n,O_max_len,dueling=False,double=False,PER=False):
        self.double=double
        self.PER=PER
        self.GAMMA=1
        self.n=n
        self.O_max_len=O_max_len


        super(Agent,self).__init__()
        if dueling:
            self.eval_net,self.target_net=CNN_dueling(self.n,self.O_max_len),CNN_dueling(self.n,self.O_max_len)
        else:
            self.eval_net,self.target_net=CNN_FNN(self.n,self.O_max_len),CNN_FNN(self.n,self.O_max_len)
        self.Q_NETWORK_ITERATION=100
        self.BATCH_SIZE=256
        self.learn_step_counter=0
        self.memory_counter=0
        if PER:
            self.memory=preMemory()
        else:
            self.memory=Memory()
        # self.EPISILO=0.8
        self.EPISILO = 0.9  # 提高初始值以鼓勵初期探索
        self.EPISILO_DECAY = 0.995  # 設定一個平穩的遞減因子
        self.EPISILO_MIN = 0.05  # 設定一個較高的最小值

        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.00001)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)  # 0.0001
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state=np.reshape(state,(-1,3,self.n,self.O_max_len))
        state=torch.FloatTensor(state)
        if np.random.randn() <= self.EPISILO:
            action_value=self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            # print("choose yes")
        else:
            action=np.random.randint(0,17)
            # print("choose no")

        # self.EPISILO = min(0.001, self.EPISILO - 0.00001)
        # self.EPISILO = max(self.EPISILO_MIN, self.EPISILO * self.EPISILO_DECAY)
        self.EPISILO = min(0.01, self.EPISILO - 0.001)
        return action

    # def reduce_epsilon(self):
    #     self.EPISILO = max(0.01, self.EPISILO - 0.0001)  # 或者其他衰减策略

    def PER_error(self,state,action,reward,next_state):

        state=torch.FloatTensor(np.reshape(state,(-1,3,self.n,self.O_max_len)))
        next_state=torch.FloatTensor(np.reshape(next_state, (-1, 3, self.n, self.O_max_len)))
        p=self.eval_net.forward(state)
        p_ = self.eval_net.forward(next_state)
        p_target=self.target_net(state)

        if self.double:
            q_a=p_.argmax(dim=1)
            q_a = torch.reshape(q_a, (-1, len(q_a)))
            qt = reward + self.GAMMA * p_target.gather(1, q_a)
        else:
            qt = reward + self.GAMMA * p_target.max(1)[0].view(self.BATCH_SIZE, 1)
        qt = qt.detach().numpy()
        p = p.detach().numpy()
        errors = np.abs(p[0][action] - qt[0][0])
        return errors

    def store_transition(self, state, action, reward, next_state):
        if self.PER:
            errors = self.PER_error(state, action, reward, next_state)
            self.memory.remember((state, action, reward, next_state), errors)
            self.memory_counter += 1
        else:
            self.memory.remember((state, action, reward, next_state))
            self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch = self.memory.sample(self.BATCH_SIZE)

        batch_state = np.array([o[0] for o in batch])
        batch_next_state = np.array([o[3] for o in batch])
        batch_action = np.array([o[1] for o in batch])
        batch_reward = np.array([o[1] for o in batch])

        batch_action = torch.LongTensor(np.reshape(batch_action, (-1, len(batch_action))))
        batch_reward = torch.LongTensor(np.reshape(batch_reward, (-1, len(batch_reward))))

        batch_state = torch.FloatTensor(np.reshape(batch_state, (-1, 3, self.n, self.O_max_len)))
        batch_next_state = torch.FloatTensor(np.reshape(batch_next_state, (-1, 3, self.n, self.O_max_len)))

        if self.double:
            q_eval = self.eval_net(batch_state).gather(1, batch_action)

            q_next_eval = self.eval_net(batch_next_state).detach()

            q_next = self.target_net(batch_next_state).detach()
            q_a = q_next_eval.argmax(dim=1)
            q_a = torch.reshape(q_a, (-1, len(q_a)))

            q_target = batch_reward + self.GAMMA * q_next.gather(1, q_a)
        else:
            q_eval = self.eval_net(batch_state).gather(1, batch_action)

            q_next = self.target_net(batch_next_state).detach()

            q_target = batch_reward + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)

        # self.optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()

        # loss.backward()
        # self.optimizer.step()
