import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim,hidden_state_dim=1024, policy_conv=True,device =None):
        super(ActorCritic, self).__init__()
        self.device = device
        # encoder with convolution layer for MobileNetV3, EfficientNet and RegNet
        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(int(state_dim * 32 / feature_dim), hidden_state_dim),
                nn.ReLU()
            )
        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )

        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, action_dim),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.action_dim = action_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, restart_batch=False, training=True):
        if restart_batch:
            del memory.hidden[:]
            memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).to(self.device))

        if not self.policy_conv:
            state = state_ini.flatten(1)
        else:
            state = state_ini

        # print(state.shape)
        state = self.state_encoder(state)

        state, hidden_output = self.gru(state.view(1, state.size(0), state.size(1)), memory.hidden[-1])
        memory.hidden.append(hidden_output)

        state = state[0]
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        if training:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_probs.max(1)[1]

        return action

    def evaluate(self, state, action):
        seq_l = state.size(0)
        batch_size = state.size(1)

        if not self.policy_conv:
            state = state.flatten(2)
            state = state.view(seq_l * batch_size, state.size(2))
        else:
            state = state.view(seq_l * batch_size, state.size(2), state.size(3), state.size(4))

        state = self.state_encoder(state)
        state = state.view(seq_l, batch_size, -1)

        state, hidden = self.gru(state, torch.zeros(1, batch_size, state.size(2)).to(self.device))
        state = state.view(seq_l * batch_size, -1)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).to(self.device)
        dist_entropy = dist.entropy().to(self.device)
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv, gpu=0,
                lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2,device=None):
        super(PPO, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.policy = ActorCritic(feature_dim, state_dim, action_dim,hidden_state_dim, policy_conv,device = device).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv,device=device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, restart_batch=False, training=True):
        return self.policy_old.act(state, memory, restart_batch, training)

    def update(self, memory):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).to(self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states, 0).to(self.device).detach()
        old_actions = torch.stack(memory.actions, 0).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs, 0).to(self.device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            print("loss:{:.6f}".format(torch.mean(loss).cpu().item()))
        self.policy_old.load_state_dict(self.policy.state_dict())
    def resame(self,ckptPath):
        ckps = torch.load(ckptPath, map_location='cuda:0')
        self.policy.load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})#,strict=False)
        self.policy_old.load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})#,strict=False)
        self.policy.train()
        self.policy_old.train()
    def evalModel(self,ckptPath):
        ckps = torch.load(ckptPath, map_location='cuda:0')
        self.policy.load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})#,strict=False)
        self.policy_old.load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})#,strict=False)
        self.policy.eval()
        self.policy_old.eval()
    def train(self):
        self.policy.train()
        self.policy_old.train()