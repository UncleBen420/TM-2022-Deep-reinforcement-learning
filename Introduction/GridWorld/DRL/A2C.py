import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from collections import OrderedDict

from tqdm import tqdm

from Environment.GridWorld import Action


class actorCriticNet(nn.Module):
    def __init__(self, env, n_hidden_layers=4, n_hidden_nodes=32,
                 learning_rate=0.01, bias=False, device='cpu'):
        super(actorCriticNet, self).__init__()

        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.bias = bias
        self.action_space = np.arange(self.n_outputs)

        # Generate network according to hidden layer and node settings
        self.layers = OrderedDict()
        self.n_layers = 2 * self.n_hidden_layers
        for i in range(self.n_layers + 1):
            # Define single linear layer
            if self.n_hidden_layers == 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs,
                    self.n_outputs,
                    bias=self.bias)
            # Define input layer for multi-layer network
            elif i % 2 == 0 and i == 0 and self.n_hidden_layers != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs,
                    self.n_hidden_nodes,
                    bias=self.bias)
            # Define intermediate hidden layers
            elif i % 2 == 0 and i != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_hidden_nodes,
                    bias=self.bias)
            else:
                self.layers[str(i)] = nn.ReLU()

        self.body = nn.Sequential(self.layers)

        # Define policy head
        self.Q = nn.Sequential(
            nn.Linear(self.n_hidden_nodes,
                      self.n_hidden_nodes,
                      bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes,
                      self.n_outputs,
                      bias=self.bias))
        # Define value head
        self.V = nn.Sequential(
            nn.Linear(self.n_hidden_nodes,
                      self.n_hidden_nodes,
                      bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes,
                      1,
                      bias=self.bias))

        if self.device == 'cuda':
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def predict(self, state):
        body_output = self.get_body_output(state)
        probs = F.softmax(self.Q(body_output), dim=-1)
        return probs, self.V(body_output)

    def get_body_output(self, state):
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.body(state_t)

    def follow_policy(self, probs):
        action = np.random.choice(self.action_space, p=probs)
        return action

    def get_log_probs(self, state):
        body_output = self.get_body_output(state)
        logprobs = F.log_softmax(self.policy(body_output), dim=-1)
        return logprobs

    def calc_loss(self, states, actions, rewards, advantages, beta=0.001):
        actions_t = torch.LongTensor(actions).to(self.network.device)
        rewards_t = torch.FloatTensor(rewards).to(self.network.device)
        advantages_t = torch.FloatTensor(advantages).to(self.network.device)

        log_probs = self.network.get_log_probs(states)
        log_prob_actions = advantages_t * log_probs[range(len(actions)), actions]
        policy_loss = -log_prob_actions.mean()

        action_probs, values = self.network.predict(states)
        entropy_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()

        value_loss = self.zeta * nn.MSELoss()(values.squeeze(-1), rewards_t)

        # Append values
        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        self.entropy_loss.append(entropy_loss)

        return policy_loss, entropy_loss, value_loss


class A2C():
    def __init__(self, env, network):

        self.env = env
        self.network = actorCriticNet()
        self.action_space = np.arange(env.action_space.n)


    def fit(self):
        losses = []
        rewards = []
        dataset = []


        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:

                episode_loss = []
                S_batch, A_batch, R_batch, D_batch, S2_batch, V_batch = [], [], [], [], [], []

                S = 0  # initial state

                reward = 0
                V_sum = 0
                nb_step = 0

                for _ in range(1000):
                    Sv = self.environment.get_env_vision(S)
                    # casting to torch tensor
                    Sv = torch.from_numpy(Sv).float()
                    probs, V = self.network.predict(Sv)
                    A = self.network.follow_policy(probs)

                    S_prime = self.environment.get_next_state(S, Action(A))
                    R = self.environment.get_reward(S, Action(A), S_prime)
                    Sv_prime = self.environment.get_env_vision(S_prime)

                    done = self.environment.states[S].value['is_terminal']

                    S_batch.append(Sv)
                    S2_batch.append(Sv_prime)
                    A_batch.append(A)
                    R_batch.append(R)
                    D_batch.append(done)
                    V_batch.append(V.detach().numpy())

                    S = S_prime

                    if done:
                        break

                G_batch = []

                for t in range(len(R_batch)):
                    Gt = 0
                    pw = 0
                    for R in R_batch[t:]:
                        Gt = Gt + self.gamma ** pw * R
                        pw += 1
                    G_batch.append(Gt)

                TD_batch = G_batch - V_batch





    def generate_episode(self):

        counter = 0
        total_count = self.batch_size * self.n_steps
        while counter < total_count:
            done = False
            while done == False:
                action = self.network.get_action(self.s_0)
                s_1, r, done, _ = self.env.step(action)
                self.reward += r
                states.append(self.s_0)
                next_states.append(s_1)
                actions.append(action)
                rewards.append(r)
                dones.append(done)
                self.s_0 = s_1

                if done:
                    self.ep_rewards.append(self.reward)
                    self.s_0 = self.env.reset()
                    self.reward = 0
                    self.ep_counter += 1
                    if self.ep_counter >= self.num_episodes:
                        counter = total_count
                        break

                counter += 1
                if counter >= total_count:
                    break
        return states, actions, rewards, dones, next_states

    def calc_rewards(self, batch):
        states, actions, rewards, dones, next_states = batch
        rewards = np.array(rewards)
        total_steps = len(rewards)

        state_values = self.network.predict(states)[1]
        next_state_values = self.network.predict(next_states)[1]
        done_mask = torch.ByteTensor(dones).to(self.network.device)
        next_state_values[done_mask] = 0.0
        state_values = state_values.detach().numpy().flatten()
        next_state_values = next_state_values.detach().numpy().flatten()

        G = np.zeros_like(rewards, dtype=np.float32)
        td_delta = np.zeros_like(rewards, dtype=np.float32)
        dones = np.array(dones)

        for t in range(total_steps):
            last_step = min(self.n_steps, total_steps - t)

            # Look for end of episode
            check_episode_completion = dones[t:t + last_step]
            if check_episode_completion.size > 0:
                if True in check_episode_completion:
                    next_ep_completion = np.where(check_episode_completion == True)[0][0]
                    last_step = next_ep_completion

            # Sum and discount rewards
            G[t] = sum([rewards[t + n:t + n + 1] * self.gamma ** n for
                        n in range(last_step)])

        if total_steps > self.n_steps:
            G[:total_steps - self.n_steps] += next_state_values[self.n_steps:] \
                                              * self.gamma ** self.n_steps
        td_delta = G - state_values
        return G, td_delta

    def train(self, n_steps=5, batch_size=10, num_episodes=2000,
              gamma=0.99, beta=1 - 3, zeta=0.5):
        self.n_steps = n_steps
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.beta = beta
        self.zeta = zeta
        self.batch_size = batch_size

        # Set up lists to log data
        self.ep_rewards = []
        self.kl_div = []
        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []
        self.total_policy_loss = []
        self.total_loss = []

        self.s_0 = self.env.reset()
        self.reward = 0
        self.ep_counter = 0
        while self.ep_counter < num_episodes:
            batch = self.generate_episode()
            G, td_delta = self.calc_rewards(batch)
            states = batch[0]
            actions = batch[1]
            current_probs = self.network.predict(states)[0].detach().numpy()

            self.update(states, actions, G, td_delta)

            new_probs = self.network.predict(states)[0].detach().numpy()
            kl = -np.sum(current_probs * np.log(new_probs / current_probs))
            self.kl_div.append(kl)

            print("\rMean Rewards: {:.2f} Episode: {:d}    ".format(
                np.mean(self.ep_rewards[-100:]), self.ep_counter), end="")

    def calc_loss(self, states, actions, rewards, advantages, beta=0.001):
        actions_t = torch.LongTensor(actions).to(self.network.device)
        rewards_t = torch.FloatTensor(rewards).to(self.network.device)
        advantages_t = torch.FloatTensor(advantages).to(self.network.device)

        log_probs = self.network.get_log_probs(states)
        log_prob_actions = advantages_t * log_probs[range(len(actions)), actions]
        policy_loss = -log_prob_actions.mean()

        action_probs, values = self.network.predict(states)
        entropy_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()

        value_loss = self.zeta * nn.MSELoss()(values.squeeze(-1), rewards_t)

        # Append values
        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        self.entropy_loss.append(entropy_loss)

        return policy_loss, entropy_loss, value_loss

    def update(self, states, actions, rewards, advantages):
        self.network.optimizer.zero_grad()
        policy_loss, entropy_loss, value_loss = self.calc_loss(states,
                                                               actions, rewards, advantages)

        total_policy_loss = policy_loss - entropy_loss
        self.total_policy_loss.append(total_policy_loss)
        total_policy_loss.backward(retain_graph=True)

        value_loss.backward()

        total_loss = policy_loss + value_loss + entropy_loss
        self.total_loss.append(total_loss)
        self.network.optimizer.step()