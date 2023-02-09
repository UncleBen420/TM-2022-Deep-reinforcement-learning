import time
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class PolicyNet(nn.Module):
    """
    Class implementing a Q-Net
    """
    def __init__(self, img_res=100):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("RUNNING ON {0}".format(self.device))

        self.action_space = np.arange(4)

        self.img_res = img_res
        self.sub_img_res = int(self.img_res / 2)

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1, 7, 7), stride=(1, 3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(16),
            torch.nn.Conv3d(16, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(32),
            torch.nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4)
        )

        self.backbone.to(self.device)
        self.head.to(self.device)

        self.head.apply(self.init_weights)

    def init_weights(self, m):
        """
        Init the model weight
        :param m:
        :return:
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        """
        prepare the state given by the environment in a format that the Policy net can accept.
        :param state: state given by the environment.
        :return: a tensor of shape (batch, sequence, channel, width, high)
        """
        img = state.permute(0, 3, 1, 2)
        patches = img.unfold(1, 3, 3).unfold(2, self.sub_img_res, self.sub_img_res).unfold(3, self.sub_img_res,
                                                                                           self.sub_img_res)
        patches = patches.contiguous().view(1, 4, -1, self.sub_img_res, self.sub_img_res)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def forward(self, state):
        """
        forward method
        :param state: the prepared state
        :return: the Q predicted
        """
        x = self.backbone(state)
        return self.head(x)


class PolicyGradient:
    """
    A modified version of a policy gradient descent.
    """

    def __init__(self, environment, learning_rate=0.001,
                 episodes=100, val_episode=100, gamma=0.6,
                 lr_gamma=0.5, pa_dataset_size=1000, pa_batch_size=100):

        self.S_pa_batch = None
        self.G_pa_batch = None
        self.A_pa_batch = None
        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.val_episode = val_episode
        self.policy = PolicyNet(img_res=environment.img_res)
        self.action_space = environment.nb_action
        self.pa_dataset_size = pa_dataset_size
        self.pa_batch_size = pa_batch_size
        print(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)

    def update_policy(self):
        """
        Update the policy Net with steps of the replay memory.
        :return:
        """
        if len(self.A_pa_batch) < self.pa_batch_size:
            return 0.

        shuffle_index = torch.randperm(len(self.A_pa_batch))
        self.A_pa_batch = self.A_pa_batch[shuffle_index]
        self.G_pa_batch = self.G_pa_batch[shuffle_index]
        self.S_pa_batch = self.S_pa_batch[shuffle_index]

        S = self.S_pa_batch[:self.pa_batch_size]
        A = self.A_pa_batch[:self.pa_batch_size]
        G = self.G_pa_batch[:self.pa_batch_size]

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs = self.policy(S)

        selected = torch.gather(action_probs, 1, A.unsqueeze(1))
        loss = torch.nn.functional.mse_loss(selected.squeeze(), G)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def calculate_advantage_tree(self, rewards):
        """
        calculate the discounted reward but in a tree like propagation.
        :param rewards: the reward received during the episode.
        :return: return the G.
        """
        rewards = np.array(rewards)

        # calculate the discount rewards
        G = rewards[:, 3].astype(float)
        node_info = rewards[:, 0:3].astype(int)

        for ni in node_info[::-1]:
            parent, current, child = ni
            parent_index = np.all(node_info[:, [1, 2]] == [parent, current], axis=1)
            current_index = np.all(node_info[:, [1, 2]] == [current, child], axis=1)

            if parent != -1:
                G[parent_index] += self.gamma * G[current_index]

        return G.tolist()

    def fit(self):
        """
        train the agent for a certain number of episode.
        :return: some metrics.
        """
        # for plotting
        losses = []
        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        nb_effective_action = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:
                # ------------------------------------------------------------------------------------------------------
                # EPISODE PREPARATION
                # ------------------------------------------------------------------------------------------------------

                S_batch = []
                R_batch = []
                A_batch = []

                S = self.environment.reload_env()

                # ------------------------------------------------------------------------------------------------------
                # EPISODE REALISATION
                # ------------------------------------------------------------------------------------------------------
                counter = 0
                sum_reward = 0
                existing_proba = None
                while True:

                    counter += 1
                    # State preprocess
                    S = torch.from_numpy(S).float()
                    S = S.unsqueeze(0).to(self.policy.device)
                    S = self.policy.prepare_data(S)

                    if existing_proba is None:
                        with torch.no_grad():
                            action_probs = self.policy(S)
                            action_probs = action_probs.detach().cpu().numpy()[0]
                    else:
                        action_probs = existing_proba

                    A = self.environment.follow_policy(action_probs)

                    S_prime, R, is_terminal, node_info, existing_pred = self.environment.take_action(A)
                    existing_proba = existing_pred
                    parent, current, child = node_info

                    S_batch.append(S)
                    A_batch.append(A)
                    R_batch.append((parent, current, child, R))
                    sum_reward += R

                    S = S_prime

                    if is_terminal:
                        break

                # ------------------------------------------------------------------------------------------------------
                # CUMULATED REWARD CALCULATION AND TD ERROR
                # ------------------------------------------------------------------------------------------------------
                G_batch = self.calculate_advantage_tree(R_batch)

                # ------------------------------------------------------------------------------------------------------
                # BATCH PREPARATION
                # ------------------------------------------------------------------------------------------------------
                S_batch = torch.concat(S_batch).to(self.policy.device)
                A_batch = torch.LongTensor(A_batch).to(self.policy.device)
                G_batch = torch.FloatTensor(G_batch).to(self.policy.device)

                # ------------------------------------------------------------------------------------------------------
                # PAST ACTION DATASET PREPARATION
                # ------------------------------------------------------------------------------------------------------

                # Add some experiences to the buffer with respect of TD error
                nb_new_memories = min(100, counter)

                weights = G_batch
                self.min_r = torch.min(weights)
                self.max_r = torch.max(weights)
                weights = (weights - self.min_r) / (self.max_r - self.min_r)
                weights /= weights.sum()
                idx = torch.multinomial(weights, nb_new_memories)

                if self.A_pa_batch is None:
                    self.A_pa_batch = A_batch[idx]
                    self.S_pa_batch = S_batch[idx]
                    self.G_pa_batch = G_batch[idx]
                else:
                    self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
                    self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
                    self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)

                # clip the buffer if it's to big
                if len(self.A_pa_batch) > self.pa_dataset_size:
                    # shuffling the batch

                    # dataset clipping
                    surplus = len(self.A_pa_batch) - self.pa_dataset_size
                    _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
                    _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
                    _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])

                # ------------------------------------------------------------------------------------------------------
                # MODEL OPTIMISATION
                # ------------------------------------------------------------------------------------------------------
                loss = self.update_policy()

                # ------------------------------------------------------------------------------------------------------
                # METRICS RECORD
                # ------------------------------------------------------------------------------------------------------
                rewards.append(sum_reward)
                losses.append(loss)
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                mz = self.environment.nb_max_zoom
                nb_action.append(st)
                nb_effective_action.append(mz)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=sum_reward, loss=loss, nb_action=st)

        return losses, rewards, nb_action, good_choices, bad_choices, nb_effective_action

    def exploit(self):
        """
        exploit (the agent doesn't learn anymore) on the environment.
        :return: some metrics.
        """
        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        conv_policy = []
        time_by_episode = []
        nb_effective_action = []

        with tqdm(range(self.val_episode), unit="episode") as episode:
            for i in episode:
                sum_episode_reward = 0
                S = self.environment.reload_env()
                existing_proba = None
                start_time = time.time()
                while True:

                    # State preprocess
                    S = torch.from_numpy(S).float()
                    S = S.unsqueeze(0).to(self.policy.device)
                    S = self.policy.prepare_data(S)

                    if existing_proba is None:
                        with torch.no_grad():
                            action_probs = self.policy(S)
                            action_probs = action_probs.detach().cpu().numpy()[0]
                    else:
                        action_probs = existing_proba

                    A = self.environment.exploit(action_probs)

                    S_prime, R, is_terminal, _, existing_pred = self.environment.take_action(A)
                    existing_proba = existing_pred

                    S = S_prime
                    sum_episode_reward += R
                    if is_terminal:
                        break

                done_time = time.time()
                rewards.append(sum_episode_reward)
                conv_policy.append(self.environment.conventional_policy_nb_step)
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                mz = self.environment.nb_max_zoom
                nb_action.append(st)
                nb_effective_action.append(mz)
                time_by_episode.append(done_time - start_time)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=rewards[-1], nb_action=st)

        return rewards, nb_action, good_choices, bad_choices, conv_policy, time_by_episode, nb_effective_action
