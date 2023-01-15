import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

class PolicyNet(nn.Module):
    def __init__(self, nb_actions=8, classes=5, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(nb_actions)
        self.nb_actions = nb_actions
        self.classes = classes

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, stride=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.conf_head = torch.nn.Sequential(
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

        self.class_head = torch.nn.Sequential(
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, classes)
        )

        self.backbone.to(self.device)
        self.policy_head.to(self.device)
        self.class_head.to(self.device)
        self.conf_head.to(self.device)

        self.policy_head.apply(self.init_weights)
        self.class_head.apply(self.init_weights)
        self.conf_head.apply(self.init_weights)

    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs)

    def get_class(self, class_preds):
        proba = torch.nn.functional.softmax(class_preds, dim=1).squeeze()
        pred = torch.argmax(proba).item()
        return pred

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        preds = self.policy_head(x)
        class_preds = self.class_head(x)
        conf = self.conf_head(x)
        return preds, conf, class_preds

class TOD:

    def __init__(self, environment, learning_rate=0.0005, gamma=0.1,
                 lr_gamma=0.9, pa_dataset_size=1000, pa_batch_size=10, ):

        self.IOU_pa_batch = None
        self.gamma = gamma
        self.environment = environment
        self.environment.tod = self

        self.policy = PolicyNet()
        self.nb_per_class = np.zeros(self.policy.classes)

        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)

        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.G_pa_batch = None

    def save(self, file):
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def update_policy(self, batch):

        S, A, G, IOU, LABEL = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, conf, class_preds = self.policy(S)

        log_probs = torch.log(action_probs)
        log_probs = torch.nan_to_num(log_probs)

        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))

        policy_loss = - (G.unsqueeze(1) * selected_log_probs).mean()
        loss = policy_loss
        loss.backward(retain_graph=True)

        conf_loss = torch.nn.functional.mse_loss(conf.squeeze(), IOU.squeeze())
        conf_loss.backward(retain_graph=True)

        class_loss = torch.nn.functional.cross_entropy(class_preds.squeeze(), LABEL.squeeze())
        class_loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), conf_loss.item(), class_loss.item()

    def fit_one_episode(self, S):

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []
        IOU_batch = []
        LABEL_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_reward = 0

        counter += 1
        # State preprocess

        while True:
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)
            with torch.no_grad():
                action_probs, conf, class_preds = self.policy(S)
                #class_preds = self.class_net(S)

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)

                label = self.policy.get_class(class_preds)

            S_prime, R, is_terminal, iou, label = self.environment.take_action_tod(A, conf.item(), label)

            S_batch.append(S)
            A_batch.append(A)
            R_batch.append(R)
            IOU_batch.append(iou)
            LABEL_batch.append(label)
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        for i in reversed(range(1, len(R_batch))):
            R_batch[i - 1] += self.gamma * R_batch[i]

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(R_batch).to(self.policy.device)
        IOU_batch = torch.FloatTensor(IOU_batch).to(self.policy.device)
        LABEL_batch = torch.LongTensor(LABEL_batch).to(self.policy.device)


        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------
        # Append the past action batch to the current batch if possible

        if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
            batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
                     torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
                     torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0),
                     torch.cat((self.IOU_pa_batch[0:self.pa_batch_size], IOU_batch), 0),
                     torch.cat((self.LABEL_pa_batch[0:self.pa_batch_size], LABEL_batch), 0),)
        else:
            batch = (S_batch, A_batch, G_batch, IOU_batch, LABEL_batch)

        # Add some experiences to the buffer with respect of TD error
        nb_new_memories = min(5, counter)

        idx = torch.randperm(len(A_batch))[:nb_new_memories]

        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
            self.IOU_pa_batch = IOU_batch[idx]
            self.LABEL_pa_batch = LABEL_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)
            self.IOU_pa_batch = torch.cat((self.IOU_pa_batch, IOU_batch[idx]), 0)
            self.LABEL_pa_batch = torch.cat((self.LABEL_pa_batch, LABEL_batch[idx]), 0)

        # clip the buffer if it's to big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch

            shuffle_index = torch.randperm(len(self.A_pa_batch))
            self.A_pa_batch = self.A_pa_batch[shuffle_index]
            self.G_pa_batch = self.G_pa_batch[shuffle_index]
            self.S_pa_batch = self.S_pa_batch[shuffle_index]
            self.IOU_pa_batch = self.IOU_pa_batch[shuffle_index]
            self.LABEL_pa_batch = self.LABEL_pa_batch[shuffle_index]

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])
            _, self.IOU_pa_batch = torch.split(self.IOU_pa_batch, [surplus, self.pa_dataset_size])
            _, self.LABEL_pa_batch = torch.split(self.LABEL_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        loss_tod, iou_loss, class_loss = self.update_policy(batch)
        return iou, sum_reward, loss_tod, class_loss, iou_loss

    def exploit_one_episode(self, S):
        sum_reward = 0
        while True:
            # State preprocess

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs, conf, class_preds = self.policy(S)
                #class_preds = self.class_net(S)
                conf = conf.item()

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)

                label = self.policy.get_class(class_preds)

            S_prime, R, is_terminal, iou, label = self.environment.take_action_tod(A, conf, label)

            S = S_prime
            sum_reward += R
            if is_terminal:
                break

        return iou, sum_reward
