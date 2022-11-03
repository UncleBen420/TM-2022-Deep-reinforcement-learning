import random
import time

import numpy as np
from tqdm import tqdm


class DummyAgent:

    def __init__(self, environment, val_episode=10):

        self.environment = environment
        self.val_episode = val_episode
        self.action_space = environment.nb_action

    def exploit(self):
        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        time_by_episode = []

        with tqdm(range(self.val_episode), unit="episode") as episode:
            for i in episode:
                sum_episode_reward = 0
                self.environment.reload_env()
                existing_proba = None
                start_time = time.time()
                while True:
                    # casting to torch tensor

                    if existing_proba is None:
                        probs = np.random.rand(self.action_space)
                    else:
                        probs = existing_proba

                    probs /= probs.sum()
                    # no need to explore, so we select the most probable action
                    A = self.environment.exploit(probs)
                    S_prime, R, is_terminal, _, _ = self.environment.take_action(A)

                    sum_episode_reward += R
                    if is_terminal:
                        break

                done_time = time.time()
                rewards.append(sum_episode_reward)
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                nb_action.append(st)
                time_by_episode.append(done_time - start_time)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=rewards[-1], nb_action=st)

        return rewards, nb_action, good_choices, bad_choices, [], time_by_episode


