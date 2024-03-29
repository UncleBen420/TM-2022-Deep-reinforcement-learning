import time
import numpy as np
from tqdm import tqdm


class DummyAgent:
    """
    Class representing an agent that choose is action randomly.
    """
    def __init__(self, environment, val_episode=10):

        self.environment = environment
        self.val_episode = val_episode
        self.action_space = environment.nb_action

    def exploit(self):
        """
        perform random actions on the environment.
        :return: some metrics.
        """
        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        time_by_episode = []
        nb_effective_action = []

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

                    # no need to explore, so we select the most probable action
                    A = self.environment.exploit(probs)
                    S_prime, R, is_terminal, _, existing_pred = self.environment.take_action(A)

                    existing_proba = existing_pred

                    sum_episode_reward += R
                    if is_terminal:
                        break

                done_time = time.time()
                rewards.append(sum_episode_reward)
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

        return rewards, nb_action, good_choices, bad_choices, [], time_by_episode, nb_effective_action


