import random

from tqdm import tqdm


class DummyAgent:

    def __init__(self, environment, val_episode=10):

        self.environment = environment
        self.val_episode = val_episode
        self.action_space = environment.nb_action

    def exploit(self):
        rewards = []
        nb_action = []
        nb_mark = []
        successful_marks = []
        good_choices = []
        bad_choices = []

        with tqdm(range(self.val_episode), unit="episode") as episode:
            for i in episode:
                sum_episode_reward = 0
                self.environment.reload_env()
                while True:
                    # casting to torch tensor

                    A, A_task = random.randint(0, self.action_space - 1), random.randint(0, 1)
                    S_prime, R, is_terminal = self.environment.take_action(A)

                    sum_episode_reward += R
                    if is_terminal:
                        break

                rewards.append(sum_episode_reward)

                nbm = self.environment.nb_mark
                nbmc = self.environment.marked_correctly
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                nb_action.append(st)
                nb_mark.append(nbm)
                good_choices.append(gt / (st + 0.00001))
                bad_choices.append(bt / (st + 0.00001))
                successful_marks.append(self.environment.marked_correctly)

                episode.set_postfix(rewards=rewards[-1], nb_action=st, marked_correctly=nbmc, nb_mark=nbm)

        return rewards, nb_mark, nb_action, successful_marks, good_choices, bad_choices


