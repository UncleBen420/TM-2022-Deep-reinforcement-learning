import numpy as np

from Environment.GridWorld import Action


class DP:
    def __init__(self, environment, threshold=0.001, gamma=0.1):
        self.policy = np.zeros((environment.size * environment.size))
        self.V = np.zeros((environment.size * environment.size))
        self.policy = environment.init_policy()

        self.threshold = threshold
        self.environment = environment
        self.gamma = gamma

    def evaluate_policy(self):

        while True:
            delta = 0
            for s in range(self.environment.states.size):
                # Analysing terminal state is not necessary
                if not self.environment.states[s].value['is_terminal']:
                    v = self.V[s]
                    a = self.policy[s]

                    ns = self.environment.get_next_state(s, Action(a))
                    r = self.environment.get_reward(s, Action(a), ns)
                    p = self.environment.get_probability(s, Action(a), ns, r)
                    self.V[s] = p * (r + self.gamma * self.V[ns])
                    delta = max(delta, abs(v - self.V[s]))

            if delta < self.threshold:
                break
        return self.V

    def update_policy(self):
        policy_stable = True

        for s in range(self.environment.states.size):
            # Analysing terminal state is not necessary
            if not self.environment.states[s].value['is_terminal']:

                old_action = self.policy[s]
                possible_actions = np.zeros(self.environment.nb_action)

                for a in range(self.environment.nb_action):
                    ns = self.environment.get_next_state(s, Action(a))
                    r = self.environment.get_reward(s, Action(a), ns)

                    p = self.environment.get_probability(s, Action(a), ns, r)

                    possible_actions[a] = p * (r + self.gamma * self.V[ns])

                self.policy[s] = np.argmax(possible_actions)
                if old_action != self.policy[s]:
                    policy_stable = False

        return policy_stable

    def policy_iteration(self):

        history = []
        v = []
        while True:
            history.append(self.policy.copy())
            v.append(self.evaluate_policy().copy())
            if self.update_policy():
                break

        return history, v
