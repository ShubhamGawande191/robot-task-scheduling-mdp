import numpy as np
from collections import defaultdict

class MultiRobotTaskEnvironment:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.states = ['idle', 'task1', 'task2']
        self.actions = ['assign_task1', 'assign_task2', 'do_nothing']
        self.rewards = {'idle': -1, 'task1': 10, 'task2': 10}
        self.transition_probabilities = {
            'idle': {'assign_task1': 'task1', 'assign_task2': 'task2', 'do_nothing': 'idle'},
            'task1': {'assign_task1': 'task1', 'assign_task2': 'task1', 'do_nothing': 'idle'},
            'task2': {'assign_task1': 'task2', 'assign_task2': 'task2', 'do_nothing': 'idle'}
        }
        self.states_per_robot = ['idle'] * num_robots

    def reset(self):
        self.states_per_robot = ['idle'] * self.num_robots
        return self.states_per_robot

    def step(self, actions):
        next_states = []
        rewards = []
        for i in range(self.num_robots):
            current_state = self.states_per_robot[i]
            action = actions[i]
            next_state = self.transition_probabilities[current_state][action]
            reward = self.rewards[next_state]
            if np.random.rand() < 0.5:
                next_state = 'idle'
            next_states.append(next_state)
            rewards.append(reward)
        self.states_per_robot = next_states
        return next_states, rewards

def policy_evaluation(policy, states, rewards, transition_probabilities, num_robots, discount_factor=0.9, theta=0.0001):
    V = {state: 0 for state in states}
    while True:
        delta = 0
        for state in states:
            v = V[state]
            action = policy[state]
            next_state = transition_probabilities[state][action]
            V[state] = rewards[next_state] + discount_factor * V[next_state]
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(V, states, actions, rewards, transition_probabilities, discount_factor=0.9):
    policy = {}
    for state in states:
        action_values = {}
        for action in actions:
            next_state = transition_probabilities[state][action]
            action_values[action] = rewards[next_state] + discount_factor * V[next_state]
        policy[state] = max(action_values, key=action_values.get)
    return policy

def policy_iteration(states, actions, rewards, transition_probabilities, num_robots, discount_factor=0.9, theta=0.0001):
    policy = {state: actions[0] for state in states}
    while True:
        V = policy_evaluation(policy, states, rewards, transition_probabilities, num_robots, discount_factor, theta)
        new_policy = policy_improvement(V, states, actions, rewards, transition_probabilities, discount_factor)
        if new_policy == policy:
            break
        policy = new_policy
    return policy, V

def simulate_policy(env, policy, num_robots, episodes=10):
    results = []
    task_counts = defaultdict(int)
    idle_counts = defaultdict(int)
    for _ in range(episodes):
        states = env.reset()
        episode_rewards = 0
        steps = 0
        while steps < 50:  # Limit to 50 steps to prevent infinite loops
            actions = [policy[state] for state in states]
            next_states, rewards = env.step(actions)
            episode_rewards += sum(rewards)
            states = next_states
            steps += 1
            # Track task completions and idle counts for each robot
            for i, state in enumerate(states):
                if state.startswith('task'):
                    task_counts[state] += 1
                elif state == 'idle':
                    idle_counts[f'robot_{i+1}'] += 1
            # Exit loop if all robots have completed at least one task
            if all(count > 0 for count in task_counts.values()):
                break
        results.append(episode_rewards)

    return results, task_counts, idle_counts

num_robots = 2
env = MultiRobotTaskEnvironment(num_robots)
optimal_policy, optimal_value_function = policy_iteration(env.states, env.actions, env.rewards, env.transition_probabilities, num_robots)

print("Optimal Policy:")
print(optimal_policy)
print("Optimal Value Function:")
print(optimal_value_function)

simulation_results, task_counts, idle_counts = simulate_policy(env, optimal_policy, num_robots, episodes=100)

print("Simulation Results:")
print(simulation_results)
print("Average Reward:", np.mean(simulation_results))
print("Task Completion Counts:")
print(task_counts)
print("Idle Counts per Robot:")
print(idle_counts)
