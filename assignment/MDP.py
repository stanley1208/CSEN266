import numpy as np

# Define the input data directly in the script

# States and Actions
states = ['S0', 'S1', 'S2']
actions = ['a0', 'a1']

# Transition probabilities (each action has a corresponding transition matrix)
transitions = {
    'a0': [
        [0.9, 0.1, 0.0],  # From S0 with action a0
        [0.3, 0.6, 0.1],  # From S1 with action a0
        [0.6, 0.0, 0.4]   # From S2 with action a0
    ],
    'a1': [
        [0.7, 0.3, 0.0],  # From S0 with action a1
        [0.0, 0.0, 1.0],  # From S1 with action a1
        [0.0, 0.0, 1.0]   # From S2 with action a1
    ]
}

# Rewards for each state-action pair
rewards = {
    ('S0', 'a0'): 2, ('S0', 'a1'): 3,
    ('S1', 'a0'): 0, ('S1', 'a1'): 1,
    ('S2', 'a0'): 0, ('S2', 'a1'): 1
}

# Parameters
gamma = 0.8     # Discount factor
epsilon = 0.001 # Convergence threshold

# Value iteration function
def value_iteration(states, actions, transitions, rewards, gamma, epsilon):
    values = np.zeros(len(states))  # Initialize state values to zero
    policy = {}

    while True:
        delta = 0
        new_values = values.copy()

        # Update value for each state
        for s_idx, state in enumerate(states):
            action_values = []

            # Calculate value for each action
            for action in actions:
                action_value = sum(
                    transitions[action][s_idx][next_s_idx] *
                    (rewards.get((state, action), 0) + gamma * values[next_s_idx])
                    for next_s_idx in range(len(states))
                )
                action_values.append(action_value)

            # Find the best action and value
            best_action_value = max(action_values)
            best_action = actions[np.argmax(action_values)]

            # Update the value and policy
            new_values[s_idx] = best_action_value
            policy[state] = (best_action, best_action_value)

            # Track convergence criteria
            delta = max(delta, abs(best_action_value - values[s_idx]))

        values = new_values

        # Stop if values have converged
        if delta < epsilon:
            break

    return policy

# Perform value iteration
optimal_policy = value_iteration(states, actions, transitions, rewards, gamma, epsilon)

# Print the optimal policy and state values
print("% Format: State: Action (Value)")
for state, (action, value) in optimal_policy.items():
    print(f"{state}: {action} ({value:.2f})")
