"""
Brains
=======

Contains classes and methods for building neural networks.

Classes:
    NN:  Traditional Neural Network with arbitrary amount of hidden layers and a training buffer
    DQN: Deep Q Network with an arbitrary amount of hidden layers and a training buffer

Author: Bradley Reimers
Date: 10/28/2023
License: GPL 3
"""
import random
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.utils import disable_interactive_logging

disable_interactive_logging()


class NN:
    """
    Neural Network for Organoid Agents.

    This class defines a neural network used by organoid agents to make decisions.

    Attributes:
        discount (float): Discount factor for future rewards.
        eps (float): Exploration rate.
        eps_decay (float): Rate at which exploration rate decays.
        state_space_size (int): Dimension of the state space.
        action_space_size (int): Dimension of the action space.
        model (Sequential): Keras Sequential model for the neural network.

    Methods:
        choose_action(state):
            Choose an action for the given state.

        train(state, action, reward, new_state):
            Train the neural network based on experiences.

    """

    def __init__(
        self,
        discount=0.95,
        eps=0.3,
        eps_decay=0.95,
        hidden_sizes=[64, 32, 48, 16, 8, 4],
        state_space_size=46,
        action_space_size=2,
        buffer_size=20,
    ):
        """
        Initialize the Neural Network.

        Args:
            discount (float, optional): Discount factor for future rewards. Default is 0.95.
            eps (float, optional): Exploration rate. Default is 0.5.
            eps_decay (float, optional): Rate at which exploration rate decays. Default is 0.999.
            hidden_sizes (list, optional): List of hidden layer sizes. Default is [20].
            state_space_size (int, optional): Dimension of the state space. Default is 7.
            action_space_size (int, optional): Dimension of the action space. Default is 2.

        """
        self.discount = discount
        self.eps = eps
        self.eps_decay = eps_decay
        self.hidden_sizes = hidden_sizes
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = int(buffer_size * 0.2)
        self.model = self.build_network()

        
    def build_network(self):
        """
        Builds the base model for training

        """
        model = Sequential()
        model.add(InputLayer((self.state_space_size)))
        for size in self.hidden_sizes:
            model.add(Dense(size, activation="relu"))
        model.add(Dense(self.action_space_size, activation="linear"))
        model.compile(loss="mse", optimizer="adam", metrics=["mae"])
        return model
    
    def choose_action(self, state):
        """
        Choose an action for the given state.

        Args:
            state (numpy.ndarray): The current state.

        Returns:
            tuple: A tuple of two discrete float values between -1 and 1 representing the 
                chosen action.

        """
        self.eps *= self.eps_decay
        if np.random.rand() <= self.eps:
            # Explore: choose a random action
            return (random.uniform(-1, 1), random.uniform(-1, 1))
        else:
            # Exploit: choose the action with the highest Q-value
            x, y = self.model(state, training=False)[0]
            mean = (abs(x) + abs(y)) / 2
            return (min([max([x / mean, -1]), 1]), min([max([y / mean, -1]), 1]))

    def train(self, state, action, reward, new_state):
        """
        Train the neural network based on experiences.

        Args:
            state (numpy.ndarray): The current state.
            action (tuple): The chosen action.
            reward (float): The reward received.
            new_state (numpy.ndarray): The new state.

        """
        ## Add experential buffer for stable training
        self.buffer.append((state, action, reward, new_state))
        if len(self.buffer) >= self.buffer.maxlen:
            # Sample a batch of experiences from the buffer
            batch = random.sample(self.buffer, self.batch_size)

            # Prepare the training data
            states, targets = [], []
            for state, action, reward, new_state in batch:
                target = reward + self.discount * np.max(self.model.predict(new_state))
                target_f = self.model.predict(state)
                target_f[0] = self.model.predict(state)[0]
                target_f[0][0] = target

                states.append(state)
                targets.append(target_f)

            # Train the model on the batch of experiences
            self.model.fit(
                np.array(states).reshape(-1, self.state_space_size),
                np.array(targets).reshape(-1, self.action_space_size),
                epochs=1,
                verbose=0,
            )

        # Implement DQN training here
        # Update the Q-values based on the Bellman equation
        target = reward + self.discount * np.max(self.model.predict(new_state))
        model_output = self.model.predict(state)

        # Copy the model's prediction
        target_f = model_output.copy()

        # Update all elements in the action space
        target_f[0] = model_output[0]

        # Set the first action element to the target
        target_f[0][0] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)


class DQN(NN):
    """
    Deep Q Network (DQN) for Reinforcement Learning.

    This class represents a Deep Q Network used in reinforcement learning tasks.

    Attributes:
        discount (float): Discount factor for future rewards (default: 0.95).
        eps (float): Epsilon value for epsilon-greedy exploration (default: 0.3).
        eps_decay (float): Epsilon decay rate (default: 0.95).
        hidden_sizes (list): List of integers, specifying the sizes of hidden layers 
            (default: [64, 32, 48, 16, 8, 4]).
        state_space_size (int): Dimension of the state space (default: 46).
        action_space_size (int): Dimension of the action space (default: 2).
        buffer_size (int): Maximum size of the replay buffer (default: 20).

    Methods:
        __init__(self, discount=0.95, eps=0.3, eps_decay=0.95, 
            hidden_sizes=[64, 32, 48, 16, 8, 4], state_space_size=46, 
              action_space_size=2, buffer_size=20):
            Initialize the DQN with the provided hyperparameters.

        build_q_network(self, state_size, action_size):
            Build a Q-network model with specified input and output dimensions.

        update_q_model(self):
            Update the target Q-network with the weights from the online Q-network.

        train(self, *args, **kwargs):
            Train the DQN using experiences from the replay buffer.

    """

    def __init__(
        self,
        discount=0.95,
        eps=0.3,
        eps_decay=0.95,
        hidden_sizes=[64, 32, 48, 16, 8, 4],
        state_space_size=46,
        action_space_size=2,
        buffer_size=20,
    ):
        """
        Initialize the DQN with the provided hyperparameters.

        Args:
            discount (float, optional): Discount factor for future rewards (default: 0.95).
            eps (float, optional): Epsilon value for epsilon-greedy exploration (default: 0.3).
            eps_decay (float, optional): Epsilon decay rate (default: 0.95).
            hidden_sizes (list, optional): List of integers, specifying the sizes of hidden layers 
                (default: [64, 32, 48, 16, 8, 4]).
            state_space_size (int, optional): Dimension of the state space (default: 46).
            action_space_size (int, optional): Dimension of the action space (default: 2).
            buffer_size (int, optional): Maximum size of the replay buffer (default: 20).

        """
        super().__init__(discount, eps, eps_decay, hidden_sizes, state_space_size, action_space_size, buffer_size)
        self.q_model = self.build_network()
        self.update_q_model()

    def build_network(self):
        """
        Build a Q-network model with specified input and output dimensions.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.

        Returns:
            keras.models.Sequential: A Q-network model.

        """
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_space_size,)))
        # Add hidden layers
        for size in self.hidden_sizes:
            model.add(Dense(size, activation="relu"))
        # Output layer with linear activation (Q-values)
        model.add(Dense(self.action_space_size, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def update_q_model(self):
        """
        Update the target Q-network with the weights from the online Q-network.

        """
        self.q_model.set_weights(self.model.get_weights())

    def train(self, *args, **kwargs):
        """
        Train the DQN using experiences from the replay buffer.

        This method samples a mini-batch from the replay buffer, calculates target Q-values
        using the target network, and updates the online network.

        """
        if len(self.buffer) < self.batch_size:
            return

        # Sample a mini-batch from the replay buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert states and next_states to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)

        # Calculate the target Q-values using the target network
        target_q_values = self.q_model.predict(next_states)
        max_target_q_values = np.max(
            target_q_values, axis=1
        )  # Calculate the maximum Q-value for each sample

        # Calculate the online network's Q-values for the current states
        current_q_values = self.model.predict(states)

        # Update the Q-values for the taken actions
        for i in range(self.batch_size):
            current_q_values[i][actions[i]] = (
                rewards[i] + self.discount * max_target_q_values[i]
            )

        # Train the online network on the mini-batch
        self.model.fit(states, current_q_values, verbose=0)

        # Update the target network periodically
        self.update_q_model()
