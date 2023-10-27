# Journal

## Experiment 1
- Environmental conditions:
    - 500 steps
    - 5s food spawn interval
    - 100 Food
    - 2 basic organoids
    - 2 evolved organoids
    - 5 obstacles
- NN Settings:
```python
    def __init__(self, discount=0.95, eps=0.5, eps_decay=0.999, hidden_sizes=[20], state_space_size=7, action_space_size=2):
        self.discount = discount
        self.eps = eps
        self.eps_decay = eps_decay
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.model = Sequential()
        self.model.add(InputLayer((6 + 10 * 4)))
        for size in hidden_sizes:
            self.model.add(Dense(size, activation='relu'))
        self.model.add(Dense(action_space_size, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
```

## Experiment 2
- Environmental conditions:
    - 500 steps
    - 5s food spawn interval
    - 100 Food
    - 2 basic organoids
    - 2 evolved organoids
    - 5 obstacles
- NN Settings:
```python
    def __init__(self, discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[14, 28, 14, 8, 4], state_space_size=7, action_space_size=2):
        self.discount = discount
        self.eps = eps
        self.eps_decay = eps_decay
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.model = Sequential()
        self.model.add(InputLayer((6 + 10 * 4)))
        for size in hidden_sizes:
            self.model.add(Dense(size, activation='relu'))
        self.model.add(Dense(action_space_size, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
```

## Experiment 3a
- Environmental conditions:
    - 500 steps
    - 5s food spawn interval
    - 100 Food
    - 2 basic organoids
    - 2 evolved organoids
    - 5 obstacles
- NN Settings:
```python
    def __init__(self, discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[92, 20, 46, 8, 4], state_space_size=46, action_space_size=2):
        self.discount = discount
        self.eps = eps
        self.eps_decay = eps_decay
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.model = Sequential()
        self.model.add(InputLayer((self.state_space_size)))
        for size in hidden_sizes:
            self.model.add(Dense(size, activation='relu'))
        self.model.add(Dense(action_space_size, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
```

## Experiment 3b
- Same conditions as 3a, except `eps=0.5`