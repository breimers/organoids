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
    discount=0.95, eps=0.5, eps_decay=0.999, hidden_sizes=[20], state_space_size=7, action_space_size=2
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
    discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[14, 28, 14, 8, 4], state_space_size=7, action_space_size=2
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
    discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[92, 20, 46, 8, 4], state_space_size=46, action_space_size=2
```

## Experiment 3b
- Same conditions as 3a, except `eps=0.5`


## Experiment 4
- Environmental conditions:
    - 500 steps
    - 5s food spawn interval
    - 100 Food
    - 2 basic organoids
    - 2 evolved organoids
    - 5 obstacles
- NN Settings:
```python
    discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[64, 32, 16, 8, 4], state_space_size=46, action_space_size=2
```

## Experiment 5
- Changelog:
    - Lowered reproduction cooldown from 500 to 100 steps
    - Updated prediction logic to go from batch to single
- Environmental conditions:
    - 500 steps
    - 5s food spawn interval
    - 100 Food
    - 2 basic organoids
    - 2 evolved organoids
    - 5 obstacles
- NN Settings:
```python
    discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[64, 32, 48, 16, 8, 4], state_space_size=46, action_space_size=2
```

## Experiment 6
CHangelog:
    - Added experential buffer
    - Added DQN option
- Environmental conditions:
    - 500 steps
    - 5s food spawn interval
    - 100 Food
    - 3 basic organoids
    - 1 evolved organoids (NN)
    - 1 dqn evolved organoids (DQN)
    - 5 obstacles
    - Lowered reproduction cooldown from 500 to 100 steps
- NN Settings:
```python
    discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[64, 32, 48, 16, 8, 4], state_space_size=46, action_space_size=2, buffer_size=100
```