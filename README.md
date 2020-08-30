# DQN Maze

![model](model.png)

The topology of the agent's NoisyNet. 
In hidden layers is used **swish** activation function and linear activation function is used in the output layer.

## States

26 inputs = 2 position + 24 objects around agent

## Actions

* Up
* Down
* Left
* Right

**Run**
```
python3 main.py
```

## License

[![MIT](https://img.shields.io/github/license/markub3327/DQN_MazeSolver.svg)](LICENSE)
