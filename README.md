# organoids
NN powered Ecosim Automata

# about

This is a passion project of mine to create an open-source ecosim powered by a flexible NN architecture.
While it's currently a simple implementation, it may serve as the base for more complex projects in the future.

# getting started

The organoids module contains five classes of objects:
 - NN: Neural network with configurable amount of hidden layers
 - Organoids: agents of the world powered by NNs
 - Obstacles: simply random areas the organoids must navigate around
 - Food: resources required to grow and sustain organoids
 - World: simulated ecosystem, configurable with varying size, abundance, replishment, and obstacles

To run the simulation, edit the values under the `if __name__ == "__main"` section, and then in your terminal `python3 ./organoids.py`.

