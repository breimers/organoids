"""
Organoids Simulation Project

This project represents an organoid simulation, which includes a world populated with organoids,
    food, and obstacles.
The organoids exhibit behaviors, interact with their environment, and evolve over time.
This simulation serves as a learning platform to explore various aspects of artificial life 
    and reinforcement learning.

Modules:
    - organoids: Contains classes and functions for defining organoids, food, and obstacles.
    - world: Defines the World class that represents the simulation environment.
    - neural_network: Implements deep learning models, including DQN, for reinforcement learning.

Usage:
    To run the organoid simulation, create an instance of the World class, 
    populate it with organoids, food, and obstacles, and call the `run_simulation()` method. 
        The simulation can be customized with different parameters.

Authors:
    - Bradley Reimers

Version:
    Organoids Project v1.0

License:
    This project is licensed under the GNU General Public License (GPL-3.0)

"""

from organoids.world import World

if __name__ == "__main__":
    # Create the world

    # Define parameters for objects
    organoid_params = {
        "name": "Silly Blob",
        "lifespan": 501,
        "size": 5,
        "calories": 50,
        "calorie_limit": 50,
        "metabolism": 0.01,
        "rgb": (255, 10, 10),
        "position": (0, 0),
        "smart": False,
        "cooldown_duration": 10,
        "modeltype": None,
        "hidden_layers": [32, 8, 16, 4],
    }

    nn_organoid_params = organoid_params.copy()
    nn_organoid_params["smart"] = True
    nn_organoid_params["name"] = "Brainy Blob"
    nn_organoid_params["modeltype"] = "NN"
    nn_organoid_params["rgb"] = (10, 10, 255)

    dqn_organoid_params = nn_organoid_params.copy()
    dqn_organoid_params["name"] = "Learny Blob"
    dqn_organoid_params["modeltype"] = "DQN"
    dqn_organoid_params["rgb"] = (128, 10, 128)

    cnn_organoid_params = nn_organoid_params.copy()
    cnn_organoid_params["name"] = "Clever Blob"
    cnn_organoid_params["modeltype"] = "CNN"
    cnn_organoid_params["hidden_layers"] = [32, 32]
    cnn_organoid_params["rgb"] = (10, 128, 128)

    dqcnn_organoid_params = cnn_organoid_params.copy()
    dqcnn_organoid_params["name"] = "Magic Blob"
    dqcnn_organoid_params["modeltype"] = "DQCNN"
    dqcnn_organoid_params["rgb"] = (47, 152, 161)

    organoid_pops = [
        (1, organoid_params),
        (1, nn_organoid_params),
        (1, cnn_organoid_params),
        (1, dqcnn_organoid_params),
    ]
    food_params = {"name": "Algae", "size": 2.5, "calories": 200, "rgb": (10, 255, 10)}
    obstacle_params = {"name": "Rock", "size": 7, "rgb": (100, 100, 100)}

    # Spawn initial organoids and food

    world = World(
        name="midgard",
        radius=100,
        doomsday_ticker=500,
        obstacle_ratio=0.05,
        abundance=100.00,
        show=True,
        food_params=food_params,
        obstacle_params=obstacle_params,
        organoid_pops=organoid_pops,
    )
    # Kick off visualization
    world.run_simulation()
