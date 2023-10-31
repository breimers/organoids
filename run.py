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

import threading
from organoids.world import World

if __name__ == "__main__":
    # Create the world
    world = World(
        name="Midgard",
        radius=100,
        doomsday_ticker=500,
        obstacle_ratio=0.01,
        abundance=100.00,
        show=True,
    )

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
        "cooldown_duration": 20,
        "modeltype": None,
        "hidden_layers": [32, 8, 16, 4],
    }
    evolved_params = organoid_params.copy()
    evolved_params["smart"] = True
    evolved_params["name"] = "Brainy Blob"
    evolved_params["modeltype"] = "NN"
    new_evolved_params = evolved_params.copy()
    new_evolved_params["name"] = "Learny Blob"
    new_evolved_params["modeltype"] = "DQN"
    food_params = {"name": "Algae", "size": 2.5, "calories": 200, "rgb": (10, 255, 10)}
    obstacle_params = {"name": "Rock", "size": 7, "rgb": (100, 100, 100)}

    # Spawn initial organoids and food

    world.spawn_food(num_food=100, food_params=food_params)
    world.spawn_obstacles(5, obstacle_params=obstacle_params)
    world.spawn_walls()
    world.spawn_organoids(num_organoids=2, organoid_params=organoid_params)
    world.spawn_organoids(num_organoids=1, organoid_params=evolved_params)
    world.spawn_organoids(num_organoids=1, organoid_params=new_evolved_params)

    FOOD_SPAWN_INTERVAL = (
        5  # Adjust the interval (in steps) for continuous food spawning
    )
    food_spawner_thread = threading.Thread(
        target=world.spawn_continuous_food, args=(FOOD_SPAWN_INTERVAL, food_params)
    )
    food_spawner_thread.start()

    # Kick off visualization
    world.run_simulation()
    food_spawner_thread.join()
