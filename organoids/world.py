"""
Worlds
=======

Contains classes and methods for building a simulated worldspace.

Classes:
    World: Represents the simulation world, including objects and simulation parameters.

Author: Bradley Reimers
Date: 11/19/2023
License: GPL 3
"""

import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
from .objects import Organoid, Food, Obstacle


class World:
    """
    World Class.

    This class represents the simulation world, including objects and simulation parameters.

    Attributes:
        name (str): The name of the self.
        radius (int): The radius of the self.
        abundance (float): Abundance factor for food generation.
        obstacle_ratio (float): Ratio of obstacles to the world size.
        doomsday_ticker (int): Maximum number of simulation iterations.
        organoids (list): List of organoid objects.
        food (list): List of food objects.
        obstacles (list): List of obstacle objects.

    Methods:
        __init__(self, name, radius, doomsday_ticker, obstacle_ratio, abundance):
            Initialize the World object with simulation parameters.

        handle_collisions(self):
            Handle collisions between organoids, food, and obstacles.

        spawn_continuous_food(self, interval):
            Spawn food objects continuously at a specified time interval.

        run_simulation(self):
            Run the simulation, updating the visualization and organoid behaviors.

        spawn_organoids(self, num_organoids, organoid_params):
            Spawn a specified number of organoids with given parameters.

        spawn_food(self, num_food, food_params):
            Spawn a specified number of food objects with given parameters.

        spawn_walls(self):
            Spawn wall obstacles along the perimeter of the self.

        spawn_obstacles(self, num_obstacles, obstacle_params):
            Spawn a specified number of obstacles with given parameters.

        random_point_in_world(self):
            Generate a random point within the boundaries of the self.

        simulate_step(self):
            Simulate a single step of the world, updating organoids, food, and handling collisions.

    """

    def __init__(
        self,
        name,
        radius,
        doomsday_ticker,
        obstacle_ratio,
        abundance,
        show,
        food_params=dict(),
        obstacle_params=dict(),
        organoid_pops=list(),
    ):
        """
        Initialize the World object with simulation parameters.

        Args:
            name (str): The name of the self.
            radius (int): The radius of the self.
            doomsday_ticker (int): Maximum number of simulation iterations.
            obstacle_ratio (float): Ratio of obstacles to the world size.
            abundance (float): Abundance factor for food generation.

        """
        self.name = str(name) or "Midgard"
        self.radius = int(radius) or 100
        self.abundance = float(abundance) or 1.00
        self.obstacle_ratio = float(obstacle_ratio) or 0.01
        self.doomsday_ticker = int(doomsday_ticker) or 100000
        self.step = 0
        self.world_run_id = (
            f"organoid_sim_{name}_{str(int(datetime.now().timestamp()))}"
        )
        self.organoids = []
        self.food = []
        self.obstacles = []
        self.show = show
        self.spawn_thread = threading.Thread(
            target=self.spawn_continuous_food,
            args=(5, food_params),
        )
        self.food_params = (food_params,)
        self.obstacle_params = (obstacle_params,)
        self.organoid_pops = organoid_pops

    def handle_collisions(self):
        """
        Handle collisions between organoids, food, and obstacles.

        This method checks for collisions between organoids and other objects,
        including food and obstacles, and handles them accordingly.

        """
        collided_organoids = set()  # To keep track of collided organoids

        for organoid in self.organoids:
            if organoid.alive and organoid is not None:
                for food in self.food:
                    if organoid.distance_to(food) <= organoid.size + food.size:
                        organoid.consume_food(food)
                for obstacle in self.obstacles:
                    if organoid.distance_to(obstacle) <= organoid.size + obstacle.size:
                        # Bounce the organoid away from the obstacle
                        x_diff = organoid.position[0] - obstacle.position[0]
                        y_diff = organoid.position[1] - obstacle.position[1]
                        organoid.position = (
                            organoid.position[0] + 1 * x_diff,
                            organoid.position[1] + 1 * y_diff,
                        )
                # Check for collisions between organoids
                for other in self.organoids:
                    if (
                        organoid != other
                        and organoid.distance_to(other) <= organoid.size + other.size
                    ):
                        collided_organoids.add(organoid)
                        collided_organoids.add(other)
        # Create new organoids for the collided ones
        if collided_organoids:
            mother = next(iter(collided_organoids))
            if mother is not None and mother.alive:
                self.organoids.append(mother.split_organoid())

    def spawn_continuous_food(self, interval, food_params):
        """
        Spawn food objects continuously at a specified time interval.

        Args:
            interval (float): The time interval for spawning food.

        """
        while True:
            time.sleep(interval)
            if len(self.food) < self.abundance:
                food = Food(**food_params)
                food.position = self.random_point_in_world()
                self.food.append(food)

    def generate_score_card(self):
        """
        Generate a scorecard containing information about all organoids.

        Returns:
            pd.DataFrame: DataFrame containing organoid information such as ID, score, name, size, etc.

        """
        scores = list()
        all_organoids = self.organoids.copy()
        all_organoids.extend(self.dead_organoids.copy())
        for o in all_organoids:
            if o is not None:
                scores.append(
                    {
                        "id": o.id,
                        "score": o.score,
                        "name": o.name,
                        "size": o.size,
                        "alive": o.alive,
                        "calories": o.calories,
                        "max_calories": o.calorie_limit,
                        "children": o.children,
                        "lifespan": o.lifespan,
                        "modeltype": o.modeltype,
                        "hidden_layers": o.hidden_layers,
                    }
                )
        scorecard = pd.DataFrame.from_records(scores)
        return scorecard

    def print_scorecard(self):
        """
        Generate and save a CSV file with the scorecard.

        The CSV file is saved with a filename based on the world run ID.

        """
        self.scorecard = self.generate_score_card()
        self.scorecard.to_csv(f"{self.world_run_id}.csv")

    def run_simulation(self):
        """
        Run the simulation, updating the visualization and organoid behaviors.

        This method initializes the visualization and runs the simulation loop,
        updating the state of the world and organoids.

        """
        self.spawn_food(num_food=self.abundance, food_params=self.food_params)
        self.spawn_obstacles(
            self.obstacle_ratio * self.abundance, obstacle_params=self.obstacle_params
        )
        self.spawn_walls()
        for pop, params in self.organoid_pops:
            self.spawn_organoids(num_organoids=pop, organoid_params=params)
        self.spawn_thread.start()
        if self.show:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 2 * self.radius)
            ax.set_ylim(0, 2 * self.radius)
            ax.set_xticks([])
            ax.set_yticks([])
            organoids_scatter = ax.scatter(
                [], [], c="red", marker="o", s=50, label="Organoids"
            )
            food_scatter = ax.scatter([], [], c="green", marker="s", s=20, label="Food")
            obstacle_scatter = ax.scatter(
                [], [], c="gray", marker="^", s=20, label="Obstacles"
            )
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=5,
            )
            ax.set_title(self.world_run_id)
            text = ax.text(
                0,
                0,
                "",
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                visible=False,
            )

        # Connect the mouse motion event to update_plot function
        def normalize_rgb(rgb):
            # Normalize RGB values to range [0, 1]
            return tuple(comp / 255.0 for comp in rgb)

        def update_plot(event=None):
            """
            Update the logical plot and visualize.
            """
            if len(self.organoids) >= 1:
                organoids_x = [
                    org.position[0] for org in self.organoids if org is not None
                ]
                organoids_y = [
                    org.position[1] for org in self.organoids if org is not None
                ]
                organoids_size = [
                    np.pi * org.size**2 for org in self.organoids if org is not None
                ]  # Calculate area as marker size
                organoids_colors = [
                    normalize_rgb(org.rgb) for org in self.organoids if org is not None
                ]
                organoids_scatter.set_offsets(
                    np.column_stack((organoids_x, organoids_y))
                )
                organoids_scatter.set_sizes(
                    organoids_size
                )  # Set the new sizes based on the area
                organoids_scatter.set_color(organoids_colors)
            if len(self.food) >= 1:
                food_x = [food.position[0] for food in self.food if food is not None]
                food_y = [food.position[1] for food in self.food if food is not None]
                food_size = [
                    np.pi * food.size**2 for food in self.food if food is not None
                ]
                food_colors = [
                    normalize_rgb(food.rgb) for food in self.food if food is not None
                ]
                food_scatter.set_offsets(np.column_stack((food_x, food_y)))
                food_scatter.set_sizes(food_size)
                food_scatter.set_color(food_colors)  # Set normalized colors
            if len(self.obstacles) >= 1:
                obstacle_x = [
                    obs.position[0] for obs in self.obstacles if obs is not None
                ]
                obstacle_y = [
                    obs.position[1] for obs in self.obstacles if obs is not None
                ]
                obstacle_size = [
                    np.pi * obs.size**2 for obs in self.obstacles if obs is not None
                ]
                obstacle_colors = [
                    normalize_rgb(obs.rgb) for obs in self.obstacles if obs is not None
                ]
                obstacle_scatter.set_offsets(np.column_stack((obstacle_x, obstacle_y)))
                obstacle_scatter.set_sizes(obstacle_size)
                obstacle_scatter.set_color(obstacle_colors)  # Set normalized colors

            if event is not None and event.inaxes == ax:
                mouse_x, mouse_y = event.xdata, event.ydata
                mouse_coords = np.array([mouse_x, mouse_y])

                # Check for proximity to organoids
                for org, x, y in zip(self.organoids, organoids_x, organoids_y):
                    obj_coords = np.array([x, y])
                    distance = np.linalg.norm(mouse_coords - obj_coords)
                    if distance <= org.size:
                        obj_info = org.get_info()
                        text.set_text(obj_info)
                        text.set_position((x, y + org.size + 2))
                        text.set_visible(True)
                        break
                else:
                    text.set_visible(False)

            plt.pause(0.001)

        if self.show:
            fig.canvas.mpl_connect(
                "motion_notify_event", lambda event: update_plot(event)
            )

        # The main thread continues with the simulation
        try:
            for i in range(self.doomsday_ticker):
                self.step = i + 1
                self.simulate_step()
                if self.show:
                    update_plot()
                time.sleep(0.005)
                print(
                    f"Step {self.step}, organoids: {len(self.organoids)}, Food: {len(self.food)}"
                )
            self.spawn_thread.join
        except KeyboardInterrupt:
            pass
        self.print_scorecard()

    def spawn_organoids(self, num_organoids, organoid_params):
        """
        Spawn a specified number of organoids with given parameters.

        Args:
            num_organoids (int): The number of organoids to spawn.
            organoid_params (dict): Parameters for configuring the organoids.

        """
        for _ in range(num_organoids):
            organoid = Organoid(**organoid_params)
            organoid.position = self.random_point_in_world()
            self.organoids.append(organoid)

    def spawn_food(self, num_food, food_params):
        """
        Spawn a specified number of food objects with given parameters.

        Args:
            num_food (int): The number of food objects to spawn.
            food_params (dict): Parameters for configuring the food objects.

        """
        food_added = 0
        while food_added < num_food:
            food = Food(**food_params)
            food.position = self.random_point_in_world()
            # Check if the obstacle position is within the world boundary
            if (
                food.position[0] >= self.radius * 2 - food.size
                or food.position[0] <= food.size
                or food.position[1] >= self.radius * 2 - food.size
                or food.position[1] <= food.size
            ):
                continue
            self.food.append(food)
            food_added += 1

    def spawn_walls(self):
        """
        Spawn wall obstacles along the perimeter of the self.

        This method creates wall obstacles to enclose the simulation self.

        """
        # Spawn walls along the perimeter of the world
        for x in range(0, 2 * self.radius + 1):
            self.obstacles.append(Obstacle("Wall", 1, (100, 100, 100)))
            self.obstacles[-1].position = (x, 0)
            self.obstacles.append(Obstacle("Wall", 1, (100, 100, 100)))
            self.obstacles[-1].position = (x, 2 * self.radius)

        for y in range(1, 2 * self.radius):
            self.obstacles.append(Obstacle("Wall", 1, (100, 100, 100)))
            self.obstacles[-1].position = (0, y)
            self.obstacles.append(Obstacle("Wall", 1, (100, 100, 100)))
            self.obstacles[-1].position = (2 * self.radius, y)

    def spawn_obstacles(self, num_obstacles, obstacle_params):
        """
        Spawn a specified number of obstacles with given parameters.

        Args:
            num_obstacles (int): The number of obstacles to spawn.
            obstacle_params (dict): Parameters for configuring the obstacles.

        """
        obstacles_added = 0
        while obstacles_added < num_obstacles:
            obstacle = Obstacle(**obstacle_params)
            obstacle.position = self.random_point_in_world()

            # Check if the obstacle position is within the world boundary
            if (
                obstacle.position[0] >= self.radius * 2 - obstacle.size
                or obstacle.position[0] <= obstacle.size
                or obstacle.position[1] >= self.radius * 2 - obstacle.size
                or obstacle.position[1] <= obstacle.size
            ):
                continue

            self.obstacles.append(obstacle)
            obstacles_added += 1

    def random_point_in_world(self):
        """
        Generate a random point within the boundaries of the self.

        Returns:
            tuple: A tuple representing the coordinates of a random point within the world
                boundaries.

        """
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, self.radius)
            x = int(self.radius + distance * np.cos(angle))
            y = int(self.radius + distance * np.sin(angle))

            # Check if the point is within the world boundaries
            if 0 <= x <= 2 * self.radius and 0 <= y <= 2 * self.radius:
                return x, y

    def simulate_step(self):
        """
        Simulate a single step of the world, updating organoids, food, and handling collisions.

        This method represents a single time step in the simulation and updates
        the state of organoids, food, and handles collisions between objects.

        """
        self.food = [food for food in self.food if food.calories > 0]
        self.organoids = [
            org for org in self.organoids if org is not None and org.alive
        ]
        self.dead_organoids = list()
        for organoid in self.organoids:
            organoid.update(self.food, self.organoids, self.obstacles, self.step)
            if (
                0 <= organoid.position[0] <= 2 * self.radius
                and 0 <= organoid.position[1] <= 2 * self.radius
            ):
                organoid.position = (organoid.position[0], organoid.position[1])
            else:
                organoid.position = (self.radius, self.radius)
            if not organoid.is_alive():
                self.dead_organoids.append(organoid)
                self.organoids.remove(organoid)
        self.handle_collisions()
        self.scorecard = self.generate_score_card()
