import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
    
class BaseObject:
    def __init__(self, name="obj", size=1, position=(0,0), rgb=(0, 0, 0)):
        self.name = str(name) or "Object"
        self.size = float(size) or 1.0
        self.position = tuple(position) or (0.0, 0.0)
        self.rgb = tuple(rgb) or (100, 100, 100)

    def get_info(self):
        return f"{self.name}\nSize: {self.size:.2f}"

class Organoid(BaseObject):
    def __init__(self, name, lifespan, size, calories, calorie_limit, position, metabolism, rgb):
        super().__init__(name, size, position, rgb)
        self.name = str(name) or "Organoid"
        self.lifespan = int(lifespan) or 60 # seconds
        self.size = float(size) or 2 # pixel radius
        self.calories = int(calories) or 100 # energy
        self.calorie_limit = int(calorie_limit) or 100
        self.position = tuple(position) or (0,0)
        self.metabolism = float(metabolism) or 0.05
        self.alive = True
        self.rgb = tuple(rgb) or (100, 100, 100)
        self.reproduction_cooldown = 0
        self.cooldown_duration = 500
        self.mutation_rate = 0.2

        self.children = 0
        self.score = 0

    def update(self):
        self.move()
        self.metabolize()
        self.update_score()
        if self.calories <= 0:
            self.alive = False

    def get_info(self):
        base_info = super().get_info()
        return f"Name: {self.name}\nSize: {self.size:.2f}\nCalories: {int(self.calories)}/{self.calorie_limit}\nScore: {self.score}"

    def metabolize(self):
        # This checks if the organoid has excess calories, if so, grow and increase the calorie limit.
        if self.calories > self.calorie_limit:
            diff = self.calories - self.calorie_limit
            self.calories = self.calorie_limit  # Cap the calories at the limit
            self.size += diff * 0.001  # Increase the organoid size by a fraction of the excess calories

        self.calories -= self.metabolism * (self.size / 4)
        
    def split_organoid(self):
        if self.reproduction_cooldown == 0:
            # Create a new organoid next to the current one with base parameters
            offset_x = random.uniform(-self.size, self.size)
            offset_y = random.uniform(-self.size, self.size)
            new_position = (self.position[0] + offset_x, self.position[1] + offset_y)
            new_organoid = Organoid(name=self.name, lifespan=self.lifespan, size=(self.size / 2), calories=(self.calorie_limit / 2),
                                    calorie_limit=(self.calorie_limit / 2), position=new_position, metabolism=self.metabolism, rgb=self.rgb)

            # Mutate the parameters within 10% of the parent's values (except for size)
            parameter_names = ["lifespan", "calorie_limit", "metabolism", "rgb"]
            for param_name in parameter_names:
                parent_value = getattr(self, param_name)
                if param_name == "rgb":
                    new_organoid.rgb = (
                        random.uniform(parent_value[0] - (parent_value[0] * self.mutation_rate), parent_value[0] + (parent_value[0] * self.mutation_rate)),
                        random.uniform(parent_value[1] - (parent_value[1] * self.mutation_rate), parent_value[1] + (parent_value[1] * self.mutation_rate)),
                        random.uniform(parent_value[2] - (parent_value[2] * self.mutation_rate), parent_value[2] + (parent_value[2] * self.mutation_rate))
                    )
                    continue
                min_value = parent_value - (parent_value * self.mutation_rate)
                max_value = parent_value - (parent_value * self.mutation_rate)
                random_value = random.uniform(min_value, max_value)
                setattr(new_organoid, param_name, random_value)

            self.reproduction_cooldown = self.cooldown_duration
            new_organoid.reproduction_cooldown = self.cooldown_duration
            if new_organoid.is_alive():
                self.children += 1
                world.organoids.append(new_organoid)
            
    def update_cooldown(self):
        # Decrease the reproduction cooldown timer (if greater than zero)
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

    def move(self):
        x_off = random.randint(-1 * self.step_size, self.step_size)
        y_off = random.randint(-1 * self.step_size, self.step_size)
        new_x = self.position[0] + x_off
        new_y = self.position[1] + y_off

        # Check if the potential new position is within the world boundaries
        if 0 <= new_x <= 2 * world.radius and 0 <= new_y <= 2 * world.radius:
            self.position = (new_x, new_y)
        else:
            # If the potential new position is outside the bounds, move the organoid to the center
            self.position = (world.radius, world.radius)

    def consume_food(self, food):
        self.calories += food.calories
        food.calories = 0

    def is_alive(self):
        return self.alive and self.calories > 0

    def distance_to(self, other):
        return np.sqrt((self.position[0] - other.position[0])**2 + (self.position[1] - other.position[1])**2)

    def update_score(self):
        self.score = int(50*self.children + 5*self.calories + 10*self.size)
        
    @property
    def step_size(self):
        return int(self.size / 2)
    
class Food(BaseObject):
    def __init__(self, name, size, calories, rgb):
        super().__init__(name, size, position=(0,0), rgb=rgb)
        self.name = str(name) or "Pellet"
        self.size = int(size) or 1 # pixel radius
        self.calories = int(calories) or 1 # energy
        self.rgb = tuple(rgb) or (100, 100, 100)

    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}\nCalories: {self.calories:.2f}"

    @property
    def visibility(self):
        return int(self.size * sum(list(self.rgb)))

class Obstacle(BaseObject):
    def __init__(self, name, size, rgb):
        super().__init__(name, size, position=(0,0), rgb=rgb)
        self.name = str(name) or "Rock"
        self.size = int(size) or 1 # pixel radius
        self.rgb = tuple(rgb) or (100, 100, 100)

class World():
    def __init__(self, name, radius, doomsday_ticker, obstacle_ratio, abundance):
        self.name = str(name) or "Midgard"
        self.radius = int(radius) or 100
        self.abundance = float(abundance) or 1.00
        self.obstacle_ratio = float(obstacle_ratio) or 0.01
        self.doomsday_ticker = int(doomsday_ticker) or 100000
        self.organoids = []
        self.food = []
        self.obstacles = []

    def handle_collisions(self):
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
                        organoid.position = (organoid.position[0] + 5 * x_diff, organoid.position[1] + 5 * y_diff)

                # Check for collisions between organoids
                for other in self.organoids:
                    if organoid != other and organoid.distance_to(other) <= organoid.size + other.size:
                        collided_organoids.add(organoid)
                        collided_organoids.add(other)

        # Create new organoids for the collided ones
        if collided_organoids:
            mother = next(iter(collided_organoids))
            if mother is not None and mother.alive:
                mother.split_organoid()

    def spawn_continuous_food(self, interval):
        while True:
            time.sleep(interval)
            if len(self.food) < self.abundance:
                food = Food(**food_params)
                food.position = self.random_point_in_world()
                self.food.append(food)

    def run_simulation(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 2 * self.radius)
        ax.set_ylim(0, 2 * self.radius)
        ax.set_xticks([])
        ax.set_yticks([])
        organoids_scatter = ax.scatter([], [], c='red', marker='o', s=50, label='Organoids')
        food_scatter = ax.scatter([], [], c='green', marker='s', s=20, label='Food')
        obstacle_scatter = ax.scatter([], [], c='gray', marker='^', s=20, label='Obstacles')
        ax.legend(loc='upper right')

        def normalize_rgb(rgb):
            # Normalize RGB values to range [0, 1]
            return tuple(comp / 255.0 for comp in rgb)

        text = ax.text(0, 0, '', fontsize=8, ha='center', va='center', color='black', visible=False)

        # Connect the mouse motion event to update_plot function

        def update_plot(event=None):
            organoids_x = [org.position[0] for org in world.organoids if org is not None]
            organoids_y = [org.position[1] for org in world.organoids if org is not None]
            organoids_size = [np.pi * org.size**2 for org in world.organoids]  # Calculate area as marker size
            organoids_colors = [normalize_rgb(org.rgb) for org in world.organoids]
            organoids_scatter.set_offsets(np.column_stack((organoids_x, organoids_y)))
            organoids_scatter.set_sizes(organoids_size)  # Set the new sizes based on the area
            organoids_scatter.set_color(organoids_colors)

            food_x = [food.position[0] for food in world.food]
            food_y = [food.position[1] for food in world.food]
            food_size = [np.pi * food.size**2 for food in world.food]
            food_colors = [normalize_rgb(food.rgb) for food in world.food]
            food_scatter.set_offsets(np.column_stack((food_x, food_y)))
            food_scatter.set_sizes(food_size)
            food_scatter.set_color(food_colors)  # Set normalized colors

            obstacle_x = [obstacle.position[0] for obstacle in world.obstacles]
            obstacle_y = [obstacle.position[1] for obstacle in world.obstacles]
            obstacle_size = [np.pi * obstacle.size**2 for obstacle in world.obstacles]
            obstacle_colors = [normalize_rgb(obstacle.rgb) for obstacle in world.obstacles]
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
            
        fig.canvas.mpl_connect('motion_notify_event', lambda event: update_plot(event))
        
        # The main thread continues with the simulation
        for i in range(self.doomsday_ticker):
            world.simulate_step()
            update_plot()
            time.sleep(0.005)
            print(f"Step {i + 1}, organoids: {len(world.organoids)}, Food: {len(world.food)}")

    def spawn_organoids(self, num_organoids, organoid_params):
        for _ in range(num_organoids):
            organoid = Organoid(**organoid_params)
            organoid.position = self.random_point_in_world()
            self.organoids.append(organoid)

    def spawn_food(self, num_food, food_params):
        food_added = 0
        while food_added < num_food:
            food = Food(**food_params)
            food.position = self.random_point_in_world()
            # Check if the obstacle position is within the world boundary
            if food.position[0] >= self.radius*2 - food.size or food.position[0] <= food.size or \
               food.position[1] >= self.radius*2 - food.size or food.position[1] <= food.size:
                continue
            self.food.append(food)
            food_added += 1

    def spawn_walls(self):
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
        obstacles_added = 0
        while obstacles_added < num_obstacles:
            obstacle = Obstacle(**obstacle_params)
            obstacle.position = self.random_point_in_world()

            # Check if the obstacle position is within the world boundary
            if obstacle.position[0] >= self.radius*2 - obstacle.size or obstacle.position[0] <= obstacle.size or \
               obstacle.position[1] >= self.radius*2 - obstacle.size or obstacle.position[1] <= obstacle.size:
                continue

            self.obstacles.append(obstacle)
            obstacles_added += 1
            
    def random_point_in_world(self):
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, self.radius)
            x = int(self.radius + distance * np.cos(angle))
            y = int(self.radius + distance * np.sin(angle))

            # Check if the point is within the world boundaries
            if 0 <= x <= 2 * self.radius and 0 <= y <= 2 * self.radius:
                return x, y

    def simulate_step(self):
        self.food = [food for food in self.food if food.calories > 0]
        self.organoids = [org for org in self.organoids if org is not None and org.alive]
        for organoid in self.organoids:
            organoid.update()
            if not organoid.is_alive():
                self.organoids.remove(organoid)

        self.handle_collisions()
    
if __name__ == "__main__":
    # Create the world
    world = World(name="Midgard", radius=100, doomsday_ticker=10000, obstacle_ratio=0.01, abundance=100.00)

    # Define parameters for organoids and food
    organoid_params = {"name": "Organoid", "lifespan": 60, "size": 5, "calories": 50, "calorie_limit": 50, "metabolism": 0.01, "rgb": (255, 10, 10), "position": (0, 0)}
    food_params = {"name": "Algae", "size": 2.5, "calories": 200, "rgb": (10, 255, 10)}
    obstacle_params = {"name": "Rock", "size": 7, "rgb": (100, 100, 100)}

    # Spawn initial organoids and food

    world.spawn_organoids(num_organoids=5, organoid_params=organoid_params)
    world.spawn_food(num_food=100, food_params=food_params)
    world.spawn_obstacles(5, obstacle_params=obstacle_params)
    world.spawn_walls()

    food_spawn_interval = 1  # Adjust the interval (in seconds) for continuous food spawning
    food_spawner_thread = threading.Thread(target=world.spawn_continuous_food, args=(food_spawn_interval,))
    food_spawner_thread.start()

    # Kick off visualization
    world.run_simulation()