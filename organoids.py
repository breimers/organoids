import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from uuid import uuid4
from itertools import chain
from datetime import datetime

# Categories
# organoids=3
# food=2
# obstacle=1
# base=0

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
    def __init__(self, discount=0.95, eps=0.3, eps_decay=0.95, hidden_sizes=[92, 20, 46, 8, 4], state_space_size=46, action_space_size=2):
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
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.model = Sequential()
        self.model.add(InputLayer((self.state_space_size)))
        for size in hidden_sizes:
            self.model.add(Dense(size, activation='relu'))
        self.model.add(Dense(action_space_size, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    def choose_action(self, state):
        """
        Choose an action for the given state.

        Args:
            state (numpy.ndarray): The current state.

        Returns:
            tuple: A tuple of two discrete float values between -1 and 1 representing the chosen action.

        """
        self.eps *= self.eps_decay
        if np.random.rand() <= self.eps:
            # Explore: choose a random action
            print("going random")
            return (random.uniform(-1, 1), random.uniform(-1, 1))
        else:
            # Exploit: choose the action with the highest Q-value
            print("doing smart")
            x, y = self.model.predict(state).squeeze()
            mean = (abs(x) + abs(y)) / 2
            return (min([max([x/mean, -1]), 1]), min([max([y/mean, -1]), 1]))

    def train(self, state, action, reward, new_state):
        """
        Train the neural network based on experiences.

        Args:
            state (numpy.ndarray): The current state.
            action (tuple): The chosen action.
            reward (float): The reward received.
            new_state (numpy.ndarray): The new state.

        """
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


class BaseObject:
    """
    Base Object Class.

    This class defines a base object with common attributes and methods for other objects in the simulation.

    Attributes:
        name (str): The name of the object.
        id (uuid.UUID): The unique identifier of the object.
        size (float): The size of the object.
        position (tuple): The position of the object.
        rgb (tuple): The color of the object in RGB format.
        category (int): The category of the object (0 for base).

    Methods:
        get_info():
            Get information about the object.

    """
    def __init__(self, name="obj", size=1, position=(0,0), rgb=(0, 0, 0)):
        """
        Initialize a BaseObject.

        Args:
            name (str, optional): The name of the object. Default is "obj".
            size (float, optional): The size of the object. Default is 1.
            position (tuple, optional): The position of the object. Default is (0, 0).
            rgb (tuple, optional): The color of the object in RGB format. Default is (0, 0, 0).

        """
        self.name = str(name) or "Object"
        self.id = uuid4()
        self.size = float(size) or 1.0
        self.position = tuple(position) or (0.0, 0.0)
        self.rgb = tuple(rgb) or (100, 100, 100)
        self.category = 0

    def get_info(self):
        """
        Get information about the object.

        Returns:
            str: Information about the object, including its name and size.

        """
        return f"{self.name}\nSize: {self.size:.2f}"

class Organoid(BaseObject):
    """
    Organoid Class.

    This class defines organoid objects, which are agents in the simulation.

    Attributes:
        name (str): The name of the organoid.
        lifespan (int): The lifespan of the organoid in seconds.
        size (float): The size of the organoid (pixel radius).
        calories (int): The current energy level of the organoid.
        calorie_limit (int): The maximum energy level of the organoid.
        position (tuple): The position of the organoid.
        metabolism (float): The metabolism rate of the organoid.
        alive (bool): Whether the organoid is alive.
        rgb (tuple): The color of the organoid in RGB format.
        reproduction_cooldown (int): Cooldown time for reproduction.
        cooldown_duration (int): The duration of the cooldown.
        mutation_rate (float): Rate of mutation for offspring.
        vision_range (int): The range of vision for the organoid.
        brain (NN): The neural network controlling the organoid.
        children (int): The number of offspring.
        score (int): The score of the organoid.
        last_score (int): The previous score of the organoid.
        category (int): The category of the organoid (3 for organoid).
        smart (bool): Indicated whether a "brain" (NN) will  be attached

    Methods:
        filter_objs_by_distance(objects):
            Filter objects within the organoid's vision range.

        update(food, organoids, obstacles):
            Update the organoid's state.

        train_neural_network(delta_score, objects):
            Train the neural network based on experiences.

        get_info():
            Get information about the organoid.

        metabolize():
            Perform metabolism and potentially grow.

        split_organoid():
            Split the organoid to create offspring.

        update_cooldown():
            Update the reproduction cooldown.

        move(x_off, y_off):
            Move the organoid.

        consume_food(food):
            Consume food and gain calories.

        is_alive():
            Check if the organoid is alive.

        distance_to(other):
            Calculate the distance to another object.

        update_score():
            Update the organoid's score.

        step_size:
            Property to get the step size for movement.

    """
    def __init__(self, name, lifespan, size, calories, calorie_limit, position, metabolism, rgb, smart):
        """
        Initialize an Organoid.

        Args:
            name (str): The name of the organoid.
            lifespan (int): The lifespan of the organoid in seconds.
            size (float): The size of the organoid (pixel radius).
            calories (int): The initial energy level of the organoid.
            calorie_limit (int): The maximum energy level of the organoid.
            position (tuple): The initial position of the organoid.
            metabolism (float): The metabolism rate of the organoid.
            rgb (tuple): The color of the organoid in RGB format.

        """
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
        self.vision_range = 10
        self.brain = NN()
        self.children = 0
        self.score = 0
        self.last_score = self.score
        self.category = 3
        self.smart = smart

    def filter_objs_by_distance(self, objects):
        """
        Filter objects within the organoid's vision range.

        Args:
            objects (list): List of objects to filter.

        Returns:
            list: Filtered list of objects based on distance.

        """
        new_obj_list = list()
        for object in objects:
            if self.distance_to(object) <= self.vision_range:
                new_obj_list.append(
                    {
                        "x-position": object.position[0],
                        "y-position": object.position[1],
                        "size": object.size, 
                        "category": object.category,
                    }
                )
        return new_obj_list
    
    def update(self, food, organoids, obstacles):
        """
        Update the organoid's state.

        Args:
            food (list): List of food objects.
            organoids (list): List of organoid objects.
            obstacles (list): List of obstacle objects.

        """
        if self.smart:
            organoids = [organoid for organoid in organoids if organoid.id != self.id]
            objects = list(food)
            objects.extend(obstacles)
            objects.extend(organoids)
            objects = self.filter_objs_by_distance(objects)
            delta_score = self.score - self.last_score
            self.train_neural_network(delta_score, objects)
        else:
            self.move(random.uniform(-1.00, 1.00), random.uniform(-1.00, 1.00))
            self.metabolize()
            self.update_score()
        if self.calories <= 0:
            self.alive = False

    def train_neural_network(self, delta_score, objects):
        """
        Train the neural network based on experiences.

        Args:
            delta_score (int): The change in the organoid's score.
            objects (list): List of objects within the organoid's vision range.

        """
        # Determine the maximum number of objects to consider
        max_objects = 10  # Adjust this based on your requirements

        # Create arrays to hold the object representations and their counts
        object_representations = np.zeros((max_objects, 4), dtype=float)  # Assuming 4 attributes per object
        object_counts = min(len(objects), max_objects)

        # Flatten attributes for each object
        for i in range(object_counts):
            obj = objects[i]
            object_representations[i] = [obj["category"], obj["size"], obj["x-position"], obj["y-position"]]

        # Fill any remaining slots with zeros
        for i in range(object_counts, max_objects):
            object_representations[i] = [0, 0, 0, 0]

        # Construct the state and action arrays
        state = np.array([
            self.calories, 
            self.size, 
            self.position[0], 
            self.position[1], 
            delta_score,
            *object_representations.flatten(),
            self.reproduction_cooldown,
        ]).reshape(1, -1)

        action = self.brain.choose_action(state)

        # Calculate new position based on action
        print(action)
        x_off, y_off = action
        self.move(x_off, y_off)
        self.metabolize()
        self.update_score()

        # Calculate reward (using delta_score as reward)
        reward = self.score - self.last_score

        new_state = np.array([
            self.calories, 
            self.size, 
            self.position[0], 
            self.position[1], 
            reward,
            *object_representations.flatten(),
            self.reproduction_cooldown,
        ]).reshape(1, -1)

        self.brain.train(state, action, reward, new_state)

    def get_info(self):
        """
        Get information about the organoid.

        Returns:
            str: Information about the organoid, including its name and current state.

        """
        base_info = super().get_info()
        return f"Name: {self.name}\nSize: {self.size:.2f}\nCalories: {int(self.calories)}/{self.calorie_limit}\nScore: {self.score}"

    def metabolize(self):
        """
        Perform metabolism and potentially grow.

        """
        # This checks if the organoid has excess calories, if so, grow and increase the calorie limit.
        if self.calories > self.calorie_limit:
            diff = self.calories - self.calorie_limit
            self.calories = self.calorie_limit  # Cap the calories at the limit
            self.size += diff * 0.001  # Increase the organoid size by a fraction of the excess calories

        self.calories -= self.metabolism * (self.size / 4)
        
    def split_organoid(self):
        """
        Split the organoid to create offspring.

        """
        if self.reproduction_cooldown == 0:
            # Create a new organoid next to the current one with base parameters
            offset_x = random.uniform(-self.size, self.size)
            offset_y = random.uniform(-self.size, self.size)
            new_position = (self.position[0] + offset_x, self.position[1] + offset_y)
            new_organoid = Organoid(name=self.name, lifespan=self.lifespan, size=(self.size / 2), calories=(self.calorie_limit / 2),
                                    calorie_limit=(self.calorie_limit / 2), position=new_position, metabolism=self.metabolism, rgb=self.rgb, smart=self.smart)

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
                max_value = parent_value + (parent_value * self.mutation_rate)
                random_value = random.uniform(min_value, max_value)
                setattr(new_organoid, param_name, random_value)

            self.reproduction_cooldown = self.cooldown_duration
            new_organoid.reproduction_cooldown = self.cooldown_duration
            if new_organoid.is_alive():
                self.children += 1
                world.organoids.append(new_organoid)
            
    def update_cooldown(self):
        """
        Update the reproduction cooldown.

        """
        # Decrease the reproduction cooldown timer (if greater than zero)
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
            
    def move(self, x_off, y_off):
        """
        Move the organoid.

        Args:
            x_off (float): Offset for the X-axis.
            y_off (float): Offset for the Y-axis.

        """
        new_x = self.position[0] + x_off*self.step_size
        new_y = self.position[1] + y_off*self.step_size

        # Check if the potential new position is within the world boundaries
        if 0 <= new_x <= 2 * world.radius and 0 <= new_y <= 2 * world.radius:
            self.position = (new_x, new_y)
        else:
            # If the potential new position is outside the bounds, move the organoid to the center
            self.position = (world.radius, world.radius)

    def consume_food(self, food):
        """
        Consume food and gain calories.

        Args:
            food (list): List of food objects.

        """
        self.calories += food.calories
        food.calories = 0

    def is_alive(self):
        """
        Check if the organoid is alive.

        Returns:
            bool: True if the organoid is alive, False otherwise.

        """
        return self.alive and self.calories > 0

    def distance_to(self, other):
        """
        Calculate the distance to another object.

        Args:
            other (BaseObject): Another object.

        Returns:
            float: The distance to the other object.

        """
        return np.sqrt((self.position[0] - other.position[0])**2 + (self.position[1] - other.position[1])**2)

    def update_score(self):
        """
        Update the organoid's score.

        """
        self.last_score = self.score
        self.score = int(20*self.children + 5*self.calories + 10*self.size)
        
    @property
    def step_size(self):
        """
        Get the step size for movement.

        Returns:
            float: The step size for movement.

        """
        return int(self.size / 2)
    
class Food(BaseObject):
    """
    Food Class.

    This class defines food objects in the simulation.

    Attributes:
        name (str): The name of the food.
        size (float): The size of the food (pixel radius).
        position (tuple): The position of the food.
        calories (int): The energy level provided by the food.
        category (int): The category of the food (1 for food).

    Methods:
        get_info():
            Get information about the food.

    """
    def __init__(self, name, size, calories, rgb):
        """
        Initialize a Food object.

        Args:
            name (str): The name of the food.
            size (float): The size of the food (pixel radius).
            position (tuple): The position of the food.
            calories (int): The energy level provided by the food.
            rgb (tuple): The color of the food in RGB format.

        """
        super().__init__(name, size, position=(0,0), rgb=rgb)
        self.name = str(name) or "Pellet"
        self.size = int(size) or 1 # pixel radius
        self.calories = int(calories) or 1 # energy
        self.rgb = tuple(rgb) or (100, 100, 100)
        self.category = 2

    def get_info(self):
        """
        Get information about the food.

        Returns:
            str: Information about the food, including its name and size.

        """
        base_info = super().get_info()
        return f"{base_info}\nCalories: {self.calories:.2f}"

    @property
    def visibility(self):
        return int(self.size * sum(list(self.rgb)))

class Obstacle(BaseObject):
    """
    Obstacle Class.

    This class defines obstacle objects in the simulation.

    Attributes:
        name (str): The name of the obstacle.
        size (float): The size of the obstacle (pixel radius).
        position (tuple): The position of the obstacle.
        category (int): The category of the obstacle (2 for obstacle).

    Methods:
        get_info():
            Get information about the obstacle.

    """
    def __init__(self, name, size, rgb):
        """
        Initialize an Obstacle object.

        Args:
            name (str): The name of the obstacle.
            size (float): The size of the obstacle (pixel radius).
            position (tuple): The position of the obstacle.
            rgb (tuple): The color of the obstacle in RGB format.

        """
        super().__init__(name, size, position=(0,0), rgb=rgb)
        self.name = str(name) or "Rock"
        self.size = int(size) or 1 # pixel radius
        self.rgb = tuple(rgb) or (100, 100, 100)
        self.category = 1

class World():
    """
    World Class.

    This class represents the simulation world, including objects and simulation parameters.

    Attributes:
        name (str): The name of the world.
        radius (int): The radius of the world.
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
            Spawn wall obstacles along the perimeter of the world.

        spawn_obstacles(self, num_obstacles, obstacle_params):
            Spawn a specified number of obstacles with given parameters.

        random_point_in_world(self):
            Generate a random point within the boundaries of the world.

        simulate_step(self):
            Simulate a single step of the world, updating organoids, food, and handling collisions.

    """

    def __init__(self, name, radius, doomsday_ticker, obstacle_ratio, abundance):
        """
        Initialize the World object with simulation parameters.

        Args:
            name (str): The name of the world.
            radius (int): The radius of the world.
            doomsday_ticker (int): Maximum number of simulation iterations.
            obstacle_ratio (float): Ratio of obstacles to the world size.
            abundance (float): Abundance factor for food generation.

        """
        self.name = str(name) or "Midgard"
        self.radius = int(radius) or 100
        self.abundance = float(abundance) or 1.00
        self.obstacle_ratio = float(obstacle_ratio) or 0.01
        self.doomsday_ticker = int(doomsday_ticker) or 100000
        self.organoids = []
        self.food = []
        self.obstacles = []

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
        scores = list()
        for o in self.organoids:
            scores.append(
                {
                    "id":f"{o.id}",
                    "score":f"{o.score}",
                    "name":f"{o.name}",
                    "size":f"{o.size}",
                    "calories":f"{o.calories}",
                    "max_calories":f"{o.calorie_limit}",
                    "children":f"{o.children}"
                }
            )
        scorecard = pd.DataFrame.from_records(scores)
        scorecard.to_csv(f"organoid_sim_{str(int(datetime.now().timestamp()))}.csv")
        return scorecard
    
    def run_simulation(self):
        """
        Run the simulation, updating the visualization and organoid behaviors.

        This method initializes the visualization and runs the simulation loop,
        updating the state of the world and organoids.

        """
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
            """
            Update the logical plot and visualize.
            """
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
        return self.generate_score_card()
        
        

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
            if food.position[0] >= self.radius*2 - food.size or food.position[0] <= food.size or \
               food.position[1] >= self.radius*2 - food.size or food.position[1] <= food.size:
                continue
            self.food.append(food)
            food_added += 1

    def spawn_walls(self):
        """
        Spawn wall obstacles along the perimeter of the world.

        This method creates wall obstacles to enclose the simulation world.

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
            if obstacle.position[0] >= self.radius*2 - obstacle.size or obstacle.position[0] <= obstacle.size or \
               obstacle.position[1] >= self.radius*2 - obstacle.size or obstacle.position[1] <= obstacle.size:
                continue

            self.obstacles.append(obstacle)
            obstacles_added += 1
            
    def random_point_in_world(self):
        """
        Generate a random point within the boundaries of the world.

        Returns:
            tuple: A tuple representing the coordinates of a random point within the world boundaries.

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
        self.organoids = [org for org in self.organoids if org is not None and org.alive]
        for organoid in self.organoids:
            organoid.update(self.food, self.organoids, self.obstacles)
            if not organoid.is_alive():
                self.organoids.remove(organoid)

        self.handle_collisions()
    
if __name__ == "__main__":
    # Create the world
    world = World(name="Midgard", radius=100, doomsday_ticker=500, obstacle_ratio=0.01, abundance=100.00)

    # Define parameters for organoids and food
    organoid_params = {
        "name": "Silly Blob", 
        "lifespan": 60, 
        "size": 5, 
        "calories": 50, 
        "calorie_limit": 50, 
        "metabolism": 0.01, 
        "rgb": (255, 10, 10), 
        "position": (0, 0), 
        "smart": False
    }
    evolved_params = organoid_params.copy()
    evolved_params["smart"] = True
    evolved_params["name"] = "Brainy Blob"
    food_params = {
        "name": "Algae", 
        "size": 2.5, 
        "calories": 200, 
        "rgb": (10, 255, 10)
    }
    obstacle_params = {
        "name": "Rock", 
        "size": 7, 
        "rgb": (100, 100, 100)
    }

    # Spawn initial organoids and food

    world.spawn_food(num_food=100, food_params=food_params)
    world.spawn_obstacles(5, obstacle_params=obstacle_params)
    world.spawn_walls()
    world.spawn_organoids(num_organoids=2, organoid_params=organoid_params)
    world.spawn_organoids(num_organoids=2, organoid_params=evolved_params)

    food_spawn_interval = 5  # Adjust the interval (in seconds) for continuous food spawning
    food_spawner_thread = threading.Thread(target=world.spawn_continuous_food, args=(food_spawn_interval,))
    food_spawner_thread.start()

    # Kick off visualization
    world.run_simulation()
    food_spawner_thread.join()