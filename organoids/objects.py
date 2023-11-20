"""
Objects
=======

Contains obejcts and methods for populating worldspace.

Classes:
    BaseObject: defines a base object with common attributes and methods for other objects
    Organoid: defines organoid objects, which are agents in the simulation
    Food: defines food objects in the simulation which are consumed by Organoids
    Obstacle: defines obstacle objects in the simulation that must be avoided by Organoids
    
Author: Bradley Reimers
Date: 11/19/2023
License: GPL 3
"""
import random
from uuid import uuid4
import numpy as np
from .brains import MODEL_SELECTOR


class BaseObject:
    """
    Base Object Class.

    This class defines a base object with common attributes and methods for other objects
        in the simulation.

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

    def __init__(self, name="obj", size=1, position=(0, 0), rgb=(0, 0, 0), category=0):
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
        self.category = category

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

        update(food, organoids, obstacles, step):
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

    def __init__(
        self,
        name,
        lifespan,
        size,
        calories,
        calorie_limit,
        position,
        metabolism,
        rgb,
        cooldown_duration,
        smart,
        modeltype,
        hidden_layers,
    ):
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
        super().__init__(name, size, position, rgb, 3)
        self.lifespan = 0
        self.max_lifespan = int(lifespan) or 100
        self.calories = int(calories) or 100  # energy
        self.calorie_limit = int(calorie_limit) or 100
        self.metabolism = float(metabolism) or 0.05
        self.alive = True
        self.reproduction_cooldown = 0
        self.cooldown_duration = int(cooldown_duration) or 500
        self.mutation_rate = 0.2
        self.vision_range = 10
        self.vision_factor = 2.5
        self.children = 0
        self.score = 0
        self.last_score = self.score
        self.smart = smart
        self.modeltype = modeltype
        self.hidden_layers = hidden_layers
        if self.smart:
            if self.modeltype in MODEL_SELECTOR.keys():
                self.brain = MODEL_SELECTOR[self.modeltype](
                    hidden_sizes=self.hidden_layers
                )
            else:
                self.brain = MODEL_SELECTOR["NN"](hidden_sizes=self.hidden_layers)
        else:
            self.brain = None

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
                        "name": object.name,
                    }
                )
        return new_obj_list

    def update(self, food, organoids, obstacles, step):
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
            self.train_neural_network(delta_score, objects, step)
        else:
            self.move(random.uniform(-1.00, 1.00), random.uniform(-1.00, 1.00))
            self.metabolize()
            self.update_score()
        self.lifespan += 1
        if self.calories <= 0:
            self.alive = False
        if self.lifespan >= self.max_lifespan:
            self.alive = False

    def train_neural_network(self, delta_score, objects, step):
        """
        Train the neural network based on experiences.

        Args:
            delta_score (int): The change in the organoid's score.
            objects (list): List of objects within the organoid's vision range.
            step (int): Time vector

        """
        # Determine the maximum number of objects to consider
        max_objects = 10  # Adjust this based on your requirements

        # Create arrays to hold the object representations and their counts
        object_representations = np.zeros(
            (max_objects, 4), dtype=float
        )  # Assuming 4 attributes per object
        object_counts = min(len(objects), max_objects)

        # Flatten attributes for each object
        for i in range(object_counts):
            obj = objects[i]
            object_representations[i] = [
                obj["category"],
                obj["size"],
                obj["x-position"],
                obj["y-position"],
            ]

        # Fill any remaining slots with zeros
        for i in range(object_counts, max_objects):
            object_representations[i] = [0, 0, 0, 0]

        # Construct the state and action arrays
        state = np.array(
            [
                self.calories,
                self.size,
                self.position[0],
                self.position[1],
                delta_score,
                *object_representations.flatten(),
                self.reproduction_cooldown,
                step,
            ]
        ).reshape(1, -1)

        action = self.brain.choose_action(state)

        # Calculate new position based on action
        x_off, y_off = action
        self.move(x_off, y_off)
        self.metabolize()
        self.update_score()

        # Calculate reward (using delta_score as reward)
        reward = self.score - self.last_score

        new_state = np.array(
            [
                self.calories,
                self.size,
                self.position[0],
                self.position[1],
                reward,
                *object_representations.flatten(),
                self.reproduction_cooldown,
                step,
            ]
        ).reshape(1, -1)

        self.brain.train(state, action, reward, new_state)

    def get_info(self):
        """
        Get information about the organoid.

        Returns:
            str: Information about the organoid, including its name and current state.

        """
        return f"Name: {self.name}\
                \nSize: {self.size:.2f}\
                \nCalories: {int(self.calories)}/{self.calorie_limit}\
                \nScore: {self.score}"

    def metabolize(self):
        """
        Perform metabolism and potentially grow.

        """
        # This checks if the organoid has excess calories, if so, grow and increase
        #  the calorie limit.
        if self.calories > self.calorie_limit:
            diff = self.calories - self.calorie_limit
            self.calories = self.calorie_limit  # Cap the calories at the limit
            self.size += (
                diff * 0.001
            )  # Increase the organoid size by a fraction of the excess calories
        self.vision_range = self.vision_factor * self.size
        self.calories -= self.metabolism * (self.size / 4)

    def mutate_layers(self):
        """
        Mutate a list of integers based on the mutation rate.

        Args:
            layer_list (list): A list of integers to be mutated.

        Returns:
            list: The mutated list of integers.
        """
        mutated_list = self.hidden_layers[:]  # Create a copy of the original list

        # Iterate through the list and apply mutations
        for i in range(len(mutated_list)):
            if random.random() < self.mutation_rate:
                # There is a mutation_rate chance of modifying this value
                if random.random() < 0.5:
                    # 50% chance of increasing the value
                    mutation_amount = (
                        self.mutation_rate * mutated_list[i]
                    )  # Increase by mutation rate percentage
                else:
                    # 50% chance of decreasing the value
                    mutation_amount = 0 - (
                        self.mutation_rate * mutated_list[i]
                    )  # Decrease by mutation rate percentage
                mutated_list[i] = round(mutated_list[i] + mutation_amount)
        if random.random() < self.mutation_rate:
            # There is a mutation_rate chance of dropping an integer from the list
            if len(mutated_list) > 2:
                index = random.randint(0, len(mutated_list) - 1)
                if random.random() < 0.5:
                    mutated_list.pop(index)
                else:
                    mutated_list.insert(index, mutated_list[index])
            else:
                mutated_list.append(mutated_list[0])
        return mutated_list

    def split_organoid(self):
        """
        Split the organoid to create offspring.

        """
        if self.reproduction_cooldown == 0:
            # Create a new organoid next to the current one with base parameters
            offset_x = random.uniform(-self.size, self.size)
            offset_y = random.uniform(-self.size, self.size)
            new_position = (self.position[0] + offset_x, self.position[1] + offset_y)
            new_organoid = Organoid(
                name=self.name,
                lifespan=0,
                size=(self.size / 2),
                calories=(self.calorie_limit / 2),
                calorie_limit=(self.calorie_limit / 2),
                position=new_position,
                metabolism=self.metabolism,
                rgb=self.rgb,
                smart=self.smart,
                modeltype=self.modeltype,
                hidden_layers=self.mutate_layers(),
                cooldown_duration=self.cooldown_duration,
            )
            # Mutate the parameters within 10% of the parent's values (except for size)
            new_organoid.max_lifespan = self.max_lifespan
            parameter_names = ["max_lifespan", "calorie_limit", "metabolism", "rgb"]
            for param_name in parameter_names:
                parent_value = getattr(self, param_name)
                if param_name == "rgb":
                    new_organoid.rgb = (
                        random.uniform(
                            parent_value[0] - (parent_value[0] * self.mutation_rate),
                            parent_value[0] + (parent_value[0] * self.mutation_rate),
                        ),
                        random.uniform(
                            parent_value[1] - (parent_value[1] * self.mutation_rate),
                            parent_value[1] + (parent_value[1] * self.mutation_rate),
                        ),
                        random.uniform(
                            parent_value[2] - (parent_value[2] * self.mutation_rate),
                            parent_value[2] + (parent_value[2] * self.mutation_rate),
                        ),
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
                return new_organoid

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
        new_x = self.position[0] + x_off * self.step_size
        new_y = self.position[1] + y_off * self.step_size
        self.position = (new_x, new_y)

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
        return np.sqrt(
            (self.position[0] - other.position[0]) ** 2
            + (self.position[1] - other.position[1]) ** 2
        )

    def update_score(self):
        """
        Update the organoid's score.

        """
        self.last_score = self.score
        self.score = int(20 * self.children + 5 * self.calories + 10 * self.size)

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
        super().__init__(name, size, (0, 0), rgb, 2)
        self.calories = int(calories) or 1  # energy

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
        super().__init__(name, size, (0, 0), rgb, 1)
