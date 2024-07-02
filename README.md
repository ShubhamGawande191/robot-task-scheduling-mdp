Description

This custom environment simulates a multi-agent taxi problem in a grid world. There are six designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), B(lue), C(yan), and M(agenta). Each taxi starts off at a random square, and each passenger is at a random designated location. The taxis must drive to the passengers’ locations, pick them up, drive to their destinations, and drop them off. The episode ends once all passengers have been delivered to their destinations.

Map
The illustration below shows the grid world with the six designated locations and the four walls. A filled cell represents a wall, and each cell represents a road. The taxi cannot pass through the walls.
+---------+
|R: | : :G|
| :C| : :M|
| : : : : |
| | : | : |
|Y| : |B: |
+---------+

Actions
Each agent (taxi) can take one of six discrete deterministic actions:

0: Move south
1: Move north
2: Move east
3: Move west
4: Pickup passenger
5: Drop off passenger

Observations

There are 46,328,400 discrete states since there are:

625 possible positions for the 2 taxis (25 positions each).
343 possible locations for the 3 passengers (7 locations each, including in the taxi).
216 possible destination configurations for the 3 passengers (6 destinations each).
Each state space is represented by the tuple: (taxi1_row, taxi1_col, taxi2_row, taxi2_col, passenger1_location, passenger2_location, passenger3_location, destination1, destination2, destination3).

Note that the number of actually reachable states will be less due to constraints and the episode ending conditions.

An observation is an integer that encodes the corresponding state. The state tuple can then be decoded with the decode method.

The observation space is a dictionary with two keys:

agents: A box with shape (num_agents, 3) representing the position and task status of each agent (x, y, carrying).
tasks: A multidiscrete space with shape (num_tasks, 4) representing the pickup and drop-off locations of each task (pickup_x, pickup_y, dropoff_x, dropoff_y).
Each state space is represented by the tuple: (taxi_row, taxi_col, passenger_location, destination).

Rewards
-1 per step unless another reward is triggered.
+20 for delivering a passenger.
-10 for executing “pickup” and “drop-off” actions illegally.