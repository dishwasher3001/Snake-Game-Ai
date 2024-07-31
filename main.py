# importing libraries
import pygame
import time
import random
import torch
import numpy as np
import os
from nn import Agent  # Import the Agent class from nn.py

snake_speed = 1000

# Window size
window_x = 720
window_y = 480

# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# Initialising pygame
pygame.init()

# Initialise game window
pygame.display.set_caption('GeeksforGeeks Snakes')
game_window = pygame.display.set_mode((window_x, window_y))

# FPS (frames per second) controller
fps = pygame.time.Clock()

# Define action space and state space dimensions
state_dim = 20  # updated state dimensions based on the final state representation
action_dim = 4  # four actions: UP, DOWN, LEFT, RIGHT

update_target_frequency = 1000  # every 1000 steps update target network

# Initialize DQN agent
agent = Agent(state_dim, action_dim)

# Initialize reward and death logs
reward_log = []
last_death = ""

# Reset game function
def reset_game():
    global snake_position, snake_body, fruit_position, direction, change_to, score, steps_done, done
    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
    direction = 'RIGHT'
    change_to = 'RIGHT'
    score = 0
    steps_done = 0
    done = False

# Initialize game state
reset_game()

def get_state():
    global snake_position, fruit_position, direction, change_to
    head_x, head_y = snake_position[0], snake_position[1]
    fruit_x, fruit_y = fruit_position[0], fruit_position[1]

    # Calculate distances to the walls
    distance_to_left_wall = head_x
    distance_to_right_wall = window_x - head_x
    distance_to_top_wall = head_y
    distance_to_bottom_wall = window_y - head_y

    state = [
        # Add the snake's head position
        head_x, head_y,
        # Add the fruit's position
        fruit_x, fruit_y,
        # Add distances to walls
        distance_to_left_wall, distance_to_right_wall,
        distance_to_top_wall, distance_to_bottom_wall,
        # Add binary indicators for immediate obstacles
        int([head_x - 10, head_y] in snake_body or head_x - 10 < 0), # left
        int([head_x + 10, head_y] in snake_body or head_x + 10 > window_x), # right
        int([head_x, head_y - 10] in snake_body or head_y - 10 < 0), # up
        int([head_x, head_y + 10] in snake_body or head_y + 10 > window_y), # down
        # Add binary direction indicators
        direction == 'UP', direction == 'DOWN',
        direction == 'LEFT', direction == 'RIGHT',
        change_to == 'UP', change_to == 'DOWN',
        change_to == 'LEFT', change_to == 'RIGHT'
    ]

    return np.array(state, dtype=np.float32)

# displaying Score function
def show_score(color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Game over function
def game_over():
    model_path = "best_snake_model.pth"
    agent.save(model_path)

    my_font = pygame.font.SysFont('times new roman', 50)
    game_over_surface = my_font.render('Your Score is : ' + str(score), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (window_x / 2, window_y / 4)
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()

    time.sleep(2)
    pygame.quit()
    os.system('python main.py')
    quit()

# Main Function
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over()

    state = get_state()
    action = agent.select_action(state)

    # Mapping action to direction
    if action == 0:
        change_to = 'UP'
    elif action == 1:
        change_to = 'DOWN'
    elif action == 2:
        change_to = 'LEFT'
    elif action == 3:
        change_to = 'RIGHT'

    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    # Calculate previous distance
    previous_distance = calculate_distance(snake_position, fruit_position)

    # Move the snake
    if direction == 'UP':
        snake_position[1] -= 10
    if direction == 'DOWN':
        snake_position[1] += 10
    if direction == 'LEFT':
        snake_position[0] -= 10
    if direction == 'RIGHT':
        snake_position[0] += 10

    # Calculate new distance
    new_distance = calculate_distance(snake_position, fruit_position)

    # Snake body growing mechanism
    snake_body.insert(0, list(snake_position))
    if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
        score += 10
        reward = 10  # Reward for getting the fruit
        fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
    else:
        snake_body.pop()
        reward = -0.1  # Slight negative reward for each step to encourage efficiency

    # Calculate the reward based on distance and direction
    distance_diff = previous_distance - new_distance
    if distance_diff > 0:
        reward += 1.0  # Greater reward for getting closer
    else:
        reward -= 2.0  # Heavier penalty for moving away

    # Additional reward for proximity to fruit
    max_distance = np.sqrt(window_x**2 + window_y**2)
    normalized_distance = new_distance / max_distance
    reward += 1.0 * (1 - normalized_distance)  # Reward closer to the fruit given more weight

    # Small survival reward
    reward += 0.1 * (steps_done / 100)

    game_window.fill(black)

    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))

    # Game Over conditions
    done = False
    cause_of_death = ""
    if snake_position[0] < 0 or snake_position[0] > window_x - 10:
        done = True
        reward = -10
        cause_of_death = "hit a wall"
    if snake_position[1] < 0 or snake_position[1] > window_y - 10:
        done = True
        reward = -10
        cause_of_death = "hit a wall"

    for block in snake_body[1:]:
        if snake_position[0] == block[0] and snake_position[1] == block[1]:
            done = True
            reward = -100
            cause_of_death = "hit itself"

    if done:
        last_death = f"Game over: The snake {cause_of_death}."

    # Update reward log
    reward_log.append(reward)
    if len(reward_log) > 10:
        reward_log.pop(0)
    
    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print the latest 10 rewards
    print("Last 10 rewards:")
    for r in reward_log:
        print(r)

    # Print the last death message
    if last_death:
        print(last_death)

    # Storing transition
    next_state = get_state()
    agent.store_transition(state, action, reward, next_state, done)

    # Perform optimization step
    agent.update_policy()
    steps_done += 1

    if steps_done % update_target_frequency == 0:
        agent.update_target_network()

    if done:
        reset_game()

    # Display score
    show_score(white, 'times new roman', 20)
    pygame.display.update()
    fps.tick(snake_speed)
