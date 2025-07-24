import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Parameters
BOARD_SIZE = 10
WIN_LENGTH = 5
TOTAL_EPISODES = 50000
ITERATIONS = 10
EPISODES_PER_ITER = TOTAL_EPISODES // ITERATIONS
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9997
MIN_EPSILON = 0.1
SHOW_EVERY = 1000

PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

q_table_X = {}
q_table_O = {}

COLOR_MAP = {
    'X': (0, 0, 255),     # Red
    'O': (255, 0, 0),     # Blue
    ' ': (255, 255, 255)  # White
}

CELL_SIZE = 40
REWARD_SHAPING = True

def empty_cells(board):
    return [i for i, v in enumerate(board) if v == EMPTY]

def to_2D(index):
    return index // BOARD_SIZE, index % BOARD_SIZE

def to_index(row, col):
    return row * BOARD_SIZE + col

def extract_line(grid, r, c, dr, dc):
    return [grid[r + i * dr][c + i * dc] for i in range(WIN_LENGTH)]

def check_line(line, player):
    return sum(1 for x in line if x == player)

def count_consecutive(board, player):
    grid = [board[i * BOARD_SIZE:(i + 1) * BOARD_SIZE] for i in range(BOARD_SIZE)]
    three, four = 0, 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                if r + (WIN_LENGTH-1)*dr < BOARD_SIZE and 0 <= c + (WIN_LENGTH-1)*dc < BOARD_SIZE:
                    line = extract_line(grid, r, c, dr, dc)
                    if EMPTY not in line:
                        continue
                    cnt = check_line(line, player)
                    if cnt == 3:
                        three += 1
                    elif cnt == 4:
                        four += 1
    return three, four

def check_five(board, player):
    grid = [board[i * BOARD_SIZE:(i + 1) * BOARD_SIZE] for i in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                if r + (WIN_LENGTH-1)*dr < BOARD_SIZE and 0 <= c + (WIN_LENGTH-1)*dc < BOARD_SIZE:
                    line = extract_line(grid, r, c, dr, dc)
                    if all(x == player for x in line):
                        return True, [(r+i*dr, c+i*dc) for i in range(WIN_LENGTH)]
    return False, []

def winner(board):
    x_win, _ = check_five(board, PLAYER_X)
    o_win, _ = check_five(board, PLAYER_O)
    if x_win:
        return PLAYER_X
    elif o_win:
        return PLAYER_O
    elif EMPTY not in board:
        return "draw"
    return None

def choose_action(q_table, state, valid_moves, epsilon):
    if state not in q_table:
        q_table[state] = [random.uniform(-1, 1) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    if random.random() < epsilon:
        return random.choice(valid_moves)
    return max(valid_moves, key=lambda a: q_table[state][a])

def update_q(q_table, state, action, reward, new_state, done):
    if state not in q_table:
        q_table[state] = [random.uniform(-1, 1) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    if new_state not in q_table:
        q_table[new_state] = [random.uniform(-1, 1) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    max_future_q = 0 if done else max(q_table[new_state])
    current_q = q_table[state][action]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[state][action] = new_q

def show_board(board, win_line=None, delay=1):
    img_size = BOARD_SIZE * CELL_SIZE
    board_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    for idx, val in enumerate(board):
        row, col = to_2D(idx)
        color = COLOR_MAP[val]
        center = (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2)
        radius = CELL_SIZE // 3
        cv2.circle(board_img, center, radius, color, -1)
    for i in range(BOARD_SIZE + 1):
        cv2.line(board_img, (0, i * CELL_SIZE), (img_size, i * CELL_SIZE), (0, 0, 0), 1)
        cv2.line(board_img, (i * CELL_SIZE, 0), (i * CELL_SIZE, img_size), (0, 0, 0), 1)
    if win_line:
        p1 = (win_line[0][1] * CELL_SIZE + CELL_SIZE // 2, win_line[0][0] * CELL_SIZE + CELL_SIZE // 2)
        p2 = (win_line[-1][1] * CELL_SIZE + CELL_SIZE // 2, win_line[-1][0] * CELL_SIZE + CELL_SIZE // 2)
        cv2.line(board_img, p1, p2, (0, 255, 0), 2)
    cv2.imshow("Tic Tac Toe", board_img)
    cv2.waitKey(delay)

# Main training loop
all_rewards_X, all_rewards_O = [], []
EPSILON = EPSILON_START

for iteration in range(ITERATIONS):
    episode_rewards_X = []
    episode_rewards_O = []

    for episode in range(EPISODES_PER_ITER):
        board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
        turn = random.choice([PLAYER_X, PLAYER_O])
        transitions = []
        win_line = None
        show = (episode % SHOW_EVERY == 0)

        if show:
            print(f"[Iter {iteration}] Episode {episode} | EPSILON: {EPSILON:.3f}")
            if episode >= SHOW_EVERY:
                print("  X Avg:", round(np.mean(episode_rewards_X[-SHOW_EVERY:]), 3),
                      "| O Avg:", round(np.mean(episode_rewards_O[-SHOW_EVERY:]), 3))

        done = False
        while not done:
            state = tuple(board)
            valid = empty_cells(board)
            player = turn
            q_table = q_table_X if player == 'X' else q_table_O
            action = choose_action(q_table, state, valid, EPSILON)
            board[action] = player
            transitions.append((player, state, action))
            result = winner(board)
            done = result is not None
            if show:
                _, win_line = check_five(board, player)
                show_board(board, win_line if result == player else None, delay=50)
            turn = PLAYER_O if turn == PLAYER_X else PLAYER_X

        for player, state, action in transitions:
            reward = 0
            if result == 'draw': reward = 0.5
            elif result == player: reward = 1.0
            else: reward = -1.0

            if REWARD_SHAPING:
                mine_3, mine_4 = count_consecutive(state, player)
                opp = PLAYER_O if player == PLAYER_X else PLAYER_X
                opp_3, opp_4 = count_consecutive(state, opp)
                reward += 0.2 * mine_3 + 0.5 * mine_4
                reward += 0.1 * opp_3 + 0.4 * opp_4

            if iteration == 0 or (iteration % 2 == 1 and player == 'X') or (iteration % 2 == 0 and player == 'O'):
                update_q(q_table_X if player == 'X' else q_table_O,
                         state, action, reward, tuple(board), True)

        episode_rewards_X.append(1 if result == 'X' else (-1 if result == 'O' else 0.5))
        episode_rewards_O.append(1 if result == 'O' else (-1 if result == 'X' else 0.5))
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    all_rewards_X.extend(episode_rewards_X)
    all_rewards_O.extend(episode_rewards_O)

print("\nShowing 5 games between final trained agents...\n")

for game_num in range(5):
    print(f"Game {game_num + 1}")
    board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
    turn = random.choice([PLAYER_X, PLAYER_O])
    done = False
    _, win_line = check_five(board, PLAYER_X)  # Dummy init

    while not done:
        state = tuple(board)
        valid = empty_cells(board)
        player = turn
        q_table = q_table_X if player == PLAYER_X else q_table_O

        if state not in q_table:
            action = random.choice(valid)
        else:
            action = max(valid, key=lambda a: q_table[state][a])

        board[action] = player
        result = winner(board)
        done = result is not None
        _, win_line = check_five(board, player)
        show_board(board, win_line if result == player else None, delay=200)

        turn = PLAYER_O if turn == PLAYER_X else PLAYER_X

    print(f"Result: {result}\n")
    cv2.waitKey(1000)  # Pause before next game

#Save Q tables
pickle.dump(q_table_X, open("q_table_X_final.pickle", "wb"))
pickle.dump(q_table_O, open("q_table_O_final.pickle", "wb"))

# Plot results
plt.plot(np.convolve(all_rewards_X, np.ones(SHOW_EVERY)/SHOW_EVERY, mode='valid'), label='X Agent')
plt.plot(np.convolve(all_rewards_O, np.ones(SHOW_EVERY)/SHOW_EVERY, mode='valid'), label='O Agent')
plt.title("Learning Progress with Alternating Training")
plt.xlabel("Episode Group")
plt.ylabel(f"Avg Reward per {SHOW_EVERY}")
plt.legend()
plt.show()

cv2.destroyAllWindows()


