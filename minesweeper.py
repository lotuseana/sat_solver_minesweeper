import sys
import time
import random as rnd
import cpmpy as cp
import numpy as np
from tkinter import *
from tkinter import ttk
from pprint import pprint


# -- FUNCTIONS -- #

#reveal current cell
def reveal_cell(position, constraints):
    r, c = position
    if cur_board[r][c] != -1:
        return
    #mini dungeon
    #canvas.create_text(c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2, text=f"{final_board[r][c]}", font=("Arial", 24), fill="black")
    cur_board[r][c] = answer_board[r][c]
    global cur_cell
    cur_cell = (r, c)
    global num_revealed_cells
    num_revealed_cells += 1

    #print(f"revealed cell: ({c}, {r})")

    #joever
    if cur_board[r][c] == -2 or num_revealed_cells == ROWS * COLS - MINES:
        print_board(cur_board, cur_cell)
        print("[WIN]" if num_revealed_cells == ROWS * COLS - MINES else "[LOSS]")
        end_time = time.time()
        print(f"total elapsed: {round(end_time - start_time, 2)} seconds")
        global game_over
        game_over = True
        sys.exit()
    
    #current cell has no mine
    #solver.add(all_unknowns[r][c] == 0)
    constraints.append(all_unknowns[r][c] == 0)


    #trivial minecount openings:
    check_minecount(r, c)

    #lin combo of current cell
    if cur_board[r][c] != 0:
        lin_combo(r, c, constraints)
    else: # recursively flood-fill
        for i in range(max(0, r - 1), min(ROWS, r + 2)):
            for j in range(max(0, c - 1), min(COLS, c + 2)):
                if (i, j) != (r, c) and cur_board[i][j] == -1:
                    reveal_cell((i, j), constraints)

#set up current cell as lin combo of surrounding cells
def lin_combo(i, j, constraints):
    if cur_board[i][j] != -1:
            X = []
            for r in range(max(0, i-1), min(ROWS, i+2)):
                for c in range(max(0, j-1), min(COLS, j+2)):
                    if cur_board[r][c] in {-1,-3}:
                        X.append(all_unknowns[r][c])
            rhs = cur_board[i][j]
            #solver.add(cp.sum(X) == rhs)
            constraints.append(cp.sum(X) == rhs)

#check for trivial openings around current cell
def check_minecount(r,c):
  
    num_surrounding_mines = cur_board[r][c]
    num_flagged_mines = 0
    for i in range(max(0, r - 1), min(ROWS, r + 2)):
        for j in range(max(0, c - 1), min(COLS, c + 2)):
            if cur_board[i][j] == -3:
                num_flagged_mines += 1
    if num_surrounding_mines == num_flagged_mines:
        for i in range(max(0, r - 1), min(ROWS, r + 2)):
            for j in range(max(0, c - 1), min(COLS, c + 2)):
                if cur_board[i][j] == -1:
                    reveal_cell((i, j), constraints)

#collect all solutions
def collect():
    val = np.array(all_unknowns.value(), copy=True)
    solutions.append(val)

#calculate probabilities
def calculate_probabilities(probabilities):
    #KEY:
    # +1 for each TRUE in sols
    # no change for each FALSE in sols
    # set to -1 for NONE in any sol
    # set to -2 if revealed in cur_board
    for sol in solutions:
        for i in range(ROWS):
            for j in range(COLS):
                if (probabilities[i][j] < 0):
                    continue
                if sol[i][j]:
                    probabilities[i][j] += 1
                elif sol[i][j] == None:
                    probabilities[i][j] = -1
                elif cur_board[i][j] not in {-1, -3}:
                    probabilities[i][j] = -2

    cells_left = ROWS * COLS - num_revealed_cells - num_found_mines
    mines_left = MINES - num_found_mines
    minecount_prob = round(mines_left / cells_left,2)
    print(f"minecount probability: {minecount_prob}; {mines_left} in {cells_left} cells")
    for i in range(ROWS):
        for j in range(COLS):
            if probabilities[i][j] > 0:
                probabilities[i][j] = round(probabilities[i][j]/len(solutions),2)
            elif probabilities[i][j] == -1 and i in {0, ROWS - 1} and j in {0, COLS - 1}:
                probabilities[i][j] = minecount_prob

#decide which cell to reveal next
def next_cell(probabilities, constraints):
    hashed_probs = {}

    #hash cells by probability
    for i in range(ROWS):
        for j in range(COLS):
            if probabilities[i][j] not in hashed_probs:
                hashed_probs[probabilities[i][j]] = []
            hashed_probs[probabilities[i][j]].append((i,j))

    #flag definite mines
    if 1.0 in hashed_probs:
        for cell in hashed_probs[1]:
            r, c = cell
            if cur_board[r][c] != -3:
                cur_board[r][c] = -3
                global num_found_mines
                num_found_mines += 1
                #print(f"flagged mine @: ({c}, {r})")
                constraints.append(all_unknowns[r][c] == 1)
                #check trivial openings around flagged mine
                for i in range(max(0, r - 1), min(ROWS, r + 2)):
                    for j in range(max(0, c - 1), min(COLS, c + 2)):
                        if cur_board[i][j] not in {-1, -3}:
                            check_minecount(i, j)
        del hashed_probs[1]
    
    
    #choose random cell from lowest prob
    decided = False
    while not decided:
        min_prob = min(hashed_probs.keys())
        #random stragglers
        if min_prob < 0:
            del hashed_probs[min_prob]
            continue
        #open definite safe cells
        if 0.0 in hashed_probs:
            for cell in hashed_probs[0.0]:
                reveal_cell(cell, constraints)
            del hashed_probs[0.0]
            continue
        cell = rnd.choice(hashed_probs[min_prob])
        decided = True
        return cell

#print board nicely
def print_board(board, cell):
    for i in range (len(board)):
        row_str = ""
        for j in range (len(board[i])):
            char = "["
            on_current_cell = False if cell is None else (i == cell[0] and j == cell[1]) 
            match on_current_cell:
                case True: char += "["
                case False: char += " "
            match board[i][j]:
                case -1: char += " "
                case -2: char += "*"
                case -3: char += "^"
                case _: char += f"{board[i][j]}"
            match on_current_cell:
                case True: char += "]"
                case False: char += " "
            row_str += char + "]"
        print(row_str)


# -- PROG START:D -- #

#vars
MINES = 99
ROWS = 16
COLS = 30
answer_board = [[0 for j in range(COLS)] for i in range(ROWS)]
cur_board = [[-1 for j in range(COLS)] for i in range(ROWS)]
game_over = False
num_revealed_cells = 0
num_found_mines = 0

#KEY FOR BOARDS:
# 0-8: surrounging mines
# -1: no info
# -2: mine
# -3: flag

#initializing unknowns
all_unknowns = cp.boolvar(shape=(ROWS, COLS), name="x")


#placing mines
for mine in range(MINES):
    r = rnd.randint(0, ROWS - 1)
    c = rnd.randint(0, COLS - 1)
    while answer_board[r][c] == -2 or (r == 0 and c == 0):
        r = rnd.randint(0, ROWS - 1)
        c = rnd.randint(0, COLS - 1)
    answer_board[r][c] = -2

#setting up board
for i in range(ROWS):
    for j in range(COLS):
        if answer_board[i][j] == -2:
            continue
        mine_count = 0
        for r in range(max(0, i-1), min(ROWS, i+2)):
            for c in range(max(0, j-1), min(COLS, j+2)):
                if answer_board[r][c] == -2:
                    mine_count += 1
        answer_board[i][j] = mine_count

print("ANSWER BOARD:")
print_board(answer_board, None)
print("\n\n")

#start in top left corner
cur_cell = (0,0)
start_time = time.time()

constraints = []

#first move
reveal_cell(cur_cell, constraints)
print_board(cur_board, cur_cell)


while not game_over:
    reveal_cell(cur_cell, constraints)
    print_board(cur_board, cur_cell)
    #check trivial openings around current cell
    #check_minecount(cur_cell)

    print("# revealed:" + str(num_revealed_cells))
    print("left to reveal:" + str(ROWS * COLS - num_revealed_cells - MINES))

    model = cp.Model(constraints)
    #collect solutions
    solutions = []

    model.solveAll(display=collect)
    print(f"stored {len(solutions)} solutions.\n")

    probabilities = [[0 for j in range(COLS)] for i in range(ROWS)]
    calculate_probabilities(probabilities)
    
    """"
    print("PROBABILITIES:")
    for i in range(ROWS):
        row_str = ""
        for j in range(COLS):
            char = "["
            match probabilities[i][j]:
                case -2: char += "   "
                case -1: char += " ? "
                case 0: char += " 0 "
                case 1: char += "1.0"
                case _: char += f".{int(probabilities[i][j]*100)}"
            row_str += char + "]"
        print(row_str)
    """

    cur_cell = next_cell(probabilities, constraints)
    print(f"CURRENT BOARD: (out of {len(solutions)} solutions; current cell mine probability = {probabilities[cur_cell[0]][cur_cell[1]]}) [time: {round(time.time() - start_time, 2)}s]")
    #print(f"next cell to reveal: ({cur_cell[1]}, {cur_cell[0]})\n\n")


