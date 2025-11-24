import sys
import time
import random as rnd
import numpy as np
from pysat.formula import CNF
from pysat.card import CardEnc
from pysat.solvers import Solver

# -------------------------
# Helper functions

def neighbors(r, c):
    return [(nr, nc) for nr in range(max(0,r-1), min(ROWS,r+2))
                     for nc in range(max(0,c-1), min(COLS,c+2))
                     if (nr,nc)!=(r,c)]

def print_board(board, cell=None):
    for i in range(ROWS):
        row_str = ""
        for j in range(COLS):
            char = "["
            on_current_cell = False if cell is None else (i == cell[0] and j == cell[1])
            if on_current_cell: char += "["
            else: char += " "
            val = board[i][j]
            if val == -1: char += " "
            elif val == -2: char += "*"
            elif val == -3: char += "^"
            else: char += str(val)
            if on_current_cell: char += "]"
            else: char += " "
            row_str += char + "]"
        print(row_str)

# -------------------------
# Minesweeper functions

def reveal_cell(position, solver, constraints):
    r, c = position
    if cur_board[r][c] != -1:
        return
    cur_board[r][c] = answer_board[r][c]
    global cur_cell, num_revealed_cells, game_over
    cur_cell = (r, c)
    num_revealed_cells += 1

    # check win/loss
    if cur_board[r][c] == -2 or num_revealed_cells == ROWS*COLS-MINES:
        print_board(cur_board, cur_cell)
        print("[WIN]" if num_revealed_cells == ROWS*COLS-MINES else "[LOSS]")
        print(f"total elapsed: {round(time.time() - start_time,2)}s")
        game_over = True
        sys.exit()

    # current cell is safe
    constraints.append((r,c,0))
    
    check_minecount(r, c)

    if cur_board[r][c] != 0:
        lin_combo(r, c, solver, constraints)
    else:
        for nr, nc in neighbors(r, c):
            if cur_board[nr][nc] == -1:
                reveal_cell((nr, nc), solver, constraints)

def lin_combo(r, c, solver, constraints):
    X = [pos for pos in neighbors(r, c) if cur_board[pos[0]][pos[1]] in {-1, -3}]
    rhs = cur_board[r][c] - sum(1 for pos in neighbors(r,c) if cur_board[pos[0]][pos[1]]==-3)
    if X:
        constraints.append((X, rhs))

def check_minecount(r, c):
    num_surrounding_mines = cur_board[r][c]
    num_flagged_mines = sum(1 for nr, nc in neighbors(r,c) if cur_board[nr][nc]==-3)
    if num_surrounding_mines == num_flagged_mines:
        for nr, nc in neighbors(r, c):
            if cur_board[nr][nc]==-1:
                reveal_cell((nr,nc), solver, constraints)

def collect_solutions(solutions, var_map, cnf):
    solver = Solver(name='Glucose3', bootstrap_with=cnf)
    all_solutions = []
    while solver.solve():
        model = solver.get_model()
        sol = np.zeros((ROWS,COLS), dtype=int)
        for (r,c), var in var_map.items():
            sol[r][c] = 1 if var in model else 0
        all_solutions.append(sol)
        # block this solution
        solver.add_clause([-var if var in model else var for var in var_map.values()])
    solver.delete()
    solutions.extend(all_solutions)

def calculate_probabilities(probabilities, solutions):
    for sol in solutions:
        for i in range(ROWS):
            for j in range(COLS):
                if probabilities[i][j]==-1: continue
                probabilities[i][j] += sol[i][j]
    if solutions:
        for i in range(ROWS):
            for j in range(COLS):
                if probabilities[i][j] != -1:
                    probabilities[i][j] = round(probabilities[i][j]/len(solutions),2)

def next_cell(probabilities):
    hashed_probs = {}
    for i in range(ROWS):
        for j in range(COLS):
            p = probabilities[i][j]
            if p not in hashed_probs:
                hashed_probs[p] = []
            hashed_probs[p].append((i,j))
    if 1.0 in hashed_probs:
        for r, c in hashed_probs[1.0]:
            cur_board[r][c] = -3
            for nr, nc in neighbors(r,c):
                if cur_board[nr][nc] not in {-1,-3}:
                    check_minecount(nr,nc)
        del hashed_probs[1.0]
    while True:
        min_prob = min(hashed_probs.keys())
        if min_prob==-1:
            del hashed_probs[min_prob]
            continue
        return rnd.choice(hashed_probs[min_prob])

# -------------------------
# Game setup

MINES = 99
ROWS = 16
COLS = 30
answer_board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
cur_board = [[-1 for _ in range(COLS)] for _ in range(ROWS)]
game_over = False
num_revealed_cells = 0

START_ROW = rnd.randint(0, ROWS-1)
START_COL = rnd.randint(0, COLS-1)

# place mines
for _ in range(MINES):
    r = rnd.randint(0, ROWS-1)
    c = rnd.randint(0, COLS-1)
    while answer_board[r][c]==-2 or (r==START_ROW and c==START_COL):
        r = rnd.randint(0, ROWS-1)
        c = rnd.randint(0, COLS-1)
    answer_board[r][c] = -2

# fill numbers
for i in range(ROWS):
    for j in range(COLS):
        if answer_board[i][j]==-2: continue
        answer_board[i][j] = sum(1 for nr, nc in neighbors(i,j) if answer_board[nr][nc]==-2)

print("ANSWER BOARD:")
print_board(answer_board)

# -------------------------
# Game loop

cur_cell = (START_ROW, START_COL)
start_time = time.time()
constraints = []

while not game_over:
    reveal_cell(cur_cell, Solver(name='Glucose3'), constraints)
    print_board(cur_board, cur_cell)
    print(f"# revealed: {num_revealed_cells}")
    print(f"left to reveal: {ROWS*COLS - num_revealed_cells - MINES}")

    # Build CNF from constraints
    var_map = {}
    cnf = CNF()
    var_counter = 1
    for r in range(ROWS):
        for c in range(COLS):
            if cur_board[r][c] in {-1,-3}:
                var_map[(r,c)] = var_counter
                var_counter +=1
    for X_rhs in constraints:
        if len(X_rhs)==3:
            # (r,c,value)
            r, c, val = X_rhs
            var = var_map[(r,c)]
            cnf.append([var] if val else [-var])
        else:
            # (X_list, rhs)
            X, rhs = X_rhs
            lits = [var_map[pos] for pos in X]
            cnf.extend(CardEnc.equals(lits=lits, bound=rhs, encoding=1))

    # collect solutions
    solutions = []
    collect_solutions(solutions, var_map, cnf)
    print(f"stored {len(solutions)} solutions.")

    # calculate probabilities
    probabilities = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    calculate_probabilities(probabilities, solutions)
    print("PROBABILITIES:")
    print(np.array2string(np.array(probabilities), formatter={'float_kind': '{:>5}'.format}))

    # pick next cell
    cur_cell = next_cell(probabilities)
    print(f"Next cell: {cur_cell} with prob {probabilities[cur_cell[0]][cur_cell[1]]}")
