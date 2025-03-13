#!/usr/bin/env python3
"""
===========================================
       Push_swap Test Visualization
===========================================
This script runs tests for push_swap for various input sizes.
The top row shows graphs for n = 2, 3, and 5 (using all permutations);
the bottom row shows graphs for n = 100 and n = 500 (using a fixed number of tests)
along with a statistics table.
Each graph updates dynamically:
 - The X and Y axis limits are recalculated using √2 over the data range.
 - Reference horizontal lines (thresholds) are drawn if they fall within the current Y limits.
The statistics table shows, for each n, the minimum, maximum, and average moves and average time (ms),
updating with every test.
Closing the window stops the tests gracefully.
"""

import subprocess
import random
import time
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

# CONFIGURATION
PUSH_SWAP_PATH = "./push_swap"
CHECKER_PATH   = "./checker_linux"
TIMEOUT        = 5  # seconds

# For each n we define:
# - tests: None -> use all permutations; or a fixed number of tests.
# - thresholds: reference lines for moves.
N_CONFIG = {
    2:   {'tests': None, 'thresholds': [1]},
    3:   {'tests': None, 'thresholds': [3]},
    5:   {'tests': None, 'thresholds': [12]},
    100: {'tests': 100,  'thresholds': [700, 900, 1100, 1300, 1500]},
    500: {'tests': 500,  'thresholds': [5500, 7000, 8500, 10000, 11500]}
}

global_stop = False  # Flag to stop tests if the window is closed

# Returns the list of tests for a given n
def get_tests(n):
    if n in [2, 3, 5]:
        return list(itertools.permutations(range(n)))
    else:
        tests = []
        count = N_CONFIG[n]['tests']
        i = 0
        while i < count:
            tests.append(tuple(random.sample(range(n), n)))
            i += 1
        return tests

# Assigns a color to a point based on the number of moves and test result
def get_point_color(n, moves, res):
    if res == "KO":
        return "red"
    thresholds = N_CONFIG[n]['thresholds']
    if n in [2, 3, 5]:
        if moves > thresholds[0]:
            return "yellow"
        else:
            return "green"
    elif n == 100:
        if moves <= thresholds[0]:
            return "green"
        elif moves <= thresholds[1]:
            return "limegreen"
        elif moves <= thresholds[2]:
            return "yellowgreen"
        elif moves <= thresholds[3]:
            return "gold"
        elif moves <= thresholds[4]:
            return "orange"
        else:
            return "yellow"
    elif n == 500:
        if moves <= thresholds[0]:
            return "green"
        elif moves <= thresholds[1]:
            return "limegreen"
        elif moves <= thresholds[2]:
            return "yellowgreen"
        elif moves <= thresholds[3]:
            return "gold"
        elif moves <= thresholds[4]:
            return "orange"
        else:
            return "yellow"
    return "green"

# Executes push_swap and checker for the given arguments and returns (result, moves, time_ms)
def run_single_test(args):
    start = time.time()
    try:
        ps = subprocess.run([PUSH_SWAP_PATH] + args, capture_output=True, text=True, timeout=TIMEOUT)
        duration = time.time() - start
        moves_output = ps.stdout.strip()
        if moves_output:
            count = len(moves_output.splitlines())
        else:
            count = 0
    except subprocess.TimeoutExpired:
        duration = TIMEOUT
        count = 0
        moves_output = ""
    if moves_output:
        ck = subprocess.run([CHECKER_PATH] + args, input=moves_output, capture_output=True, text=True)
        res = ck.stdout.strip()
    else:
        nums = list(map(int, args))
        res = "OK" if nums == sorted(nums) else "KO"
    duration_ms = duration * 1000
    return res, count, duration_ms

# Updates the scatter plot with the list of points (each point: [time, moves, color])
def update_scatter(scatter, points):
    if len(points) == 0:
        offsets = np.empty((0, 2))
    else:
        offsets = np.array([[p[0], p[1]] for p in points])
    colors = [p[2] for p in points]
    scatter.set_offsets(offsets)
    scatter.set_color(colors)
    plt.draw()

# Calculates and updates the dynamic limits for both axes (using √2 over the data range)
def update_limits(ax, points):
    if len(points) > 0:
        x_vals = [p[0] for p in points]
        x_min = min(x_vals)
        x_max = max(x_vals)
        center_x = (x_min + x_max) / 2.0
        half_range_x = (x_max - x_min) / 2.0
        if half_range_x == 0:
            half_range_x = 1
        new_xmin = center_x - half_range_x * math.sqrt(2)
        new_xmax = center_x + half_range_x * math.sqrt(2)
        y_vals = [p[1] for p in points]
        y_min = min(y_vals)
        y_max = max(y_vals)
        center_y = (y_min + y_max) / 2.0
        half_range_y = (y_max - y_min) / 2.0
        if half_range_y == 0:
            half_range_y = 1
        new_ymin = center_y - half_range_y * math.sqrt(2)
        new_ymax = center_y + half_range_y * math.sqrt(2)
    else:
        new_xmin, new_xmax = 0, 10
        new_ymin, new_ymax = 0, 10
    ax.set_xlim(new_xmin, new_xmax)
    ax.set_ylim(new_ymin, new_ymax)
    return new_xmin, new_xmax, new_ymin, new_ymax

# Runs all tests for a given n on the axis ax; updates the table (ax_table) at each point
def run_tests_for_n(n, ax, ax_table, table_data, row_index):
    tests = get_tests(n)
    total_tests = len(tests)
    points = []  # Each point: (time_ms, moves, color)
    scatter = ax.scatter([], [], s=40)
    ax.set_box_aspect(1)  # Keep the graph square
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Moves")
    ax.set_title("n = " + str(n))
    
    ref_lines = []  # Current reference lines
    total_moves = []
    total_time = 0.0
    index = 0
    while index < total_tests and not global_stop:
        current_test = tests[index]
        args = list(map(str, current_test))
        res, moves, t_ms = run_single_test(args)
        total_moves.append(moves)
        total_time += t_ms
        color = get_point_color(n, moves, res)
        points.append((t_ms, moves, color))
        update_scatter(scatter, points)
        new_xmin, new_xmax, new_ymin, new_ymax = update_limits(ax, points)
        # Remove previous reference lines and redraw those within limits
        j = 0
        while j < len(ref_lines):
            try:
                ref_lines[j].remove()
            except Exception:
                pass
            j += 1
        ref_lines = []
        thresholds = N_CONFIG[n]['thresholds']
        k = 0
        while k < len(thresholds):
            th = thresholds[k]
            if th >= new_ymin and th <= new_ymax:
                line = ax.axhline(y=th, color="black", linestyle="--", linewidth=1)
                ref_lines.append(line)
            k += 1
        # Update statistics and the table at each point
        if total_moves:
            min_moves = min(total_moves)
            max_moves = max(total_moves)
            avg_moves = sum(total_moves) / len(total_moves)
            avg_time = total_time / len(total_moves)
        else:
            min_moves = max_moves = avg_moves = avg_time = 0
        table_data[row_index] = [str(n), str(min_moves), str(max_moves), f"{avg_moves:.2f}", f"{avg_time:.2f}"]
        update_table(ax_table, table_data)
        plt.pause(0.001)
        index += 1
    if total_moves:
        min_moves = min(total_moves)
        max_moves = max(total_moves)
        avg_moves = sum(total_moves) / len(total_moves)
        avg_time = total_time / len(total_moves)
    else:
        min_moves = max_moves = avg_moves = avg_time = 0
    return min_moves, max_moves, avg_moves, avg_time

# Updates the table in ax with table_data
def update_table(ax, table_data):
    ax.clear()
    ax.axis('tight')
    ax.axis('off')
    col_labels = ["N", "Min moves", "Max moves", "Avg moves", "Avg time (ms)"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.draw()

def on_close(event):
    global global_stop
    global_stop = True

def main():
    plt.ion()
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("push_swap Correction Test Visualization", fontsize=16)
    # Layout: Top row for n = 2, 3, 5; Bottom row for n = 100, 500, and the table.
    gs = fig.add_gridspec(2, 3)
    ax_n2    = fig.add_subplot(gs[0, 0])
    ax_n3    = fig.add_subplot(gs[0, 1])
    ax_n5    = fig.add_subplot(gs[0, 2])
    ax_n100  = fig.add_subplot(gs[1, 0])
    ax_n500  = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[1, 2])
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Initialize the table with one row for each n in the order: 2, 3, 5, 100, 500
    order = [2, 3, 5, 100, 500]
    table_data = []
    i = 0
    while i < len(order):
        table_data.append([str(order[i]), "0", "0", "0.00", "0.00"])
        i += 1
    
    pairs = [(2, ax_n2), (3, ax_n3), (5, ax_n5), (100, ax_n100), (500, ax_n500)]
    i = 0
    while i < len(pairs) and not global_stop:
        current_n, current_ax = pairs[i]
        run_tests_for_n(current_n, current_ax, ax_table, table_data, i)
        i += 1
    
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
