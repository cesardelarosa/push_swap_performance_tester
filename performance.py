#!/usr/bin/env python3
import subprocess
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.widgets import TextBox
from scipy.optimize import curve_fit

# Configuration
N_MAX          = 500         # maximum number of elements (n from 1 to N_MAX)
PUSH_SWAP_PATH = "./push_swap"
CHECKER_PATH   = "./checker"
TIMEOUT        = 5            # timeout (seconds) for push_swap
M_LIM          = 5000       # default limit for "moves"
T_LIM          = 150         # default limit for "time" (ms)

# Number of tests per n (set to 1, but you can increase for more robust averages)
def tests_count(n):
	# You can increase this value to average over multiple runs per n.
	return 50

# Run a test for a given n and return:
#   t: number of tests executed
#   green_pts: list of tuples (n, moves, time_ms) for tests that passed (OK)
#   red_pts: list of tuples (n, moves, time_ms) for tests that failed (KO)
#   ok_count: count of OK tests
def run_test(n):
	t = tests_count(n)
	green_pts = []
	red_pts   = []
	ok_count  = 0

	for _ in range(t):
		# Generate a random permutation of [0, n)
		nums = random.sample(range(n), n)
		args = list(map(str, nums))
		try:
			start = time.time()
			ps = subprocess.run([PUSH_SWAP_PATH] + args, capture_output=True, text=True, timeout=TIMEOUT)
			duration = time.time() - start
			moves_output = ps.stdout.strip()
			count = len(moves_output.splitlines()) if moves_output else 0
		except subprocess.TimeoutExpired:
			duration = TIMEOUT
			count = 0
			moves_output = ""
		# Verify the result with checker
		if moves_output:
			ck = subprocess.run([CHECKER_PATH] + args, input=moves_output, capture_output=True, text=True)
			res = ck.stdout.strip()
		else:
			# If the list is already sorted, consider it OK
			if list(nums) == sorted(nums):
				res = "OK"
			else:
				res = "KO"
		# Store data (time in milliseconds)
		duration_ms = duration * 1000
		if res == "OK":
			ok_count += 1
			green_pts.append((n, count, duration_ms))
		else:
			red_pts.append((n, count, duration_ms))
	return t, green_pts, red_pts, ok_count

# Model function: f(n) = A * n^a + B * log(n)
def model_complexity(n, A, a, B):
	return A * np.power(n, a) + B * np.log(n)

# Fit the non-linear model using curve_fit and return the estimated parameters and covariance.
def fit_complexity_nonlinear(n_vals, y_vals):
	# Ensure inputs are numpy arrays and n_vals are > 0 (log(n) requires n > 0)
	n_vals = np.array(n_vals, dtype=float)
	y_vals = np.array(y_vals, dtype=float)
	# Provide an initial guess for A, a, and B
	initial_guess = [1e-3, 1.0, 1e-3]
	
	# Perform the curve fitting
	popt, pcov = curve_fit(model_complexity, n_vals, y_vals, p0=initial_guess, maxfev=10000)
	
	# Calculate residuals and RMSE
	y_pred = model_complexity(n_vals, *popt)
	residuals = y_vals - y_pred
	rmse = np.sqrt(np.mean(residuals**2))
	
	print("Non-linear model fit (f(n) = A*n^a + B*log(n)):")
	print(f"  Estimated A: {popt[0]:.4e}")
	print(f"  Estimated a: {popt[1]:.4f}")
	print(f"  Estimated B: {popt[2]:.4e}")
	print(f"  RMSE: {rmse:.4e}")
	
	return popt, pcov

# Update the 3D plot (ax1)
def update_ax1(scatter_green, scatter_red, green_points, red_points):
	green_n = [pt[0] for pt in green_points]
	green_moves = [pt[1] for pt in green_points]
	green_time  = [pt[2] for pt in green_points]
	red_n   = [pt[0] for pt in red_points]
	red_moves = [pt[1] for pt in red_points]
	red_time  = [pt[2] for pt in red_points]
	scatter_green._offsets3d = (green_n, green_moves, green_time)
	scatter_red._offsets3d   = (red_n, red_moves, red_time)

# Update the 2D plot of moves vs n (ax2)
def update_ax2(scatter, green_points):
	x = [pt[0] for pt in green_points]
	y = [pt[1] for pt in green_points]
	if x and y:
		scatter.set_offsets(np.column_stack((x, y)))

# Update the 2D plot of time vs n (ax3)
def update_ax3(scatter, green_points):
	x = [pt[0] for pt in green_points]
	y = [pt[2] for pt in green_points]
	if x and y:
		scatter.set_offsets(np.column_stack((x, y)))

# Update the plot of moves vs time for a specific n (ax4)
def update_ax4(ax, scatter_green, scatter_red, green_points, red_points, n_val, extra_data):
	# Filter data for the selected n (x = time, y = moves)
	green_filtered = [(pt[2], pt[1]) for pt in green_points if pt[0] == n_val]
	red_filtered   = [(pt[2], pt[1]) for pt in red_points if pt[0] == n_val]
	if green_filtered:
		green_arr = np.array(green_filtered)
		scatter_green.set_offsets(green_arr)
		# Calculate new axis limits with a margin (√2 factor)
		x_min = np.min(green_arr[:,0])
		x_max = np.max(green_arr[:,0])
		y_min = np.min(green_arr[:,1])
		y_max = np.max(green_arr[:,1])
		factor = np.sqrt(2)
		center_x = (x_min + x_max) / 2.0
		half_range_x = (x_max - x_min) / 2.0
		new_xmin = center_x - half_range_x * factor if half_range_x > 0 else center_x - 1
		new_xmax = center_x + half_range_x * factor if half_range_x > 0 else center_x + 1
		center_y = (y_min + y_max) / 2.0
		half_range_y = (y_max - y_min) / 2.0
		new_ymin = center_y - half_range_y * factor if half_range_y > 0 else center_y - 1
		new_ymax = center_y + half_range_y * factor if half_range_y > 0 else center_y + 1
		ax.set_xlim(new_xmin, new_xmax)
		ax.set_ylim(new_ymin, new_ymax)
	else:
		scatter_green.set_offsets(np.empty((0,2)))
	if red_filtered:
		red_arr = np.array(red_filtered)
		scatter_red.set_offsets(red_arr)
	else:
		scatter_red.set_offsets(np.empty((0,2)))
	ax.set_title(f"Moves vs Time for n = {n_val}")
	# Save the limits in extra_data
	if green_filtered:
		extra_data[n_val] = ((new_xmin, new_xmax), (new_ymin, new_ymax))
	else:
		if n_val in extra_data:
			del extra_data[n_val]

# Update global axis limits for ax1, ax2, and ax3 based on all data up to current n.
def update_global_limits(ax1, ax2, ax3, current_n, all_green_points, all_red_points):
	all_points = [pt for pt in (all_green_points + all_red_points) if pt[0] <= current_n]
	if all_points:
		moves_vals = [pt[1] for pt in all_points]
		time_vals  = [pt[2] for pt in all_points]
		max_moves = max(moves_vals)
		max_time  = max(time_vals)
	else:
		max_moves = 0
		max_time  = 0
	factor = np.sqrt(2)
	ax1.set_xlim(1, current_n)
	ax1.set_ylim(0, max_moves*factor if max_moves > 0 else M_LIM)
	ax1.set_zlim(0, max_time*factor if max_time > 0 else T_LIM)
	ax2.set_xlim(1, current_n)
	ax2.set_ylim(0, max_moves*factor if max_moves > 0 else M_LIM)
	ax3.set_xlim(1, current_n)
	ax3.set_ylim(0, max_time*factor if max_time > 0 else T_LIM)

# Main function
def main():
	plt.ion()  # interactive mode
	fig = plt.figure(figsize=(14,10))
	fig.suptitle("push_swap Performance Test Visualization", fontsize=16)    

	# Create a 2x2 layout:
	ax1 = fig.add_subplot(2,2,1, projection='3d')
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)

	# Configure ax1 (3D plot)
	ax1.set_xlabel("Number of elements", fontsize=12)
	ax1.set_ylabel("Number of moves", fontsize=12)
	ax1.set_zlabel("Execution time (ms)", fontsize=12)
	ax1.set_title("3D: n vs moves vs time", fontsize=14)
	ax1.set_xlim(1, N_MAX)
	ax1.set_ylim(0, M_LIM)
	ax1.set_zlim(0, T_LIM)

	# Configure ax2: moves vs n
	ax2.set_xlabel("Number of elements", fontsize=12)
	ax2.set_ylabel("Number of moves", fontsize=12)
	ax2.set_title("Moves vs n", fontsize=14)
	ax2.set_xlim(1, N_MAX)
	ax2.set_ylim(0, M_LIM)

	# Configure ax3: time vs n
	ax3.set_xlabel("Number of elements", fontsize=12)
	ax3.set_ylabel("Execution time (ms)", fontsize=12)
	ax3.set_title("Time vs n", fontsize=14)
	ax3.set_xlim(1, N_MAX)
	ax3.set_ylim(0, T_LIM)

	# Configure ax4: moves vs time for a specific n
	ax4.set_xlabel("Execution time (ms)", fontsize=12)
	ax4.set_ylabel("Number of moves", fontsize=12)
	ax4.set_title("Moves vs Time for n = 250", fontsize=14)

	# Initialize empty scatter plots
	scatter_green_3d = ax1.scatter([], [], [], color="green", s=10, label="OK")
	scatter_red_3d   = ax1.scatter([], [], [], color="red", s=10, label="KO")
	scatter_ax2 = ax2.scatter([], [], color="green", s=10)
	scatter_ax3 = ax3.scatter([], [], color="green", s=10)
	scatter_ax4_green = ax4.scatter([], [], color="green", s=20, label="OK")
	scatter_ax4_red   = ax4.scatter([], [], color="red", s=20, label="KO")
	ax4.legend()

	# Global lists to store data points
	all_green_points = []
	all_red_points   = []
	extra_data = {}  # dictionary for storing axis limits for ax4
	n_selected = [N_MAX]

	# Create a TextBox to change n in ax4 (active after data collection)
	textbox_ax = fig.add_axes([0.25, 0.01, 0.1, 0.05])
	text_box = TextBox(textbox_ax, 'Select n:', initial=str(N_MAX))

	def submit(text):
		try:
			val = int(text)
			if 1 <= val <= N_MAX:
				n_selected[0] = val
				update_ax4(ax4, scatter_ax4_green, scatter_ax4_red,
				           all_green_points, all_red_points, n_selected[0], extra_data)
				plt.draw()
			else:
				print("n must be between 1 and", N_MAX)
		except ValueError:
			print("Enter a valid integer")
	text_box.on_submit(submit)

	stop_flag = [False]
	def on_close(event):
		stop_flag[0] = True
	fig.canvas.mpl_connect('close_event', on_close)

	# Loop over n values with a progress bar
	for n in tqdm(range(1, N_MAX+1), desc="Processing n values"):
		if not plt.fignum_exists(fig.number) or stop_flag[0]:
			break
		_, green_pts, red_pts, _ = run_test(n)
		all_green_points.extend(green_pts)
		all_red_points.extend(red_pts)
		update_ax1(scatter_green_3d, scatter_red_3d, all_green_points, all_red_points)
		update_ax2(scatter_ax2, all_green_points)
		update_ax3(scatter_ax3, all_green_points)
		update_global_limits(ax1, ax2, ax3, n, all_green_points, all_red_points)
		update_ax4(ax4, scatter_ax4_green, scatter_ax4_red,
		           all_green_points, all_red_points, n, extra_data)
		plt.pause(0.001)

	# After data collection, perform the non-linear regression analysis
	if all_green_points:
		# Extract data for n, moves, and time from the successful tests
		n_vals    = np.array([pt[0] for pt in all_green_points])
		moves_vals = np.array([pt[1] for pt in all_green_points])
		time_vals  = np.array([pt[2] for pt in all_green_points])
		
		print("\nFitting non-linear model for 'moves' (f(n) = A*n^a + B*log(n))...")
		popt_moves, _ = fit_complexity_nonlinear(n_vals, moves_vals)
		print("\nFitting non-linear model for 'time' (f(n) = A*n^a + B*log(n))...")
		popt_time, _ = fit_complexity_nonlinear(n_vals, time_vals)
		
		# Prepare a range of n values for plotting the fitted curves
		n_range = np.linspace(1, max(n_vals), 200)
		
		# Plot fitted curve on ax2 (moves vs n)
		f_moves = model_complexity(n_range, *popt_moves)
		ax2.plot(n_range, f_moves, color="blue", linewidth=2,
		         label=f"Fit: A*n^a + B*log(n)")
		ax2.legend(fontsize=10)
		
		# Plot fitted curve on ax3 (time vs n)
		f_time = model_complexity(n_range, *popt_time)
		ax3.plot(n_range, f_time, color="blue", linewidth=2,
		         label=f"Fit: A*n^a + B*log(n)")
		ax3.legend(fontsize=10)
		
	plt.ioff()
	plt.show()

if __name__ == '__main__':
	main()
