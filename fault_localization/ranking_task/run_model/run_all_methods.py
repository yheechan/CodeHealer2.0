import concurrent.futures
import os
import sys

METHODS = ["test_dataset"]

for mid in range(1, 11):
    mtc_key = f"mutCnt_{mid}"
    METHODS.append(mtc_key)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 run_all_methods.py <experiment_label> <repeat>")
        sys.exit(1)

    experiment_label = sys.argv[1]
    repeat_range = int(sys.argv[2])

    # python3 run_group <experiment_label> <repeat> <method>
    # Implement a code that run all methods in parallel with maximum 4 batches
    tasks = []
    for method in METHODS:
        for rid in range(1, repeat_range + 1):
            tasks.append((experiment_label, f"repeat_{rid}", method))

    print(f"Running tasks for experiment: {experiment_label} with repeat range: {repeat_range}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for method in METHODS:
            for rid in range(1, repeat_range + 1):
                task = (experiment_label, f"repeat_{rid}", method)
                futures.append(executor.submit(
                    os.system, f"python3 run_group.py {task[0]} {task[1]} {task[2]} > /dev/null 2>&1"
                ))

        for future in concurrent.futures.as_completed(futures):
            if future.result() != 0:
                print("An error occurred while running the task.")
            else:
                print("Task completed successfully.")
