import concurrent.futures
import os
import sys
import json
from dotenv import load_dotenv

METHODS = ["test_dataset"]

load_dotenv()
EXTRACTOR_DIR = os.getenv("EXTRACTOR_DIR")
dot_experiment_config_file = os.path.join(EXTRACTOR_DIR, ".experiment_config")
assert os.path.exists(dot_experiment_config_file), f"Experiment config file not found: {dot_experiment_config_file}"

EXP_CONFIG = json.load(open(dot_experiment_config_file, "r"))

REPEAT_RANGE = EXP_CONFIG["num_repeats"]
TCS_REDUCTION = EXP_CONFIG["tcs_reduction"]

for line_cnt in EXP_CONFIG["target_lines"]:
    for mut_cnt in EXP_CONFIG["mutation_cnt"]:
        method_key = f"lineCnt{line_cnt}_mutCnt{mut_cnt}_tcs{TCS_REDUCTION}"
        METHODS.append(method_key)

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python3 run_all_methods.py <experiment_label> <feature_type> <project1> <project2> ...")
        print("\t<feature_type>: 0 for all features, 1 for SBFL features, 2 for MBFL features")
        sys.exit(1)

    experiment_label = sys.argv[1]
    repeat_range = REPEAT_RANGE
    feature_type = int(sys.argv[2])
    projects_list = sys.argv[3:]
    project_list = " ".join(projects_list)

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
                task = (experiment_label, f"repeat_{rid}", method, feature_type, repeat_range, project_list)
                futures.append(executor.submit(
                    os.system, f"python3 run_group.py {task[0]} {task[1]} {task[2]} {task[3]} {task[4]} {task[5]} > /dev/null 2>&1"
                ))

        for future in concurrent.futures.as_completed(futures):
            if future.result() != 0:
                print("An error occurred while running the task.")
            else:
                print("Task completed successfully.")
