import os
import sys
import dotenv
import json
import time


def parse_model_output_file(results_dir):
    total_result = []
    version_list = []
    for group_index in range(1, 11):
        txt_file = os.path.join(results_dir, f"result_{group_index}.txt")
        with open(txt_file, "r") as file:
            content = file.readlines()
        current_result = []
        first_flag = True
        for line in content:
            line = line.strip()
            if line.startswith("==="):
                if len(version_list) > 0:
                    if len(current_result) == 0:
                        if not first_flag:
                            version_list.pop()
                    else:
                        total_result.append(current_result[:])
                        current_result = []
                version_list.append(line.replace("===", ""))
            else:
                current_result.append(float(line))
            first_flag = False
        if len(current_result) > 0:
            total_result.append(current_result[:])
        else:
            version_list.pop()
    # print(len(total_result))
    # print(len(version_list))
    return total_result, version_list


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 run_group.py <experiment_label> <repeat> <method> <feature_type> <project_list:>")
        sys.exit(1)
    
    dotenv.load_dotenv()
    experiment_label = sys.argv[1]
    repeat = sys.argv[2]
    method = sys.argv[3]
    feature_type = int(sys.argv[4])
    projects_list = sys.argv[5:]

    print(f"Experiment Label: {experiment_label}")
    print(f"Repeat: {repeat}")
    print(f"Method: {method}")

    #10-fold cross-validation training and testing
    train_start_time = time.time()
    for group_index in range(1, 11):
        status = os.system("python3 train.py {} {} {} {} {} > /dev/null 2>&1".format(
            experiment_label, repeat, method, feature_type, group_index
        ))
        assert status == 0
    train_time_taken_seconds = time.time() - train_start_time
    
    RESEARCH_DATA_DIR = os.getenv("RESEARCH_DATA")
    if not RESEARCH_DATA_DIR:
        print("Please set the RESEARCH_DATA_DIR environment variable.")
        sys.exit(1)
    
    # Make DLFL results directory
    if feature_type == 0:
        exp_dir_name = "experiment_raw_results"
    elif feature_type == 1:
        exp_dir_name = "experiment_raw_results_sbfl"
    elif feature_type == 2:
        exp_dir_name = "experiment_raw_results_mbfl"

    dlfl_out_base_dir = os.path.join(RESEARCH_DATA_DIR, experiment_label, "dlfl_out", exp_dir_name, repeat)
    if not os.path.exists(dlfl_out_base_dir):
        os.makedirs(dlfl_out_base_dir)

    if method == "test_dataset":
        output_base_dir = os.path.join(dlfl_out_base_dir, "test_dataset")
    else:
        output_base_dir = os.path.join(dlfl_out_base_dir, "methods", method)
    
    results_dir = os.path.join(output_base_dir, "results")

    total_result, version_list = parse_model_output_file(results_dir)

    top1_total = 0
    top3_total = 0
    top5_total = 0
    top10_total = 0
    all_position_total = []
    first_position_total = []

    print("\nStatistics for each project.")
    projects = projects_list
    final_results = {}
    for project in projects:
        print("=" * 20)
        print(project)

        top1 = 0
        top3 = 0
        top5 = 0
        top10 = 0
        all_position = []
        first_position = []

        for i, version in enumerate(version_list):
            if not version.startswith(project):
                continue
            bugs = total_result[i]
            rank = []
            for bug in bugs:
                rank.append(bug)
            min_rank = min(rank)
            avg_rank = sum(rank) / len(rank)

            if min_rank <= 1:
                # print(version)
                top1 += 1
            if min_rank <= 3:
                top3 += 1
            if min_rank <= 5:
                top5 += 1
            if min_rank <= 10:
                top10 += 1
            first_position.append(min_rank)
            all_position.append(avg_rank)
        final_results[project] = {
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "top10": top10,
            "mfr": round(sum(first_position) / len(first_position), 2),
            "mar": round(sum(all_position) / len(all_position), 2),
        }
        print("Top1\t{}".format(top1))
        print("Top3\t{}".format(top3))
        print("Top5\t{}".format(top5))
        print("Top10\t{}".format(top10))
        print("MFR\t{}".format(round(sum(first_position) / len(first_position), 2)))
        print("MAR\t{}".format(round(sum(all_position) / len(all_position), 2)))

        top1_total += top1
        top3_total += top3
        top5_total += top5
        top10_total += top10
        all_position_total += all_position
        first_position_total += first_position

    print("\nStatistics for all projects.")
    print("=" * 20)
    print("Top1\t{}".format(top1_total))
    print("Top3\t{}".format(top3_total))
    print("Top5\t{}".format(top5_total))
    print("Top10\t{}".format(top10_total))
    print("MFR\t{}".format(round(sum(first_position_total) / len(first_position_total), 2)))
    print("MAR\t{}".format(round(sum(all_position_total) / len(all_position_total), 2)))

    final_results_path = os.path.join(output_base_dir, "final_results.json")
    final_results["total"] = {
        "top1": top1_total,
        "top3": top3_total,
        "top5": top5_total,
        "top10": top10_total,
        "mfr": round(sum(first_position_total) / len(first_position_total), 2),
        "mar": round(sum(all_position_total) / len(all_position_total), 2),
        "train_time_seconds": train_time_taken_seconds,
    }
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
