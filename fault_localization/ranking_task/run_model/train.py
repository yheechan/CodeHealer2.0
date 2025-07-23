import math
import os
import sys
import torch
import time
import pickle
import random
import dotenv
import shutil
from model.model_semantic_spec_mutation import MLP


def get_batch(x_pos, x_neg, idx, bs):
    x_pos_batch = x_pos[idx: idx + bs]
    x_neg_batch = x_neg[idx: idx + bs]
    return torch.FloatTensor(x_pos_batch), torch.FloatTensor(x_neg_batch)


def get_x_batch(x, idx, bs):
    x_batch = x[idx: idx + bs]
    return torch.FloatTensor(x_batch)


def load_from_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def print_parameter_statistics(model):
    total_num = [p.numel() for p in model.parameters()]
    trainable_num = [p.numel() for p in model.parameters() if p.requires_grad]
    print("Total parameters: {}".format(sum(total_num)))
    print("Trainable parameters: {}".format(sum(trainable_num)))


def find_all_index(l, value):
    results = []
    for i in range(0, len(l)):
        if l[i] == value:
            results.append(i)
    return results


def copy_list(l):
    results = []
    for i in l:
        results.append(i)
    return results


def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).mean()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).mean()

    return model1_loss, model2_loss


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 train.py <experiment_label> <repeat> <method> <task_id>")
        sys.exit(1)
    
    dotenv.load_dotenv()

    experiment_label = sys.argv[1]
    repeat = sys.argv[2]
    method = sys.argv[3]
    task_id = sys.argv[4]

    print(f"Experiment Label: {experiment_label}")
    print(f"Repeat: {repeat}")
    print(f"Method: {method}")
    print(f"Task ID: {task_id}")

    RESEARCH_DATA_DIR = os.getenv("RESEARCH_DATA")
    if not RESEARCH_DATA_DIR:
        print("Please set the RESEARCH_DATA_DIR environment variable.")
        sys.exit(1)

    # Get statements path and faulty statements path
    statements_path = os.path.join(RESEARCH_DATA_DIR, experiment_label, "postprocessed_dataset", "statement_info", "statements.pkl")
    faulty_statements_path = os.path.join(RESEARCH_DATA_DIR, experiment_label, "postprocessed_dataset", "statement_info", "faulty_statement_set.pkl")
    if not os.path.exists(statements_path) or not os.path.exists(faulty_statements_path):
        print("Statements or faulty statements file not found.")
        sys.exit(1)
    if not os.path.exists(faulty_statements_path):
        print("Faulty statements file not found: {}".format(faulty_statements_path))
        sys.exit(1)
    
    # Make DLFL results directory
    dlfl_out_base_dir = os.path.join(RESEARCH_DATA_DIR, experiment_label, "dlfl_out", "experiment_raw_results", repeat)
    if not os.path.exists(dlfl_out_base_dir):
        os.makedirs(dlfl_out_base_dir)

    test_root_dir = os.path.join(RESEARCH_DATA_DIR, experiment_label, "postprocessed_dataset", repeat, "test_dataset", f"group_{task_id}")
    if method == "test_dataset":
        train_root_dir = test_root_dir
        output_base_dir = os.path.join(dlfl_out_base_dir, "test_dataset")
        sus_pos_rerank_dir = os.path.join(dlfl_out_base_dir, "test_dataset", "sus_pos_rerank")
    else:
        train_root_dir = os.path.join(RESEARCH_DATA_DIR, experiment_label, "postprocessed_dataset", repeat, "methods", method, f"group_{task_id}")
        output_base_dir = os.path.join(dlfl_out_base_dir, "methods", method)
        sus_pos_rerank_dir = os.path.join(dlfl_out_base_dir, "methods", method, "test_dataset", "sus_pos_rerank")

    # Make sure the results directory exists
    models_dir = os.path.join(output_base_dir, "saved_models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    results_dir = os.path.join(output_base_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(sus_pos_rerank_dir):
        os.makedirs(sus_pos_rerank_dir)

    # Load data
    train_x_pos = load_from_file(train_root_dir + '/train/x_pos.pkl')
    train_x_neg = load_from_file(train_root_dir + '/train/x_neg.pkl')
    val_x_pos = load_from_file(train_root_dir + '/val/x_pos.pkl')
    val_x_neg = load_from_file(train_root_dir + '/val/x_neg.pkl')
    test_x_data = load_from_file(test_root_dir + '/test/x.pkl')
    test_y_data = load_from_file(test_root_dir + '/test/y.pkl')

    EPOCHS = 40
    BATCH_SIZE = 64
    t = 0.005
    USE_GPU = True

    model = MLP()

    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-4)
    print_parameter_statistics(model)

    train_loss_ = []
    val_loss_ = []

    best_result = 1e3

    # double co-teaching
    model_d = MLP()
    if USE_GPU:
        model_d.cuda()
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001, weight_decay=1e-4)

    # end

    print('Start training...')
    for epoch in range(EPOCHS):
        print(epoch)
        start_time = time.time()

        # shuffle data before training for each epoch
        random.seed(888)
        random.shuffle(train_x_pos)
        random.seed(888)
        random.shuffle(train_x_neg)

        # training phase
        model.train()
        total_loss = 0.0
        total = 0
        i = 0
        # co-teaching
        model_d.train()
        # end

        while i < len(train_x_pos):
            batch = get_batch(train_x_pos, train_x_neg, i, BATCH_SIZE)
            i += BATCH_SIZE
            batch_x_pos, batch_x_neg = batch
            if USE_GPU:
                batch_x_pos, batch_x_neg = batch_x_pos.cuda(), batch_x_neg.cuda()
            pre_pos = model(batch_x_pos)
            pre_neg = model(batch_x_neg)
            pre_pos = torch.softmax(pre_pos, dim=-1)[:, 0]
            pre_neg = torch.softmax(pre_neg, dim=-1)[:, 0]
            loss = torch.max(torch.zeros_like(pre_pos), 0.1 - (pre_pos - pre_neg))

            pre_pos_d = model_d(batch_x_pos)
            pre_neg_d = model_d(batch_x_pos)
            pre_pos_d = torch.softmax(pre_pos_d, dim=-1)[:, 0]
            pre_neg_d = torch.softmax(pre_neg_d, dim=-1)[:, 0]
            loss_d = torch.max(torch.zeros_like(pre_pos_d), 0.1 - (pre_pos_d - pre_neg_d))

            # co-teaching
            R = 1 - t * epoch

            loss_filter, loss_filter_d = co_teaching_loss(loss, loss_d, R)

            model.zero_grad()
            model_d.zero_grad()

            loss_filter.backward()
            loss_filter_d.backward()

            optimizer.step()
            optimizer_d.step()

            total += int(BATCH_SIZE * R)
            total_loss += loss_filter.item() * BATCH_SIZE * R

        train_loss_.append(total_loss / total)

        # validation phase
        model.eval()
        total_loss = 0.0
        total = 0

        i = 0
        while i < len(val_x_pos):
            batch = get_batch(val_x_pos, val_x_neg, i, BATCH_SIZE)
            i += BATCH_SIZE
            batch_x_pos, batch_x_neg = batch
            if USE_GPU:
                batch_x_pos, batch_x_neg = batch_x_pos.cuda(), batch_x_neg.cuda()
            output_pos = model(batch_x_pos)
            output_neg = model(batch_x_neg)
            output_pos = torch.softmax(output_pos, dim=-1)[:, 0]
            output_neg = torch.softmax(output_neg, dim=-1)[:, 0]
            # hinge loss
            loss = torch.max(torch.zeros_like(output_pos), 0.1 - (output_pos - output_neg)).mean()

            total += len(batch_x_pos)
            total_loss += loss.item() * len(batch_x_pos)

        val_loss_.append(total_loss / total)

        if val_loss_[-1] < best_result:
            model_path = os.path.join(models_dir, "model_params_{}.pkl".format(task_id))
            model_d_path = os.path.join(models_dir, "model_params_d_{}.pkl".format(task_id))
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(model_d_path):
                os.remove(model_d_path)
            torch.save(model.state_dict(), model_path)
            torch.save(model_d.state_dict(), model_d_path)
            best_result = val_loss_[-1]
            print("Saving model: epoch_{} ".format(epoch + 1) + "=" * 20)

        end_time = time.time()
        print('[Epoch: %3d/%3d]\nTraining Loss: %.10f,\t\tValidation Loss: %.10f\nTime Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], end_time - start_time))

    # test phase
    model_path = os.path.join(models_dir, "model_params_{}.pkl".format(task_id))
    if not os.path.exists(model_path):
        print("Model not found: {}".format(model_path))
        sys.exit(1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(statements_path, "rb") as file:
        statements = pickle.load(file)
    
    with open(faulty_statements_path, "rb") as file:
        faulty_statements = pickle.load(file)
    
    results_txt = os.path.join(results_dir, "result_{}.txt".format(task_id))
    if os.path.exists(results_txt):
        os.remove(results_txt)

    with open(results_txt, "w") as result_file:
        for version in test_x_data:
            result_file.write("==={}\n".format(version))
            version_x_data = test_x_data[version]
            predict_score = torch.empty(0)
            if USE_GPU:
                predict_score = predict_score.cuda()

            i = 0
            while i < len(version_x_data):
                test_inputs = get_x_batch(version_x_data, i, BATCH_SIZE)
                i += BATCH_SIZE
                if USE_GPU:
                    test_inputs = test_inputs.cuda()

                output = model(test_inputs)
                output_softmax = torch.softmax(output, dim=-1)[:, 0]
                predict_score = torch.cat((predict_score, output_softmax))

            predict_score = predict_score.cpu().detach().numpy().tolist()
            sus_lines = statements[version]
            sus_pos_rerank_dict = {}
            for i, line in enumerate(sus_lines):
                sus_pos_rerank_dict[line] = predict_score[i]
            sorted_sus_list = sorted(sus_pos_rerank_dict.items(), key=lambda x: x[1], reverse=True)

            # output new suspicious file generated by our model
            out_susp_dir = os.path.join(sus_pos_rerank_dir, version)
            if os.path.exists(out_susp_dir):
                shutil.rmtree(out_susp_dir)
            os.makedirs(out_susp_dir)

            with open(os.path.join(out_susp_dir, "ranking.txt"), "w") as file:
                for (line, score) in sorted_sus_list:
                    file.write("{} {}\n".format(line, score))

            rerank_sus_lines = [line for (line, score) in sorted_sus_list]
            rerank_sus_scores = [float(score) for (line, score) in sorted_sus_list]
            current_faulty_statements = faulty_statements[version]
            for one_position_set in current_faulty_statements:
                current_min_index = 1e8
                for buggy_line in one_position_set:
                    if buggy_line not in rerank_sus_lines:
                        continue
                    buggy_index = len(rerank_sus_scores) - rerank_sus_scores[::-1].index(
                        rerank_sus_scores[rerank_sus_lines.index(buggy_line)])
                    if buggy_index < current_min_index:
                        current_min_index = buggy_index
                if current_min_index == 1e8:
                    continue
                result_file.write(str(current_min_index) + "\n")
