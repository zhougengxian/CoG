import argparse
import os
import json
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="hotpot_e", help="choose the dataset.")
    
    # Allow specifying either a direct file path or an experiment ID folder
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_file", type=str, help="The path to the output file (e.g., summary_light_results.json).")
    group.add_argument("--experiment_id", type=str, help="The experiment ID (folder name) to evaluate. Assumes new format in ../CoG/results.")

    parser.add_argument("--data_question_string", type=str,
                        default='question', help="key to access the question in output.")
    parser.add_argument("--DoG", action='store_true', help="Indicate that the output file is from the DoG format.")
    args = parser.parse_args()

    # Determine the correct path to the results file
    output_file_path = args.output_file
    if args.experiment_id:
        # This assumes the script is run from the 'eval/' directory or project root in a way that this relative path works.
        # Let's make it more robust by anchoring from the script's own location.
        output_file_path = os.path.join(os.path.dirname(__file__), "..", "CoG", "results", args.experiment_id, "summary_light_results.json")
        print(f"Info: Evaluating experiment ID '{args.experiment_id}'. Using result file: {output_file_path}")

    try:
        ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, output_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the output file at '{output_file_path}' or the dataset file.")
        if args.experiment_id:
            print("Hint: Make sure the experiment ID is correct and the 'summary_light_results.json' file exists in its directory.")
        exit() # Exit if files are not found
    except json.JSONDecodeError:
        print(f"Error: The file '{output_file_path}' is not a valid JSON or JSONL file.")
        exit()

    ground_truth_datas_question2id = {v[question_string]: i for i, v in enumerate(ground_truth_datas)}
    data_question_string = args.data_question_string

    num_right = 0
    num_error = 0
    err_list = []

    for data_i, data in enumerate(output_datas):
        cur_question = data[data_question_string]
        
        if cur_question not in ground_truth_datas_question2id:
            print(f"Warning: Skipping a result because its question could not be found in the ground truth data: '{cur_question}'")
            continue

        origin_data_i = ground_truth_datas_question2id[cur_question]
        answers = align(args.dataset, question_string, data, ground_truth_datas, origin_data=ground_truth_datas[origin_data_i], data_question_string=data_question_string)

        if args.DoG:
            results = data.get('results', '')
        else:
            results = data.get('answer', '') # Use .get for safety
            
        if results is None:
            results = ''
        matched = exact_match(results, answers)

        if matched:
            num_right += 1
        else:
            num_error += 1
            err_list.append(data_i)

    em_score = float(num_right/len(output_datas)) if len(output_datas) > 0 else 0
    print(f'total questions: {len(output_datas)}')
    print("Exact Match: {}".format(em_score))
    print("right: {}, error: {}".format(num_right, num_error))
