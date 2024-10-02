import json, re
import copy
import os, json
import numpy as np
import openai
from dotenv import load_dotenv
import warnings
import argparse
from tqdm import tqdm
from requests.packages.urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter('ignore', NotOpenSSLWarning)
load_dotenv()
from interactive import Debate

def write_file_check(args):
    # create a new json file or edit existing json file
    file_name = os.path.join(args.output, f"right_aff_{args.dataset}_{args.max_round}_{args.model_config}.json")
    if not args.resume:
        if os.path.exists(file_name): os.remove(file_name)
    os.makedirs(os.path.dirname(file_name), exist_ok=True) # Create the directory if it doesn't exist

def save_json(args, data):
    file_name = os.path.join(args.output, f"right_aff_{args.dataset}_{args.max_round}_{args.model_config}.json")
    with open(file_name, 'a') as json_file:
        json.dump(data, json_file, indent=4)
        json_file.write('\n')

def correct_json(file_path):
    # correct the json file format
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace('}\n{', '},\n{')
    content = '[\n' + content + '\n]'
    data = json.loads(content)
    return data

def eval(args):
    def compute_accuracy(gt, pred):
        assert len(gt) == len(pred)
        no_questions = len(gt)
        acc = []
        for i in range(0, no_questions):
            if args.dataset=="chess":
                if pred[i] in gt[i]: acc.append(1)
                else: acc.append(0)
            else:
                if gt[i] == pred[i]: acc.append(1)
                else: acc.append(0)

        # find the wrong question index number
        wrong_index = find_wrong_question_index(acc) 
        final_acc = np.mean(acc)
        var = np.std(acc) / (len(acc)**0.5)
        print(final_acc, var)
        return final_acc, var, wrong_index

    def find_wrong_question_index(acc_list):
        return np.where(np.array(acc_list)==0)[0].tolist()

    file_path = f"{args.output}right_aff_{args.dataset}_{args.max_round}_{args.model_config}.json"
    
    # read the json file
    try: 
        with open(file_path, 'r') as file:
            data = json.load(file)
    except: data = correct_json(file_path)

    gts, predictions = [], []
    try:
        for d in data:
            gts.append(d['ground_truth'])
            predictions.append(d['prediction'])

        final_acc, var, wrong_index = compute_accuracy(gts, predictions)
        save_json(args,
            {
                "accuracy": final_acc, 
                "varience": var, 
                "wrong_answered_index": wrong_index,
            }
        ) # save the json file
        data = correct_json(file_path) # fetch the new file
        with open(file_path, 'w') as corrected_file:
            json.dump(data, corrected_file, indent=4)
    except: final_acc, var, wrong_index = compute_accuracy(gts, predictions)

def get_resume_index(args):
    # this will return from which qid we need to resume the experiments
    file_path = f"{args.output}right_aff_{args.dataset}_{args.max_round}_{args.model_config}.json"
    data = correct_json(file_path)
    return data[-1]['qid'] + 1


def compute_gsm8k(args, config):
    # read the GSM8k dataset 
    def read_gsm8k():
        # read the json file
        file_path = os.path.join(args.dataset_root, 'datasets/GSM8k_test.jsonl')
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return data

    # extract the final answer from the response 
    def solve_math_problems(input_str):
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, input_str)
        if matches:
            return int(matches[-1])
        return None

    def extract_answer(ans):
        try:
            pattern = r"\\boxed{([^}]*)}" # Regular expression to find text inside \boxed{}       
            match = re.search(pattern, ans) # Search for the pattern

            if match: ans = match.group(1)
        except: pass
        try:
            return int(float(ans.replace(",", "")))
        except:
            return int(ans.split(':')[0])

    gsm8k_data = read_gsm8k() # read the data
    if args.resume: resume = get_resume_index(args)
    else: resume = 0
    counter = resume
    for ith in tqdm(range(resume, len(gsm8k_data))):
        data = gsm8k_data[ith]
        question = data['question']
        gt = solve_math_problems(data['answer'])

        # debate topic
        message = "Can you solve the following math problem? \n {} \n Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.".format(question)
        # controlled prompt for Affirmative side
        controlled_message = "Can you solve the following math problem? \n {} \n Is {} the correct answer to this problem? Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.".format(question, gt)

        # initiate the debate
        urrent_script_path = os.path.abspath(__file__)
        MAD_path = current_script_path.rsplit("/", 1)[0]
        config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
        config['debate_topic'] = message
        config['affirmative_prompt'] = controlled_message
        debate = Debate(max_round=args.max_round, model_name=args.model_config, apply_max_token=args.apply_max_token, num_players=3, openai_api_key=args.openai_key, config=config, temperature=0, sleep_time=0)
        rounds_happend, affirmative_answers, negative_answers, moderator_answers, final_ans = debate.run()
        ans = extract_answer(final_ans) # get the int answere

        # save the response in a json file
        save_json(args, {
            "qid": int(counter),
            "question": question,
            "affirmative_responses": affirmative_answers,
            "negative_responses": negative_answers,
            "moderator_responses": moderator_answers,
            "rounds_happend": rounds_happend,
            "ground_truth": gt,
            "prediction": ans,
        })
        counter+=1


def compute_piqa(args, config):

    def read_piqa():
        # read the json file and .lst file
        file_path = os.path.join(args.dataset_root, 'datasets/piqa/valid.jsonl')
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        file_path = os.path.join(args.dataset_root, 'datasets/piqa/valid-labels.lst')
        with open(file_path, 'r') as file:
            lines = file.readlines()
            labels = [int(line.strip()) for line in lines]
        return data, labels

    def get_final_answer(input_str):
        # get the final answer from the final GPT response
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, input_str)
        if matches:
            return int(matches[-1])
        return None
    
    questions, labels = read_piqa()
    if args.resume: resume = get_resume_index(args)
    else: resume = 0
    for i in tqdm(range(resume, len(questions))):
        question, label = questions[i], labels[i]
        goal = question['goal']; prompt = f"Given the goal: {goal}\n there are two solutions:\n" # get the question
        solutions = [question['sol1'], question['sol2']] # get the answere
        for ith, sol in enumerate(solutions, 1):
            prompt += f"solution-{ith}. {sol}\n"
        controlled_message = copy.deepcopy(prompt)
        prompt += "\nYou need to choose either solution-1 or solution-2 and explain why that solution is correct? If the final solution is solution-1 then return 0 and if it is solution-2 then return 1 in the form \boxed{{answer}}, at the end of your response."

        # create the controlled message here.
        if label==0: sol_prompt = "solution-1"
        elif label==1: sol_prompt = "solution-2"
        else: raise ValueError("wrong format for label")
        controlled_message+= "\nIs {} the correct solution to this goal? Explain your reasoning. If the final answer is solution-1 then return 0 and if it is solution-2 then return 1 in the form \boxed{{answer}}, at the end of your response.".format(sol_prompt)

        # initiate the debate
        urrent_script_path = os.path.abspath(__file__)
        MAD_path = current_script_path.rsplit("/", 1)[0]
        config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
        config['debate_topic'] = prompt
        config['affirmative_prompt'] = controlled_message
        debate = Debate(max_round=args.max_round, model_name=args.model_config, apply_max_token=args.apply_max_token, num_players=3, openai_api_key=args.openai_key, config=config, temperature=0, sleep_time=0)
        rounds_happend, affirmative_answers, negative_answers, moderator_answers, final_ans = debate.run()
        ans = get_final_answer(final_ans) # get the int answer
        
        # save the response in a json file
        save_json(args, {
            "qid": int(i),
            "question": question,
            "affirmative_responses": affirmative_answers,
            "negative_responses": negative_answers,
            "moderator_responses": moderator_answers,
            "rounds_happend": rounds_happend,
            "ground_truth": label,
            "prediction": ans,
        })


def compute_strategyqa(args, config):
    def read_strategyqa():
        file_path = os.path.join(args.dataset_root, 'datasets/strategyqa_train.json')
        with open(file_path, 'r') as file: 
            data = json.load(file)
        return data

    def get_final_answer(input_str):
        pattern = r'\btrue\b|\bfalse\b'
        matches = re.findall(pattern, input_str, re.IGNORECASE)
        return matches[-1]


    data = read_strategyqa()
    if args.resume: resume = get_resume_index(args)
    else: resume = 0
    for i in tqdm(range(resume, len(data))):
        question, gt = data[i]['question'], data[i]['answer']
        prompt = "Can you give the answer of the following question?\n {} \nAnswer with true or false and explain your reasoning. Your final answer should be either true or false, in the form \boxed{{answer}}, at the end of your response.".format(question)
        
        controlled_message = "Can you give the answer of the following question?\n {} \nIs {} the correct answer to this question? Explain your reasoning. Your final answer should be either true or false, in the form \boxed{{answer}}, at the end of your response.".format(question, gt)
        # initiate the debate
        urrent_script_path = os.path.abspath(__file__)
        MAD_path = current_script_path.rsplit("/", 1)[0]
        config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
        config['debate_topic'] = prompt
        config['affirmative_prompt'] = controlled_message
        debate = Debate(max_round=args.max_round, model_name=args.model_config, apply_max_token=args.apply_max_token, num_players=3, openai_api_key=args.openai_key, config=config, temperature=0, sleep_time=0)
        rounds_happend, affirmative_answers, negative_answers, moderator_answers, final_ans = debate.run()
        ans = get_final_answer(final_ans) # get the final answer format
        ans = False if ans.lower() == "false" else True

        # save the response in a json file
        save_json(args, {
            "qid": int(i),
            "question": question,
            "affirmative_responses": affirmative_answers,
            "negative_responses": negative_answers,
            "moderator_responses": moderator_answers,
            "rounds_happend": rounds_happend,
            "ground_truth": gt,
            "prediction": ans,
        })


def compute_chess_validity(args, config):
    def read_chess():
        file_path = os.path.join(args.dataset_root, 'datasets/chess.jsonl')
        with open(file_path, 'r') as fh:
            return [json.loads(line) for line in fh.readlines() if line]

    def get_final_answer(pred):
        pred = pred.replace("(", "").replace(")", "")
        if len(pred)==2: pass
        else: pred = pred[-2:]
        return pred

    
    data = read_chess()
    if args.resume: resume = get_resume_index(args)
    else: resume = 0
    for i in tqdm(range(resume, len(data))):
        question, answer = data[i]['question'], data[i]['answer']
        prompt = "Given the chess game prefix {} and the starting square of the current move {}, please give one valid destination square for the chess piece at. State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5). Give a one line explanation of why your destination square is a valid move.".format(question[:-3], question[-2:])
        controlled_message = "Given the chess game prefix {question[:-3]} and the starting square of the current move {question[-2:]}, please give one valid destination square for the chess piece at. \n From the list of possible destination squares {answer}, is any square a valid destination? State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5). Give a one line explanation of why your destination square is a valid move."""
        # controlled_message = "Given the chess game prefix {}, the starting square of the current move {}, and the list of possible destination squares ["", "", "", "", "", ""], please determine if any of these squares is a valid destination for the chess piece at the starting square.# initiate the debate
        urrent_script_path = os.path.abspath(__file__)
        MAD_path = current_script_path.rsplit("/", 1)[0]
        config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
        config['debate_topic'] = prompt
        config['affirmative_prompt'] = controlled_message
        debate = Debate(max_round=args.max_round, model_name=args.model_config, apply_max_token=args.apply_max_token, num_players=3, openai_api_key=args.openai_key, config=config, temperature=0, sleep_time=0)
        rounds_happend, affirmative_answers, negative_answers, moderator_answers, final_ans = debate.run()
        ans = get_final_answer(final_ans) # get the final answer format
        
        # save the response in a json file
        save_json(args, {
            "qid": int(i),
            "question": question,
            "affirmative_responses": affirmative_answers,
            "negative_responses": negative_answers,
            "moderator_responses": moderator_answers,
            "rounds_happend": rounds_happend,
            "ground_truth": answer,
            "prediction": ans,
        })
        

def mad(args, config):
    if args.eval: eval(args)
    else:
        # compute mode
        write_file_check(args)
        if args.dataset == "gsm8k": compute_gsm8k(args, config)
        elif args.dataset == "piqa": compute_piqa(args, config)
        elif args.dataset=="strategyqa": compute_strategyqa(args, config)
        elif args.dataset=="chess": compute_chess_validity(args, config)
        

if __name__ == "__main__":
    # create the parser 
    parser = argparse.ArgumentParser(description="Multi-Agent-Debate parser")
    parser.add_argument('--max_round', type=int, default=10, help="number of agenets for ensamble")
    parser.add_argument('--dataset', type=str, help='dataset name', choices=['gsm8k', 'piqa', 'strategyqa', 'chess'])
    parser.add_argument('--dataset_root', type=str, help='dataset root path')
    parser.add_argument('--model_config', type=str, help="OpenAI model configurations", default='gpt-4o', choices=['gpt-4o', 'gpt-4', 'gpt-4o-mini'])
    parser.add_argument('--output', type=str, help="output path for saving the json file", default="./output/MAD/")
    parser.add_argument('--openai_key', type=str, help="OpenAI key for GPT models")
    parser.add_argument('--resume', action='store_true', help="whather you want to resume from where it's left")
    parser.add_argument('--eval', action='store_true', help="add this flag when you want to compute the final accuracy from the json file")
    parser.add_argument('--apply_max_token', action='store_true', help="add this flag to enable MAD as it is")
    args = parser.parse_args() # Parse the arguments

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]
    config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))

    mad(args, config)
