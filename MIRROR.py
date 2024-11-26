import os
import time
import json
import random
import requests
import argparse
from tqdm import tqdm
from termcolor import colored
import multiprocessing as mp
from base_agent import BaseAgent
from server import get_rapidapi_response
from prompt import (
    PLAN_AGENT_SYSTEM_PROMPT, PLAN_AGENT_USER_PROMPT,
    TOOL_AGENT_SYSTEM_PROMPT, TOOL_AGENT_USER_PROMPT,
    ANSWER_AGENT_SYSTEM_PROMPT, ANSWER_AGENT_USER_PROMPT,
    LONG_MEMORY_REFLECTION_TEMPLATE, SHORT_MEMORY_REFLECTION_TEMPLATE
)
from utils import (
    change_name, standardize, get_white_list,
    generate_task_list, fetch_api_json, api_json_to_openai_json
)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_query_id_dir', type=str, default="./solvable_queries/test_query_ids", required=False, help='test query ids for different test sets')
    parser.add_argument('--query_dir', type=str, default="./solvable_queries/test_instruction", required=False, help='the directory that contains test sets')
    parser.add_argument('--answer_dir', type=str, default="./data/answer_gpt35", required=False, help='the directory that contains test sets')
    parser.add_argument('--method', type=str, default="mirror", required=False, help='method')
    parser.add_argument('--model_name', type=str, default="gpt-4o-2024-05-13", required=False, help='the model name for the vllm model')
    parser.add_argument('--tool_root_dir', type=str, default="./server/tools", required=False, help='tool environment for the toolbench')
    parser.add_argument('--toolbench_key', type=str, default="",required=False, help='your toolbench key to request rapidapi service')
    parser.add_argument('--rapidapi_key', type=str, default="",required=False, help='your rapidapi key to request rapidapi service')
    parser.add_argument('--use_rapidapi_key', action="store_true", help="To use customized rapidapi service or not.")
    parser.add_argument('--api_customization', action="store_true", help="To use customized api or not.")
    parser.add_argument('--max_observation_length', type=int, default=1024, required=False, help='maximum observation length')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], required=False, help='observation compress method')
    parser.add_argument('--num_process', type=int, default=5, required=False, help='number of processes')
    parser.add_argument('--max_round', type=int, default=5, required=False, help='number of round')
    parser.add_argument('--max_step', type=int, default=3, required=False, help='number of step')
    parser.add_argument('--service_url', type=str, default="http://localhost:8080/virtual", required=False, help='local stabletoolbench service url')
    parser.add_argument('--service_type', type=str, default="stabletoolbench", required=False, help='')
    parser.add_argument('--test_set', nargs='+', default=['G2_category'], help='test set name')

    args = parser.parse_args()
    return args


def parse_json_response(response_str, error_msg):
    try:
        response = json.loads(response_str.replace("`", "").replace("json", ""))
        return response
    except json.JSONDecodeError:
        print(f"{error_msg} - JSON error")
    except Exception as e:
        print(f"{error_msg} - Unknown error: {str(e)}")
    return None

def call_rapidapi(action_name, action_input, functions, api_name_reflect, cate_names, tool_names, args):
    backoff = 1  # Initial backoff time
    for k, function in enumerate(functions):
        if function["name"].endswith(action_name):
            pure_api_name = api_name_reflect[function["name"]]
            payload = {
                "category": cate_names[k],
                "tool_name": tool_names[k],
                "api_name": pure_api_name,
                "tool_input": action_input,
                "strip": args.observ_compress_method,
                "toolbench_key": args.toolbench_key
            }
            print(colored(f"query to {cate_names[k]}-->{tool_names[k]}-->{action_name}", color="yellow"))
            if args.use_rapidapi_key or args.api_customization:
                payload["rapidapi_key"] = args.rapidapi_key
                response = get_rapidapi_response(payload, api_customization=args.api_customization)
            else:
                time.sleep(3)  # rate limit: 30 per minute
                headers = {"toolbench_key": args.toolbench_key}
                timeout = None if args.service_url.endswith("virtual") else 15
                try:
                    response = requests.post(args.service_url, json=payload, headers=headers, timeout=timeout)
                except requests.exceptions.Timeout:
                    return json.dumps({"error": f"Timeout error...", "response": ""}), 5
                if response.status_code != 200:
                    return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
                try:
                    response = response.json()
                except json.JSONDecodeError:
                    return json.dumps({"error": "JSON decoding error from the response."}), 12
                except Exception as e:
                    return json.dumps({"error": f"Unexpected error: {str(e)}"}), 12

            if response["error"] == "API not working error...":
                status_code = 6
            elif response["error"] == "Unauthorized error...":
                status_code = 7
            elif response["error"] == "Unsubscribed error...":
                status_code = 8
            elif response["error"] == "Too many requests error...":
                status_code = 9
            elif response["error"] == "Rate limit per minute error...":
                print(f"Rate limit reached, sleeping for {backoff} seconds.")
                time.sleep(backoff)
                backoff *= 2  # Exponentially increase backoff time
                if backoff > 64:  # Cap backoff time
                    break
                status_code = 10
            elif response["error"] == "Message error...":
                status_code = 11
            else:
                status_code = 0
            return json.dumps(response), status_code

    return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1

def process_task(task_with_args):    
    global base_agent 
    task, args = task_with_args
    
    query_id = task[2]
    output_dir_path = task[-2]
    os.makedirs(output_dir_path, exist_ok=True)
    output_file_path = os.path.join(output_dir_path, f"{query_id}_{args.method}.json")
    data_dict = task[-3]
    tool_descriptions = task[-1]
    query = data_dict["query"]
    print(colored(f"[{os.getpid()}] query --> {query}", color="red"))

    tool_names = []
    cate_names = []
    functions = []
    api_name_reflect = {}

    data_dict = fetch_api_json(data_dict, tool_root_dir=args.tool_root_dir)
    if not data_dict["api_list"]:
        print("No available api")
        return

    for idx_tool, api_json in enumerate(data_dict["api_list"]):
        standard_tool_name = tool_descriptions[idx_tool][0]
        openai_function_json, cate_name, pure_api_name = api_json_to_openai_json(api_json, standard_tool_name)
        functions.append(openai_function_json)
        api_name_reflect[openai_function_json["name"]] = pure_api_name
        tool_names.append(standard_tool_name)
        cate_names.append(cate_name)

    round_index = 0
    long_memory = ""
    total_steps = 0
    best_answer = ""
    best_plan = {}
    best_reflection = {}
    max_score = -1
    best_trajectory = []
    task_completed = False 
    
    plan_scores, tool_scores, answer_scores = [], [], []
    while round_index < args.max_round and not task_completed:
        trajectory = []
        plan = None
        plan_current_score = -1
        plan_max_score = -1
        plan_best = None
        for plan_step in range(args.max_step):
            planner_agent_system_prompt = PLAN_AGENT_SYSTEM_PROMPT
            planner_agent_user_prompt = PLAN_AGENT_USER_PROMPT.format(
                task_description=query,
                functions=functions,
                long_memory=long_memory
            )
            plan_response = base_agent.query_openai(
                system_prompt=planner_agent_system_prompt,
                user_prompt=planner_agent_user_prompt,
                json_mode=True
            )
            total_steps += 1
            plan = parse_json_response(plan_response, "PLANNER---ERROR")
            if plan and "nodes" in plan and "score" in plan["self_reflection"]:
                print(f'PLANNER AGENT: {plan}')
                plan_current_score = plan["self_reflection"]["score"]

                plan_scores.append(plan_current_score)

                if plan_current_score > plan_max_score:
                    plan_max_score = plan_current_score
                    plan_best = plan
            if plan_current_score >= 9:
                break
        plan = plan_best
        if plan_best is None:
            round_index += 1
            continue

        print(colored(f"[{os.getpid()}] plan --> {plan}", color="light_yellow"))
        subtasks = [t["subtask"] for t in plan["nodes"]]
        print(colored(f"[{os.getpid()}] subtasks --> {subtasks}", color="light_red"))

        outputs = {}
        related_outputs = []
        
        for idx, subtask in enumerate(subtasks):
            short_memory = ""
            exe_current_score = -1
            exe_max_score = -1
            exe_best = None
            for tool_step in range(args.max_step):
                executor_agent_system_prompt = TOOL_AGENT_SYSTEM_PROMPT
                executor_agent_user_prompt = TOOL_AGENT_USER_PROMPT.format(
                    subtask=subtask,
                    functions=functions,
                    related_outputs=related_outputs,
                    short_memory=short_memory
                )
                execution_response = base_agent.query_openai(
                    system_prompt=executor_agent_system_prompt,
                    user_prompt=executor_agent_user_prompt,
                    json_mode=True
                )
                total_steps += 1
                execution = parse_json_response(execution_response, "EXECUTOR---ERROR")
                if execution and "self_reflection" in execution and "score" in execution["self_reflection"] and "function" in execution and "parameters" in execution:
                    exe_current_score = execution["self_reflection"]["score"]
                    print(f"TOOL AGENT: {execution}")
                    tool_scores.append(exe_current_score)

                    if exe_current_score > exe_max_score:
                        exe_max_score = exe_current_score
                        exe_best = execution
                    
                if exe_max_score >= 9:
                    execution = exe_best
                    action = execution["function"]
                    action_input = execution["parameters"]
                    if action is None or type(action)== list:
                        continue
                    print(colored(f"[{os.getpid()}] action --> {action} --> {action_input}", color="light_cyan"))
                    observation_str, exe_status = call_rapidapi(
                        action_name=action,
                        action_input=action_input,
                        functions=functions,
                        api_name_reflect=api_name_reflect,
                        cate_names=cate_names,
                        tool_names=tool_names,
                        args=args
                    )

                    observation_json = json.loads(observation_str)
                    observation = observation_json.get("response", "")
                    observation = observation if isinstance(observation, str) else json.dumps(observation)
                    print(colored(f"[{os.getpid()}] observation --> {observation}", color="cyan"))

                    observation = observation[:1000] if len(observation) >= 1000 else observation

                    if exe_status == 0 or tool_step >= args.max_step - 1:
                        plan["nodes"][idx]["status"] = 1 if exe_status == 0 else 0
                        outputs[plan["nodes"][idx]["id"]] = observation
                        trajectory.append({
                            "subtask": subtask,
                            "status": plan["nodes"][idx]["status"],
                            "action": action,
                            "action_input": action_input,
                            "observation": observation
                        })
                        related_outputs.append({
                            "subtask": subtask,
                            "observation": observation
                        })
                        break
                    else:
                        short_memory += SHORT_MEMORY_REFLECTION_TEMPLATE.format(
                            step=tool_step+1,
                            subtask=subtask,
                            action=action,
                            action_input=action_input,
                            observation=observation,
                            reflection=""
                        ) + "\n\n"
                else:
                    execution = exe_best
                    reflection = execution["self_reflection"] if execution and "self_reflection" in execution else ""
                    action = execution["function"] if execution and "function" in execution else ""
                    action_input = execution["parameters"] if execution and "parameters" in execution else ""
                    short_memory += SHORT_MEMORY_REFLECTION_TEMPLATE.format(
                        step=tool_step+1,
                        subtask=subtask,
                        action=action,
                        action_input=action_input,
                        observation="",
                        reflection=reflection
                    ) + "\n\n"
        
        answer_step = 0
        for _ in range(args.max_step):
            answer_agent_system_prompt = ANSWER_AGENT_SYSTEM_PROMPT
            answer_agent_user_prompt = ANSWER_AGENT_USER_PROMPT.format(
                task_description=query,
                trajectory=trajectory
            )
            answer_response = base_agent.query_openai(
                system_prompt=answer_agent_system_prompt,
                user_prompt=answer_agent_user_prompt,
                json_mode=True
            )
            total_steps += 1
            answer = parse_json_response(answer_response, "ANSWER_AGENT---ERROR")

            if answer and "self_reflection" in answer and "score" in answer["self_reflection"]:
                print(colored(f"[step {answer_step}] current answer --> {answer}", color="light_magenta"))
                answer_step += 1
                print(f"ANSWER AGENT: {answer}")
                current_score = answer["self_reflection"]["score"]
                answer_scores.append(current_score)
                if current_score > max_score:
                    max_score = current_score
                    best_answer = answer["answer"]
                    best_trajectory = trajectory
                    best_plan = plan
                    best_reflection = answer["self_reflection"]
                    
                if current_score >= 9:
                    task_completed = True
                    break
            else:
                print("JSON format error")
        
        if task_completed:
            break
        else:
            round_index += 1
            long_memory += LONG_MEMORY_REFLECTION_TEMPLATE.format(
                round_index=round_index,
                trajectory=best_trajectory,
                reflection=best_reflection
            ) + "\n"

    out_line = {
        "query_id": query_id,
        "query": query,
        "available_tools": functions,
        "answer": {
            "method": args.method,
            "total_steps": total_steps,
            "final_answer": best_answer,
            "answer_details": best_trajectory,
        }   
    }

    with open(output_file_path, 'w', encoding='utf-8') as fp:
        json.dump(out_line, fp, ensure_ascii=False, indent=2)
    
    return out_line

def init_process(args):
    global base_agent
    base_agent = BaseAgent(model=args.model_name)

def main(num_processes):
    args = parse_arg()
        
    all_tasks = []
    for group in args.test_set:
        query_path = f'{args.query_dir}/{group}.json'
        answer_path = f"{args.answer_dir}/{group}"
        task_list = generate_task_list(query_path, answer_path, args.tool_root_dir, args.method, args.model_name)
        random.seed(42)
        random.shuffle(task_list)
        new_task_list = [task for task in task_list if not os.path.exists(os.path.join(task[-2], f"{task[2]}_{args.method}.json"))]
        all_tasks.extend(new_task_list)

    print(f"Total tasks: {len(all_tasks)}")

    tasks_with_args = [(task, args) for task in all_tasks]

    pool = mp.Pool(num_processes, initializer=init_process, initargs=(args,))

    results = []
    for result in tqdm(pool.imap_unordered(process_task, tasks_with_args), total=len(tasks_with_args)):
        results.append(result)
    
    pool.close()
    pool.join()

    print(f"Already processed {len(results)} task")

if __name__ == '__main__':
    args = parse_arg()
    num_processes = args.num_process 
    main(num_processes)
