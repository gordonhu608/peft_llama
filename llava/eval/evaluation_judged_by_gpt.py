import argparse
import json
import os

import openai
import tqdm
import ray
import time

@ray.remote(num_cpus=4)
def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(1)

    print('success!')
    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question', default="/home/shawn/nvme/vl_research/peft_llama/data/OwlEval/questions.jsonl")
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-a', '--answer-list', nargs='+', default=["/home/shawn/nvme/vl_research/peft_llama/data/OwlEval/answer/mPLUG_Owl_7b_answer.jsonl", \
                                                                    "/home/shawn/nvme/vl_research/peft_llama/data/OwlEval/answer/llava_13b_answer.jsonl"])
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))

    review_file = open(f'{args.output}', 'w')

    js_list = []
    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        # if idx == 1:
        #     break

        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        prompt = "We would like to request your feedback on the performance of two AI assistants in response to the \
            user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses.\
                Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall \
                    performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. \
                        The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, \
                            avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
        role = "Assistant"
        content = (f'[Question]\n{ques["question"]}\n\n'
                   f'[{role} 1]\n{ans1["answer"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["answer"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        js_list.append({
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1['question_id'],
            'answer2_id': ans2['question_id']})
        idx += 1
        handles.append(get_eval.remote(content, args.max_tokens))
        # To avoid the rate limit set by OpenAI
        time.sleep(1)

    reviews = ray.get(handles)
    for idx, review in enumerate(reviews):
        scores = parse_score(review)
        js_list[idx]['content'] = review
        js_list[idx]['tuple'] = scores
        review_file.write(json.dumps(js_list[idx]) + '\n')
    review_file.close()
