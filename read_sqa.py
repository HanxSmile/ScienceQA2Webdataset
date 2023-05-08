import json
from dataclasses import dataclass
import os.path as osp


@dataclass
class SqaConfig:
    split: str
    prompt_format: str
    data_root: str = r"/mnt/lustre/hanxiao/input/scienceqa"
    problems_path: str = osp.join(data_root, "ScienceQA/data/scienceqa/problems.json")
    pid_split_path: str = osp.join(data_root, "ScienceQA/data/scienceqa/pid_splits.json")
    captions_path: str = osp.join(data_root, "ScienceQA/data/captions.json")
    images_dir: str = osp.join(data_root, "images")
    output_dir: str = osp.join(data_root, "webdataset")


def load_data(args: SqaConfig):
    problems = json.load(open(args.problems_path))
    pid_splits = json.load(open(args.pid_split_path))
    captions = json.load(open(args.captions_path))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    qids = pid_splits[args.split]
    print(f"number of chosen problems: {len(qids)}\n")
    return problems, qids


class ParseProblem:
    USE_CAPTION = True
    OPTIONS = ("A", "B", "C", "D", "E")
    PROMPT_TEMPLATE = [
        'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
        'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
        'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
    ]

    @staticmethod
    def get_question_text(problem):
        question = problem['question']
        return question

    @staticmethod
    def get_context_text(problem, use_caption=USE_CAPTION):
        txt_context = problem['hint']
        img_context = problem['caption'] if use_caption else ""
        context = " ".join([txt_context, img_context]).strip()
        if context == "":
            context = "N/A"
        return context

    @staticmethod
    def get_choice_text(probelm, options=OPTIONS):
        choices = probelm['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        # print(choice_txt)
        return choice_txt

    @staticmethod
    def get_answer(problem, options=OPTIONS):
        return options[problem['answer']]

    @staticmethod
    def get_lecture_text(problem):
        # \\n: GPT-3 can generate the lecture with more tokens.
        lecture = problem['lecture'].replace("\n", "\\n")
        return lecture

    @staticmethod
    def get_solution_text(problem):
        # \\n: GPT-3 can generate the solution with more tokens
        solution = problem['solution'].replace("\n", "\\n")
        return solution

    @staticmethod
    def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=False):

        input_format, output_format = format.split("-")

        ## Inputs
        input, output = "", ""
        if input_format == "CQM":
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
        elif input_format == "QCM":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
        # upper bound experiment
        elif input_format == "QCML":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
        elif input_format == "QCME":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
        elif input_format == "QCMLE":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

        elif input_format == "QCLM":
            input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
        elif input_format == "QCEM":
            input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
        elif input_format == "QCLEM":
            input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

        # Outputs
        if test_example:
            output = "Answer:"
        elif output_format == 'A':
            output = f"Answer: The answer is {answer}."
        elif output_format == 'AL':
            output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
        elif output_format == 'AE':
            output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
        elif output_format == 'ALE':
            output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
        elif output_format == 'AEL':
            output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

        elif output_format == 'LA':
            output = f"Answer: {lecture} The answer is {answer}."
        elif output_format == 'EA':
            output = f"Answer: {solution} The answer is {answer}."
        elif output_format == 'LEA':
            output = f"Answer: {lecture} {solution} The answer is {answer}."
        elif output_format == 'ELA':
            output = f"Answer: {solution} {lecture} The answer is {answer}."

        text = input + output
        text = text.replace("  ", " ").strip()
        if text.endswith("BECAUSE:"):
            text = text.replace("BECAUSE:", "").strip()
        return text

    @classmethod
    def build_prompt(cls, problem, args: SqaConfig):
        assert args.prompt_format in cls.PROMPT_TEMPLATE
        question = cls.get_question_text(problem)
        context = cls.get_context_text(problem)
        choice = cls.get_choice_text(problem)
        answer = cls.get_answer(problem)
        lecture = cls.get_lecture_text(problem)
        solution = cls.get_solution_text(problem)

        train_example = cls.create_one_example(args.prompt_format,
                                               question,
                                               context,
                                               choice,
                                               answer,
                                               lecture,
                                               solution,
                                               test_example=False)

        return train_example
