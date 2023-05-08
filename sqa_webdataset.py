import os
import uuid
import os.path as osp
import base64
import argparse
import webdataset as wds
from read_sqa import SqaConfig, load_data
from tqdm.auto import tqdm


def convert(args: SqaConfig):
    problems, qids = load_data(args)
    dst_dir = osp.join(args.output_dir, args.split)
    os.makedirs(dst_dir, exist_ok=True)
    total_samples = 0
    no_image_samples = 0
    with wds.ShardWriter(dst_dir + "/%09d.tar", maxcount=1000) as sink:
        for qid in tqdm(qids, desc="parsing problems..."):
            total_samples += 1
            sample_data = {}
            problem = problems[qid]
            # prompt = ParseProblem.build_prompt(problem, args)
            image_name = problem["image"]
            image_path = os.path.join(args.images_dir, problem["split"], qid, str(image_name))
            if os.path.exists(image_path) and image_name:
                with open(image_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            else:
                no_image_samples += 1
                image_base64 = ""
            sample_data["problem"] = problem
            sample_data["image_base64"] = image_base64
            key_str = uuid.uuid4().hex
            sink.write({"__key__": key_str, "json": sample_data})
    print(f"Totally there are {total_samples} samples, and there are {no_image_samples} samples have no images in which.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test', 'trainval', 'minitrain', 'minival', 'minitest'])
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = SqaConfig(args.split, args.prompt_format)
    convert(config)


