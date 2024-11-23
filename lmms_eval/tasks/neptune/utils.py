import json
import os
import sys
from glob import glob
from pathlib import Path

import yaml
from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "neptune_full.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def neptune_full_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_path"].split(".")[0] + "*.mp4"
    video_path = os.path.join(cache_dir, video_path)
    video_path = [f for f in glob(video_path) if "temp" not in f]
    if len(video_path) > 1:
        return video_path[:1]
    elif len(video_path) > 0:
        return video_path
    else:
        # Some stupid hardcode to skip this
        return [f"video path:{video_path} does not exist, please check"]


def neptune_full_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def neptune_full_process_results(doc, results):
    pred = results[0]
    key_id = doc["key"]
    return {"submission": {"key": key_id, "answer": pred}}


def neptune_full_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("neptune_full_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
