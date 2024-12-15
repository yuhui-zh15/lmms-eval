import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import openai
import requests as url_requests
from accelerate import Accelerator, DistributedType
from pydantic import BaseModel
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger

if API_TYPE == "openai":
    client = openai.OpenAI()
elif API_TYPE == "azure":
    client = openai.AzureOpenAI()


class Step(BaseModel):
    step: str
    explanation_for_confidence: str
    confidence: int  # 1-10


class SimpleResponse(BaseModel):
    steps: List[Step]
    conclusion: str


SIMPLE_QA_PROMPT = """\
Answer the question step by step. This question may be difficult to respond to. However, you need to try your best to answer. If you cannot answer, show me what you think about this question. What aspects are you less certain about? If this question requires a lot of external knowledge, also think about what kind of help you would need. Please summarize the help you need as concisely as possible. Of course, before reaching your final concise conclusion, please describe your entire thought process in detail. Mark each step with your confidence (1-10), and if you are not confident in a step, please explain why.
"""


@register_model("gpt4v_mmmu")
class GPT4V_MMMU(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: Union[str, None] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                img = self.encode_image(visual)
                imgs.append(img)

            image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in imgs]
            simple_response = client.beta.chat.completions.parse(
                model=self.model_version,
                messages=[
                    {
                        "role": "system",
                        "content": SIMPLE_QA_PROMPT,
                    },
                    {"role": "user", "content": [{"type": "text", "text": contexts}] + image_contents},
                ],
                response_format=SimpleResponse,
            )

            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT4V")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
