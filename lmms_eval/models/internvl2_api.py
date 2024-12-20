import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

import logging
import re

from PIL import Image

eval_logger = logging.getLogger("lmms_eval_internvl2_api")


@register_model("internvl2_api")
class InternVL2API(lmms):
    def __init__(
        self,
        api_url: Union[str, None] = None,
        api_token: Union[str, None] = None,
        api_key: Union[str, None] = None,
        timeout: int = 120,
        continual_mode: bool = False,
        # modality: str = "image",
        max_frames_num: int = 32,
        response_persistent_folder: Union[str, None] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.modality = modality
        self.max_frames_num = max_frames_num

        if api_url:
            self.api_url = api_url
        elif "INTERNVL2_API_URL" in os.environ:
            self.api_url = os.getenv("INTERNVL2_API_URL")
        else:
            raise ValueError("Please provide a valid API URL for InternVL2, or set the INTERNVL2_API_URL environment variable.")

        if api_token:
            self.api_token = api_token
        elif "INTERNVL2_API_TOKEN" in os.environ:
            self.api_token = os.getenv("INTERNVL2_API_TOKEN")
        else:
            raise ValueError("Please provide a valid API token for InternVL2, or set the INTERNVL2_API_TOKEN environment variable.")

        if api_key:
            self.api_key = api_key
        elif "INTERNVL2_API_KEY" in os.environ:
            self.api_key = os.getenv("INTERNVL2_API_KEY")
        else:
            raise ValueError("Please provide a valid API key for InternVL2, or set the INTERNVL2_API_KEY environment variable.")

        self.timeout = timeout
        self.continual_mode = continual_mode
        if self.continual_mode:
            pattern = r"/([^/]+_key_api)/"
            match = re.search(pattern, self.api_url)
            if match:
                self.model_version = match.group(1).replace("_key_api", "")
            else:
                print("Model version not found in the API URL. Use internvl2_pro as the default model version.")

            self.model_version = "internvl2_pro"

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

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

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
                if isinstance(visual, Image.Image):
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif isinstance(visual, str):
                    frames = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)

            payload = {"messages": []}

            response_json = {"role": "user", "content": []}

            payload["messages"].append(deepcopy(response_json))
            payload["messages"][0]["content"].append({"type": "text", "text": contexts})
            for img in imgs:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

            # If n image tokens are in the contexts
            # contexts will be splitted into n+1 chunks
            # Manually add it into the payload
            payload["messages"].append(deepcopy(response_json))
            payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]
            payload["api_key"] = self.api_key

            payload["messages"] = json.dumps(payload["messages"])

            for attempt in range(5):
                try:
                    response = url_requests.post(self.api_url, json=payload, timeout=self.timeout, headers={"Authorization": self.api_token})
                    response_data = response.json()

                    response_text = response_data["choices"][0]["message"]["content"].strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = ""

                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                    if attempt <= 5:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty string
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {response.json()}")
                        response_text = ""
            res.append(response_text)
            pbar.update(1)

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
