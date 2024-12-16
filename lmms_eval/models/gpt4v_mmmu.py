import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import openai
import requests as url_requests
from accelerate import Accelerator, DistributedType
from duckduckgo_search import DDGS
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


class SimpleResponseStep(BaseModel):
    step: str
    explanation_for_confidence: str
    confidence: int  # 1-10

    def __str__(self):
        return f"Step: {self.step}\nConfidence: {self.confidence}\nExplanation for confidence: {self.explanation_for_confidence}"


class SimpleResponse(BaseModel):
    steps: list[SimpleResponseStep]
    conclusion: str

    def __str__(self):
        return f"[Steps]\n\n" + "\n\n".join([str(step) for step in self.steps]) + f"\n\n[Conclusion]\n\n{self.conclusion}"


class SearchResponse(BaseModel):
    reason: list[str]
    # use_image_search: bool
    search_text: str


class FinalResponse(BaseModel):
    steps: list[str]
    result: str


SIMPLE_QA_PROMPT = """\
Answer the question step by step. This question may be difficult to respond to. However, you need to try your best to answer. If you cannot answer, show me what you think about this question. What aspects are you less certain about? If this question requires a lot of external knowledge, also think about what kind of help you would need. Please summarize the help you need as concisely as possible. Of course, before reaching your final concise conclusion, please describe your entire thought process in detail. Mark each step with your confidence (1-10), and if you are not confident in a step, please explain why.
"""

REVIEW_PROMPT_SYSTEM = """\
You are a reviewer. I will provide you with a model's response to a question. The model may express uncertainty in some areas while providing definitive answers in others. Please carefully analyze the response and:

1. Identify any potential inaccuracies or questionable statements
2. List specific points that require further discussion or clarification to reach a solid conclusion
3. Evaluate the model's reasoning process and assumptions
4. Suggest areas where additional context or information would be helpful

Please structure your review with clear sections for:
- Potential Inaccuracies
- Points Needing Clarification
- Assessment of Reasoning
- Additional Context Needed

Be specific in your critique and explain your reasoning for each point raised.
"""

REVIEW_PROMPT = """\
## Question

{question}

## Model's Response

{response}
"""

REQUERY_PROMPT_SYSTEM = """\
You are a web search agent. Your goal is to guide a model step by step in answering a question. You have one opportunity to perform an online search. Based on the question, the model's response, and the reviewer's feedback, you need to decide whether to assist the model by conducting a search to help it provide a better answer. You must analyze both the model's response and the reviewer's feedback to determine what information is uncertain or missing. The reviewer's feedback provides an additional perspective on potential gaps or issues in the model's response. If you still cannot identify the uncertainties after considering both perspectives, you should rely on your own experience to understand what information is missing. Use this comprehensive understanding to perform the search.

Note: Your search should address the model's uncertainties rather than simply searching the question directly.

For example:
If the model's response shows uncertainty about a specific detail or concept, focus your search on clarifying that uncertainty rather than repeating the original question.

If the reviewer points out gaps in the model's knowledge, target those specific gaps in your search.

The goal is to resolve uncertainties in the model's understanding.
"""

REQUERY_PROMPT = """\
Now, here is the query and base model's response:

[Query]
{question}

[Base Model's Response]
{response}

[Review Response]
{review}
"""

FINAL_PROMPT = """\
Based on the search results, please provide a more accurate answer to the question. The base model's response may contain inaccuracies or errors that need to be corrected. Please critically evaluate the base response and use the search results to verify, correct, or enhance the answer. Please provide a detailed explanation of your answer, highlighting any corrections or improvements made to the base response.

If you cannot see the search results, it may because of some technical issues. Don't worry, you need to still provide an answer based on the base model's response and the review feedback.

[Query]

{question}

[Base Model's Response]

{response}

[Review Response]

{review}
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
                    if response_text and len(response_text) < 16:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            try:
                search_dir = Path("temp") / "search" / task / split / str(doc_id)
                if not search_dir.exists():
                    search_dir.mkdir(parents=True)
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)
                imgs = []  # multiple images or frames for video
                for visual in visuals:
                    img = self.encode_image(visual)
                    imgs.append(img)

                image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in imgs]
                simple_response = (
                    client.beta.chat.completions.parse(
                        model=self.model_version,
                        messages=[
                            {
                                "role": "system",
                                "content": SIMPLE_QA_PROMPT,
                            },
                            {"role": "user", "content": [{"type": "text", "text": contexts}] + image_contents},
                        ],
                        response_format=SimpleResponse,
                        max_tokens=4096,
                    )
                    .choices[0]
                    .message.parsed
                )

                review = (
                    client.chat.completions.create(
                        model=self.model_version,
                        messages=[
                            {
                                "role": "system",
                                "content": REVIEW_PROMPT_SYSTEM,
                            },
                            {"role": "user", "content": [{"type": "text", "text": REVIEW_PROMPT.format(question=contexts, response=str(simple_response))}] + image_contents},
                        ],
                        max_tokens=4096,
                    )
                    .choices[0]
                    .message.content
                )

                requery = (
                    client.beta.chat.completions.parse(
                        model=self.model_version,
                        messages=[
                            {
                                "role": "system",
                                "content": REQUERY_PROMPT_SYSTEM,
                            },
                            {"role": "user", "content": [{"type": "text", "text": REQUERY_PROMPT.format(question=contexts, response=str(simple_response), review=review)}] + image_contents},
                        ],
                        response_format=SearchResponse,
                        max_tokens=4096,
                    )
                    .choices[0]
                    .message.parsed
                )

                search_content = requery.search_text

                with open(search_dir / "search_content.json", "w") as f:
                    json.dump(
                        {
                            "doc_id": doc_id,
                            "contexts": contexts,
                            "search_content": search_content,
                            "simple_response": str(simple_response),
                            "review": review,
                        },
                        f,
                    )

                # Search using DuckDuckGo and get first result URL
                ddgs = DDGS(timeout=50)
                news_results = ddgs.text(keywords=search_content, region="wt-wt", safesearch="off", timelimit="m", max_results=10)
                urls = [news["href"] for news in news_results]

                if urls:
                    # Take screenshot of the first 3 webpages
                    import selenium.webdriver
                    from selenium.webdriver.chrome.options import Options

                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    chrome_options.add_argument("--window-size=1024,1024")

                    search_image_contents = []

                    for url_idx, url in enumerate(urls[:3]):
                        try:
                            driver = selenium.webdriver.Chrome(options=chrome_options)
                            driver.get(url)

                            # Take 3 screenshots while scrolling down
                            for i in range(3):
                                # Scroll down
                                if i > 0:
                                    driver.execute_script(f"window.scrollTo(0, {1024 * i})")
                                    time.sleep(1)  # Wait for content to load

                                screenshot_path = search_dir / f"search_result_{url_idx}_{i}.png"

                                driver.save_screenshot(screenshot_path)

                                # Load and encode screenshot
                                with open(screenshot_path, "rb") as f:
                                    screenshot_bytes = f.read()
                                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                                search_image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}})

                            driver.quit()
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            continue

                final_response = (
                    client.beta.chat.completions.parse(
                        model=self.model_version,
                        messages=[
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": FINAL_PROMPT.format(question=contexts, response=str(simple_response), review=review, requery=requery)}]
                                + [{"type": "text", "text": "Here is the image for the query."}]
                                + image_contents
                                + [{"type": "text", "text": "Here are the search results."}]
                                if search_image_contents
                                else [] + search_image_contents + [{"type": "text", "text": f'In "result", you need to directly answer the question as concise as possible: {contexts}'}],
                            },
                        ],
                        response_format=FinalResponse,
                        max_tokens=4096,
                    )
                    .choices[0]
                    .message.parsed
                )

                res.append(final_response.result)

                if self.continual_mode is True:  # Cache the response
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    self.response_cache[doc_uuid] = final_response.result
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(self.response_cache, f)

            except Exception as e:
                eval_logger.error(f"Error occurred: {e}")
                res.append("")

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT4V")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
