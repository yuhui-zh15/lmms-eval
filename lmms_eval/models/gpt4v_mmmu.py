import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import crawl4ai
import numpy as np
import openai
import requests as url_requests
from accelerate import Accelerator, DistributedType
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
from duckduckgo_search import DDGS
from googlesearch import search
from pydantic import BaseModel
from tqdm import tqdm
from transformers.pipelines import question_answering

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
    review_this_step: str
    confidence: int  # 1-10

    def __str__(self):
        return f"Step: {self.step}\nConfidence: {self.confidence}\nExplanation for confidence: {self.review_this_step}"


class SimpleResponse(BaseModel):
    steps: list[SimpleResponseStep]
    conclusion: str

    def __str__(self):
        return f"[Steps]\n\n" + "\n\n".join([str(step) for step in self.steps]) + f"\n\n[Conclusion]\n\n{self.conclusion}"


class SearchResponse(BaseModel):
    reason: list[str]
    # need_search: bool
    use_image_search: bool
    search_text: list[str]


class FinalResponse(BaseModel):
    # overall_planning_and_thinking: str
    steps: list[SimpleResponseStep]
    final_answer: str

    def to_dict(self):
        return {
            # "overall_planning_and_thinking": self.overall_planning_and_thinking,
            "thinking": [str(step) for step in self.steps],
            "final_answer": self.final_answer,
        }


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

If you think a certain step is wrong, you can try to search for knowledge about that step. If you think there is a problem with the idea or direction of the whole answer, you can also try to search for similar questions or tutorials to find the direction.

For example: If the model's response shows uncertainty about a specific detail or concept, focus your search on clarifying that uncertainty rather than repeating the original question.

If the reviewer points out gaps in the model's knowledge, target those specific gaps in your search.

The goal is to resolve uncertainties in the model's understanding.

You have 2 options for searching. If you use text search, it will search on google, so you need to provide search_text as concise as possible. If use enabled image search, it will find similar images on the web. Image search can only search similar images, you cannot add any text information. If there are multiple images, it sill search for the first image.

You can provide 1-2 search_text, each search_text will be searched separately.
"""

# If you think there is nothing uncertain, then there is no need to search.

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
You are a agent who needs to provide an answer to the question.

You need to answer the question step by step and write a concise solution (maybe one or two words) in the final answer.

[Question]

{question}
"""

import asyncio
import os
from urllib.parse import urlparse

from playwright.async_api import async_playwright


async def load_all_images(page):
    # 保存当前滚动位置
    original_position = await page.evaluate("() => ({ x: window.scrollX, y: window.scrollY })")

    # 找到所有图片元素
    locators = page.locator("//img")

    # 创建一个 Promise 数组，每个对应于一个图片的加载
    promises = await locators.evaluate_all(
        """
    elements => elements.map(img => {
        if (img.complete) return Promise.resolve();
        return new Promise(resolve => {
            img.onload = resolve;
            img.onerror = resolve;  // 处理加载失败
            // 如果图片没有 src，可能是懒加载的图片
            if (!img.src && img.dataset.src) {
                img.src = img.dataset.src;
            }
        });
    })
    """
    )

    # 等待所有图片加载完成
    await page.evaluate("promises => Promise.all(promises)", promises)

    # 恢复原始滚动位置
    await page.evaluate("position => window.scrollTo(position.x, position.y)", original_position)

    # 给页面一些时间来稳定
    await page.wait_for_timeout(1000)


async def _search_by_image(image_url, delay=10.0, headless=True, max_results=10):
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=["--lang=en-US", "--force-locale=en-US", "--force-ui-locale=en-US"])
        context = await browser.new_context(locale="en-US", viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        image_url = os.path.abspath(image_url)
        parsed_url = urlparse(image_url)

        await page.goto("https://www.google.com/imghp?hl=en&gl=us")
        # await page.wait_for_selector('div[jsname="R5mgy"]', state='visible')
        # await page.click('div[jsname="R5mgy"]')
        await page.wait_for_selector('div[aria-label="Search by image"]', state="visible")
        await page.click('div[aria-label="Search by image"]')

        if parsed_url.scheme in ("http", "https"):
            # 如果是 URL，直接使用图片链接进行搜索
            await page.wait_for_selector('input[placeholder="Paste image link"]', state="visible")
            await page.fill('input[placeholder="Paste image link"]', image_url)
            await page.wait_for_selector('div[jsname="hSRGPd"]', state="visible")
            await page.click('div[jsname="hSRGPd"]')
        else:
            # await page.click('span[jsname="tAPGc"]')
            await page.set_input_files('input[type="file"]', image_url)
            await asyncio.sleep(2.0)

        await load_all_images(page)
        await asyncio.sleep(delay)

        # 提取搜索结果
        result_cards = await page.query_selector_all(".Vd9M6")
        count = 0
        for card in result_cards:
            image_element = await card.query_selector("img.wETe9b")
            snippet_element = await card.query_selector(".UAiK1e")
            a_element = await card.query_selector("a.GZrdsf")

            if image_element and snippet_element and a_element:
                img_url = await image_element.get_attribute("src")
                snippet = await snippet_element.inner_text()
                web_url = await a_element.get_attribute("href")

                if img_url.startswith("data:image"):
                    continue

                results.append({"image_url": img_url, "snippet": snippet, "web_url": web_url})
                count += 1
                if count == max_results:
                    break

        await browser.close()

    return results


def search_by_image(image_url=None, max_results=10):
    return asyncio.run(_search_by_image(image_url, max_results=max_results))


def image_search(image_url, max_results=10):
    googlelens_results = search_by_image(image_url, max_results)
    urls = [url["web_url"] for url in googlelens_results]
    return urls


@register_model("gpt4v_mmmu")
class GPT4V_MMMU(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: Union[str, None] = None,
        with_search: bool = True,
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

        self.with_search = with_search

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

        num_image_search = 0
        num_text_search = 0

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            print(f"Generating for {task} {split} {doc_id}")

            try:
                q = json.loads(contexts)
                field = q["field"]
                contexts = q["question"]
            except Exception as e:
                print(f"Error: {e}")
                field = "Unknown"

            try:
                if self.with_search:
                    search_dir = Path("temp") / "search" / task / split / field / str(doc_id)
                    if not search_dir.exists():
                        search_dir.mkdir(parents=True)
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)
                imgs = []  # multiple images or frames for video
                for visual in visuals:
                    img = self.encode_image(visual)
                    imgs.append(img)

                image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in imgs]
                search_image_contents = []

                if self.with_search:
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
                            max_tokens=2048,
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
                            max_tokens=1024,
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
                            max_tokens=1024,
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
                            indent=4,
                        )

                    # requery.need_search = False
                    # if requery.need_search:
                    # if True:
                    # Search using DuckDuckGo and get first result URL
                    # requery.use_image_search = True
                    if requery.use_image_search:
                        temp_dir = search_dir / "temp"
                        if not temp_dir.exists():
                            temp_dir.mkdir(parents=True)
                        image_url = temp_dir / "image.png"
                        img = visuals[0]
                        img.save(image_url)
                        urls = image_search(image_url, max_results=5)
                        num_image_search += 1
                    else:
                        # urls = search(search_content, num_results=5)
                        urls_list = []
                        for text in search_content:
                            urls_list.append(list(search(text, num_results=5)))
                        urls = []
                        for i in range(5):
                            x = list(range(len(urls_list)))
                            import random

                            random.shuffle(x)
                            for j in x:
                                if len(urls_list[j]) > i:
                                    urls.append(urls_list[j][i])
                        num_text_search += 1

                    if urls:
                        # Take screenshot of the first 3 webpages

                        for url_idx, url in enumerate(urls):
                            if "sciencedirect" in url:
                                continue

                            if len(search_image_contents) >= 4:
                                break

                            # if url.endswith(".pdf"):
                            #     continue

                            try:
                                # Create and run async screenshot capture
                                import asyncio

                                async def capture(screenshot_path, markdown_path=None, return_image=False, return_text=False):
                                    if return_image:
                                        browser_config = BrowserConfig(viewport_width=768, viewport_height=2048)
                                        async with AsyncWebCrawler(config=browser_config) as crawler:
                                            result = await crawler.arun(
                                                url=url,
                                                screenshot=True,
                                                screenshot_wait_for=2.0,
                                                simulate_user=True,
                                                magic=True,
                                                cache_mode=CacheMode.BYPASS,
                                                remove_overlay_elements=True,
                                                excluded_tags=["form", "header", "footer"],
                                            )

                                        if result.screenshot is None:
                                            return

                                        with open(screenshot_path, "wb") as f:
                                            f.write(base64.b64decode(result.screenshot))

                                        if return_image:
                                            img = Image.open(screenshot_path)
                                            if img.height > 2048:
                                                img = img.crop((0, 0, img.width, 2048))
                                                img.save(screenshot_path)

                                            img1 = img.crop((0, 0, img.width, img.height // 2))
                                            img2 = img.crop((0, img.height // 2, img.width, img.height))

                                            # Convert PIL Image to base64
                                            buffered = BytesIO()
                                            img1.save(buffered, format="PNG")
                                            img_str1 = base64.b64encode(buffered.getvalue()).decode()
                                            img2.save(buffered, format="PNG")
                                            img_str2 = base64.b64encode(buffered.getvalue()).decode()

                                            search_image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str1}"}})
                                            search_image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str2}"}})

                                    if return_text:
                                        async with AsyncWebCrawler() as crawler:
                                            result = await crawler.arun(
                                                url=url,
                                                screenshot=True,
                                                screenshot_wait_for=2.0,
                                                simulate_user=True,
                                                magic=True,
                                                cache_mode=CacheMode.BYPASS,
                                                remove_overlay_elements=True,
                                                excluded_tags=["form", "header", "footer"],
                                            )

                                        if "verify" in result.html.lower() and "human" in result.html.lower():
                                            return

                                        search_image_contents.append({"type": "text", "text": result.markdown})

                                        if markdown_path:
                                            with open(markdown_path, "w") as f:
                                                f.write(result.markdown)

                                        # print(result.markdown)

                                screenshot_path = search_dir / f"search_result_{url_idx}.png"
                                markdown_path = search_dir / f"search_result_{url_idx}.md"

                                asyncio.run(capture(screenshot_path, markdown_path, return_image=True, return_text=True))

                                with open(search_dir / f"search_result_url_{url_idx}.txt", "w") as f:
                                    f.write(url)

                            except Exception as e:
                                print(f"Error occurred: {e}")
                                continue

                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": FINAL_PROMPT.format(question=contexts)}]
                        + image_contents
                        + (
                            [
                                {
                                    "type": "text",
                                    "text": "Here are some reference to help you better answer the question. If you cannot see these reference, it may because of some technical issues. Don't worry, you need to still provide an answer.",
                                }
                            ]
                            if search_image_contents
                            else []
                        )
                        + search_image_contents,
                    },
                ]

                final_response = (
                    client.beta.chat.completions.parse(
                        model=self.model_version,
                        messages=messages,
                        response_format=FinalResponse,
                        max_tokens=4096,
                    )
                    .choices[0]
                    .message.parsed
                )

                res.append(final_response.final_answer)

                if self.with_search:
                    with open(search_dir / "final_response.json", "w") as f:
                        json.dump(
                            final_response.to_dict(),
                            f,
                            indent=4,
                        )

                if self.continual_mode is True:  # Cache the response
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    self.response_cache[doc_uuid] = final_response.final_answer
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(self.response_cache, f)

                print(f"Image search: {num_image_search}, Text search: {num_text_search}")

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
