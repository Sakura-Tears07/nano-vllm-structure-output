from enum import Enum

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pydantic import BaseModel
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

from nanovllm.sampling_params import StructuredOutputsParams


# Structured outputs by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

#json_schema = CarDescription.model_json_schema()
json_schema = {
    **CarDescription.model_json_schema(),
    "required": ["brand", "model", "car_type"]
}

structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
sampling_params_json = SamplingParams(
    temperature=0.1,
    max_tokens=64,
    structured_outputs=structured_outputs_params_json, 
)
prompt_json = [
    "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
    # "please Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
]


def main():
    path = "/data/zy/Small_model/"
    #path = "/data/zy/models/Qwen/Qwen3/Qwen3-8B/"
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True
    )
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024 * 2)
    #prompts = [
    #    "请你用100字简单讲述一下三顾茅庐的故事",
    #    "请你用100字简单讲述一下空城计的故事",
    #]
    
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompt_json
    ]
    outputs = llm.generate(prompt_json, [sampling_params_json] * len(prompt_json))

    for prompt, output in zip(prompt_json, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
