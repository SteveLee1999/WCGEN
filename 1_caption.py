import os
import json
import torch
import operator
import argparse
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


INPUT_DIR = "datasets/Matterport3D/rgb"
CAPTION_PATH = "captions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
MODEL = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16, device_map="auto"
)
QUESTION = "how can you best describe this image?"


def generate_caption_single(path):
    image = Image.open(path).convert('RGB')
    inputs = PROCESSOR(image, QUESTION, return_tensors="pt").to(DEVICE, torch.float16)
    out = MODEL.generate(**inputs)
    prompt = PROCESSOR.decode(out[0], skip_special_tokens=True)
    return prompt


def generate_caption_batch(index, gap=1000):
    results = dict()
    for scan in os.listdir(INPUT_DIR):
        for vp in os.listdir(os.path.join(INPUT_DIR, scan)):
            for idx in range(36):
                caption = generate_caption_single(os.path.join(INPUT_DIR, scan, vp, str(idx)+".jpg"))
                results[scan+"_"+vp+"_"+str(idx)] = caption
                with open(CAPTION_PATH, "w") as f:
                    json.dump(dict(sorted(results.items(), key=operator.itemgetter(0))), f, indent=4)
                    f.close()
                exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int, help="index for task")
    args = parser.parse_args()
    generate_caption_batch(args.index)
