import subprocess
import argparse
import os
from concurrent.futures import ThreadPoolExecutor


def run_1_caption(device_id, index):
    cmd = f"python 1_caption_36.py --index {index} --gpu-id {device_id}"
    with open(os.path.join("logs", str(index)+"_log.txt"), "w") as log_file:
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()


def run_2_panorama(device_id, index):
    cmd = f"CUDA_VISIBLE_DEVICES={device_id} python 2_panorama.py --index {index}"
    with open(os.path.join("logs", str(index)+"_log.txt"), "w") as log_file:
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="run the task")
    parser.add_argument("--devices", type=int, nargs='+', help="gpu id")
    args = parser.parse_args()
    executor = ThreadPoolExecutor(max_workers=len(args.devices))
    futures = []
    for i in range(len(args.devices)):
        if args.task == "caption":
            futures.append(executor.submit(run_1_caption, args.devices[i], i))
        elif args.task == "panorama":
            futures.append(executor.submit(run_2_panorama, args.devices[i], i+40))
        else:
            raise ValueError("ERROR WITH TASK!")
    for i in range(len(args.devices)):
        futures[i].result()
