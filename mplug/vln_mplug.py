import argparse
import sys
import os
# import ruamel_yaml as yaml
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_caption_mplug_vatex import MPLUG
from models.vit import resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vln_collate_fn


def generate_test_results(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating VLN datasets")
    datasets = [create_dataset('vln', config)[1]]
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [False], num_tasks, global_rank)
    else:
        samplers = [None, ]
    test_loader = create_loader(datasets, samplers,
                                batch_size=[config['batch_size_test']],
                                num_workers=[8], is_trains=[False], collate_fns=[vln_collate_fn])[0]

    #### Model ####
    print("Creating model")
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = MPLUG(config=config, tokenizer=tokenizer)
    model = model.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']
        # reshape positional embedding to accomodate for image resolution change
        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                               pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
        for key in list(state_dict.keys()):
            if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                encoder_key = key.replace('fusion.', '').replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        import apex
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_without_ddp = model.module

    print("Start evaluating")
    vqa_result = evaluation(model, test_loader, tokenizer, device, config, test_submit=True)
    result_file = save_result(vqa_result, args.result_dir, 'speaker_epoch4_prevalent_no_prompt')
    dist.barrier()


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, test_submit=False):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate Vatex Cap test result:'
    print_freq = 2
    
    result = []

    answer_input = None
    for n, (video, video_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if config['prompt'] != "":
            caption = [config['prompt'] + config['eos']] * video.size(0)
            caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length,
                                return_tensors="pt").to(device)
        else:
            caption = None
        # print (caption.input_ids.size())
        # image = image.to(device,non_blocking=True)

        topk_ids, topk_probs = model(video, caption, None, train=False, device=device)
        
        for image_id, topk_id, topk_prob in zip(video_ids, topk_ids, topk_probs):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            ans += ' .'
            # if test_submit:
                # print (image_id, int(image_id.replace(".jpg", "").split("_")[-1]))
            result.append({image_id: ans})
            # else:
            #     result.append({"question_id": image_id, "pred_caption": ans, "gold_caption": gold_caption_list})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='mPLUG/configs/vln_speaker_inference.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/speaker/checkpoint_04.pth')
    parser.add_argument('--output_dir', default='results/instruct_delete')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='checkpoints/bert-base-uncased')
    parser.add_argument('--text_decoder', default='checkpoints/bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=30, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--lr1', default=2e-5, type=float)
    parser.add_argument('--lr2', default=5e-6, type=float)
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = yaml.YAML(typ='rt').load(open(args.config, 'r'))

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    yaml.YAML(typ='unsafe', pure=True).dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))


    # main(args, config)
    generate_test_results(args, config)
