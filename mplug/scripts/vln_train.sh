#pip install colorlog
#apt-get update
#apt-get install default-jdk
#apt-get install default-jre
#pip install git+git://github.com/j-min/language-evaluation@master
#python -c "import language_evaluation; language_evaluation.download('coco')"


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=3223  --use_env vln_mplug.py \
    --config ./configs/vln_speaker.yaml \
    --output_dir checkpoints/speaker \
    --checkpoint checkpoints/mplug_base.pth \
    --do_two_optim \
    --min_length 15 \
    --beam_size 10 \
    --max_length 80 \
    --max_input_length 25 \
    --do_amp
