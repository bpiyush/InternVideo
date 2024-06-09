"""
Evaluates on SSv2.

cd InternVideo2/multi_modality/demo
python timebound_eval_ssv2.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from glob import glob
from tqdm import tqdm
import pandas as pd

repo_path = os.path.dirname(os.path.abspath("."))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import numpy as np
import os
import io
import cv2

import torch

from config import (
    Config,
    eval_dict_leaf,
)

from demo_utils import (
    retrieve_text,
    _frame_from_video,
    setup_internvideo2,
    frames2tensor,
    compute_features,
)


def num_params(model):
    n = np.sum([p.numel() for p in model.parameters()]) / 1e6
    print(f"Number of parameters in {type(model).__name__}: {np.round(n, 3)}M")


def get_video_path(video_dir, video_id, ext="webm"):
    paths = glob(os.path.join(video_dir, f"*/{video_id}.{ext}"))
    assert len(paths) == 1
    return paths[0]


def text_correct(sim):
    """
    Given a 2x2 similarity matrix, computes text score.

    Based on WinoGround's evaluation code.
    """
    return sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0]


def video_correct(sim):
    """
    Given a 2x2 similarity matrix, computes video score.

    Based on WinoGround's evaluation code.
    """
    return sim[0, 0] > sim[1, 0] and sim[1, 1] > sim[0, 1]


def group_correct(sim):
    """
    Given a 2x2 similarity matrix, computes group score.

    Based on WinoGround's evaluation code.
    """
    return text_correct(sim) and video_correct(sim)


if __name__ == "__main__":
    # Define config
    config = Config.from_file('internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)

    # Define path to pre-trained checkpoints
    ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/InternVideo/"
    model_pth = os.path.join(ckpt_root, "InternVideo2-stage2_1b-224p-f4.pt")
    assert os.path.exists(model_pth)
    config["model"]["vision_encoder"]["pretrained"] = model_pth

    # config["num_frames"] = 8
    # config['num_frames_test'] = 8

    # Define path to text encoder config
    text_encoder_config_path = os.path.join(repo_path, "configs/config_bert_large.json")
    assert os.path.exists(text_encoder_config_path)
    config['TextEncoders']['bert_large']['config'] = text_encoder_config_path

    # Load model
    intern_model, tokenizer = setup_internvideo2(config)
    intern_model.eval()
    num_params(intern_model)


    # Run on the entire dataset
    print("[:::] Running on the entire dataset")

    # Load data
    csv_path = "/scratch/shared/nfs2/piyush/datasets/SSv2/metadata/time_antonyms-validation.csv"
    df = pd.read_csv(csv_path)

    data_dir = "/scratch/shared/beegfs/shared-datasets/SomethingSomething-V2/"
    video_dir = os.path.join(data_dir, "videos")

    iterator = tqdm(df.iterrows(), total=len(df))
    text_corrects = []
    video_corrects = []
    group_corrects = []
    failed = []
    for i, row in iterator:
        row = row.to_dict()
        video_path_x = get_video_path(video_dir, row["id_x"])
        video_path_y = get_video_path(video_dir, row["id_y"])
        label_x = row["label_x"]
        label_y = row["label_y"]
        video_paths = [video_path_x, video_path_y]
        texts = [label_x, label_y]
        with torch.no_grad():
            sim = compute_features(
                video_paths, texts, model=intern_model, config=config
            )
            sim = sim.cpu().numpy()
        """
        import ipdb; ipdb.set_trace()
        video_x = cv2.VideoCapture(video_path_x)
        frames_x = [x for x in _frame_from_video(video_x)]
        video_y = cv2.VideoCapture(video_path_y)
        frames_y = [x for x in _frame_from_video(video_y)]

        frames_tensor = frames2tensor(frames_x, fnum=fn, target_size=(size_t, size_t))

        import ipdb; ipdb.set_trace()

        # Compute video to text similarity for video_x
        video = cv2.VideoCapture(video_path_x)
        frames = [x for x in _frame_from_video(video)]
        text_candidates= [label_x, label_y]
        with torch.no_grad():
            texts, probs_x = retrieve_text(
                frames, text_candidates, model=intern_model, topk=2, config=config,
            )

        # Compute text to video similarity for video_y
        video = cv2.VideoCapture(video_path_y)
        frames = [x for x in _frame_from_video(video)]
        text_candidates= [label_x, label_y]
        with torch.no_grad():
            texts, probs_y = retrieve_text(
                frames, text_candidates, model=intern_model, topk=2, config=config,
            )

        # Get text to video similarity matrix
        sim = np.stack([probs_x, probs_y], axis=1)
        """
        
        text_corrects.append(text_correct(sim))
        video_corrects.append(video_correct(sim))
        group_corrects.append(group_correct(sim))

        # if i == 10:
        #     break

    # Compute final metrics
    text_corrects = np.array(text_corrects)
    video_corrects = np.array(video_corrects)
    group_corrects = np.array(group_corrects)

    print("Text score:", text_corrects.mean())
    print("Video score:", video_corrects.mean())
    print("Group score:", group_corrects.mean())
