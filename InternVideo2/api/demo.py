"""
Script to compute s(v, t) for a random video text pair
with InternVideo2-Stage2 model.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import io
import cv2

import torch

import sys
# repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "multi_modality")
repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(repo_path)

repo_path = os.path.join(repo_path, "multi_modality")
sys.path.append(repo_path)

from multi_modality.demo.config import (Config,
                    eval_dict_leaf)

from multi_modality.demo.demo_utils import (retrieve_text,
                  _frame_from_video,
                  setup_internvideo2)


def num_params(model):
    n = np.sum([p.numel() for p in model.parameters()]) / 1e6
    print(f"Number of parameters in {type(model).__name__}: {np.round(n, 3)}M")


if __name__ == "__main__":

    # Load video
    video_path = "/users/piyush/projects/TimeBound.v1/sample_data/folding_paper.mp4"
    assert os.path.exists(video_path)
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    print("Number of frames:", len(frames))
    text_candidates = [
        "Someone folding a paper",
        "Someone unfolding a paper",
    ]
    print("Number of text candidates:", len(text_candidates))

    # Load model
    config = Config.from_file('InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)

    ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/InternVideo/"
    """
    model_pth = os.path.join(ckpt_root, "InternVideo2-stage2_1b-224p-f4.pt")
    config["model"]["vision_encoder"]["pretrained"] = model_pth
    # Define path to text encoder config
    text_encoder_config_path = os.path.join(repo_path, "configs/config_bert_large.json")
    assert os.path.exists(text_encoder_config_path)
    config['TextEncoders']['bert_large']['config'] = text_encoder_config_path
    """
    model_pth = os.path.join(ckpt_root, "InternVideo2-stage2_1b-224p-f4.pt")
    config['pretrained_path'] = model_pth

    # Had to set this to avoid loading separate vision checkpoint
    config['model']['vision_encoder']['pretrained'] = None

    # Had to set this to ensure correct config path
    text_encoder_config_path = os.path.join(repo_path, "configs/config_bert_large.json")
    assert os.path.exists(text_encoder_config_path)
    config['TextEncoders']['bert_large']['config'] = text_encoder_config_path

    intern_model, tokenizer = setup_internvideo2(config)
    intern_model = intern_model.eval()
    num_params(intern_model)

    with torch.no_grad():
        texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=2, config=config)
        for t, p in zip(texts, probs):
            print(f'text: {t} ~ prob: {p:.4f}')

        from multi_modality.demo.demo_utils import compute_video_text_features
        v_feats, t_feats = compute_video_text_features(
            [frames], text_candidates, intern_model, config,
        )
        sim_v2t = torch.nn.functional.cosine_similarity(v_feats, t_feats, dim=-1)
        print(sim_v2t)
