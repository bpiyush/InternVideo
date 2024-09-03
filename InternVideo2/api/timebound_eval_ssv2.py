"""Script to evaluate on SSv2."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import io
import cv2
import pandas as pd

import torch

import sys
repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(repo_path)
repo_path = os.path.join(repo_path, "multi_modality")
sys.path.append(repo_path)

sys.path.append("../TimeBound.v1/")
import video_language.datasets.ssv2 as ssv2


# from multi_modality.demo.config import (
#     Config,
#     eval_dict_leaf,
# )
from multi_modality.demo.demo_utils import (
    _frame_from_video,
    compute_video_text_features,
)
# from api.api_utils import num_params
from api.api_utils import load_model


def video_check(sim):
    """Computes video score."""
    return (sim[0, 0] > sim[1, 0]) and (sim[1, 1] > sim[0, 1])


def text_check(sim):
    """Computes text score."""
    return (sim[0, 0] > sim[0, 1]) and (sim[1, 1] > sim[1, 0])


def group_check(sim):
    return video_check(sim) and text_check(sim)


def get_check_for_row(row):
    id_a = row["id_a"]
    video_path_a = ssv2.get_video_path_basic(id_a)
    label_a = df_main[df_main.id == str(id_a)].iloc[0].label

    id_b = row["id_b"]
    video_path_b = ssv2.get_video_path_basic(id_b)
    label_b = df_main[df_main.id == str(id_b)].iloc[0].label

    texts = [label_a, label_b]
    frames = [
        [x for x in _frame_from_video(cv2.VideoCapture(y))] \
            for y in [video_path_a, video_path_b]
    ]
    z_v, z_t = compute_video_text_features(
        model=model, texts=texts, video_frames=frames, config=config,
    )
    sim = (z_v @ z_t.T).cpu().numpy()
    vc = video_check(sim)
    tc = text_check(sim)
    gc = group_check(sim)
    return dict(video_score=vc, text_score=tc, group_score=gc)


if __name__ == "__main__":

    init_with_clip = False
    if init_with_clip:
        # THIS IS TOO COMPLICATED TO SETUP
        # SKIPPING FOR NOW

        from multi_modality.demo.config import (
            Config,
            eval_dict_leaf,
        )
        from multi_modality.tasks_clip.shared_utils import setup_model
        from multi_modality.models.internvideo2_clip import InternVideo2_CLIP
    
        config = Config.from_file('InternVideo2/multi_modality/scripts/evaluation/clip/zero_shot/1B/config_ssv2_mc.py')
        config = eval_dict_leaf(config)

        # Updates
        # config["model"]["tokenizer_path"] = "/work/piyush/pretrained_checkpoints/LargeModels/InternVideo/chinese-alpaca-lora-7b"
        ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/InternVideo/"
        ckpt_name = "InternVideo2-stage2_1b-224p-f4.pt"
        model_pth = os.path.join(ckpt_root, ckpt_name)
        config['pretrained_path'] = model_pth


        model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
        model, _ = setup_model(
            config,
            model_cls=model_cls,
            pretrain=False,
            find_unused_parameters=False,
        )
        raise NotImplementedError
        import ipdb; ipdb.set_trace()

        # Ref: https://github.com/OpenGVLab/InternVideo/issues/114
        ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/InternVideo/"
        model_pth = os.path.join(ckpt_root, "1B_clip.pth")
        assert os.path.exists(model_pth)
        ckpt = torch.load(model_pth)
        msg = model.load_state_dict(ckpt, strict=False)
        import ipdb; ipdb.set_trace()
    else:
        ckpt_name = "InternVideo2-stage2_1b-224p-f4.pt"
        model, _, config = load_model(ckpt_name=ckpt_name)


    # Load data
    split = "validation"
    paths = ssv2.get_paths()
    df_main = ssv2.load_main_csv(paths, split=split)
    df_time = ssv2.load_time_csv(paths, split=split)
    df_pair = pd.read_csv(
        os.path.join(
            "/work/piyush/from_nfs2/datasets/SSv2/time_antonyms_annotations",
            "video_single_antonym_pairs-validation.csv",
        )
    )

    # Evaluate on a single row
    debug = False
    if debug:
        i = 0
        row = df_pair.iloc[i].to_dict()
        id_a = row["id_a"]
        video_path_a = ssv2.get_video_path_basic(id_a)
        label_a = df_main[df_main.id == str(id_a)].iloc[0].label
        df_main[df_main.id == id_a]

        id_b = row["id_b"]
        video_path_b = ssv2.get_video_path_basic(id_b)
        label_b = df_main[df_main.id == str(id_b)].iloc[0].label

        texts = [label_a, label_b]
        frames = [[x for x in _frame_from_video(cv2.VideoCapture(y))] for y in [video_path_a, video_path_b]]
        z_v, z_t = compute_video_text_features(
            model=model, texts=texts, video_frames=frames, config=config,
        )
        sim = (z_v @ z_t.T).cpu().numpy()
        vc = video_check(sim)
        tc = text_check(sim)
        gc = group_check(sim)
    
    # Run on entire dataset
    import shared.utils as su
    N = len(df_pair)
    # N = 100
    iterator = su.log.tqdm_iterator(range(N), desc="Evaluating")
    scores = []
    for i in iterator:
        row = df_pair.iloc[i].to_dict()
        _scores = get_check_for_row(row)
        scores.append(_scores)
    scores = pd.DataFrame(scores)
    print(scores.mean())

    # Save results
    os.makedirs("results", exist_ok=True)
    scores.to_csv("results/scores_ssv2_internvideo2-s2.csv", index=False)
