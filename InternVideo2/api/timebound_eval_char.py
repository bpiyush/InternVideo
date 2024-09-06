"""Evaluation on Charades dataset."""
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
import video_language.datasets.charades as charades
import shared.utils as su

from multi_modality.demo.demo_utils import (
    # _frame_from_video,
    compute_video_text_features,
    get_video_features,
    get_text_features,
)
# from api.api_utils import num_params
from api.api_utils import load_model

def load_frames_decord(video_path, start_time=None, end_time=None, nf=8):
    import decord
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    sf = int(start_time * fps) if start_time else 0
    ef = int(end_time * fps) if end_time else len(vr)
    indices = np.linspace(sf, ef, nf, endpoint=False).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def video_check(sim):
    """Computes video score."""
    return (sim[0, 0] > sim[1, 0]) and (sim[1, 1] > sim[0, 1])


def text_check(sim):
    """Computes text score."""
    return (sim[0, 0] > sim[0, 1]) and (sim[1, 1] > sim[1, 0])


def group_check(sim):
    return video_check(sim) and text_check(sim)


def get_scores(sim):
    return dict(video=video_check(sim), text=text_check(sim), group=group_check(sim))


if __name__ == "__main__":

    # Load data
    split = "test"
    paths = charades.get_paths()
    _, df_main = charades.load_main_csv(paths, split)
    df_time = charades.load_time_csv(paths, split)
    df_pair = charades.load_pair_csv(paths, split)

    ckpt_name = "InternVideo2-stage2_1b-224p-f4.pt"
    model, _, config = load_model(ckpt_name=ckpt_name)


    # Feature computation
    video_features_path = ".internvideo2_charades_video_features.pt"
    text_features_path = ".internvideo2_charades_text_features.pt"
    if os.path.exists(video_features_path):
        video_features = torch.load(video_features_path)
        text_features = torch.load(text_features_path)
    else:

        # Step 1: Compute video features for all IDs
        ids_a = df_pair.id_a.unique()
        ids_b = df_pair.id_b.unique()
        ids = np.unique(np.concatenate([ids_a, ids_b]))

        iterator = su.log.tqdm_iterator(ids, desc="Computing features")
        video_features = dict()
        text_features = dict()
        for _id in iterator:
            video_id = _id.split("_")[0]
            video_path = charades.get_video_path_basic(video_id)
            assert os.path.exists(video_path), \
                f"Video path {video_path} does not exist for ID {_id}"
            row = df_main[df_main.item_id == str(_id)].iloc[0].to_dict()
            # label = f"{row['verb']} {row['noun']}"
            label = row["cls_name"]
            st = row["start_time"]
            et = row["end_time"]
            try:
                frames = load_frames_decord(video_path, st, et)
            except:
                print("Error in loading frames for", _id)
                continue
            with torch.no_grad():
                video_feats = get_video_features(frames, model, config)
                text_feats = get_text_features([label], model)
            video_features[_id] = video_feats[0].cpu().numpy()
            text_features[_id] = text_feats[0].cpu().numpy()
        
        # Save video features
        torch.save(video_features, video_features_path)
        torch.save(text_features, text_features_path)

    N = len(df_pair)
    results = []
    iterator = su.log.tqdm_iterator(range(N), desc="Evaluating")
    for i in iterator:
        row = df_pair.iloc[i].to_dict()
        id_a = row["id_a"]
        id_b = row["id_b"]
        if id_a not in video_features or id_b not in video_features:
            continue

        vid_a = video_features[id_a]
        vid_b = video_features[id_b]
        txt_a = text_features[id_a]
        txt_b = text_features[id_b]
        vid = np.stack([vid_a, vid_b])
        txt = np.stack([txt_a, txt_b])

        sim = (vid @ txt.T)
        r = get_scores(sim)
        results.append(r)
    results = pd.DataFrame(results)
    print(results.mean())

    # Save results
    os.makedirs("results", exist_ok=True)
    results.mean().to_csv("results/scores_char_internvideo2-s2.csv", index=False)

