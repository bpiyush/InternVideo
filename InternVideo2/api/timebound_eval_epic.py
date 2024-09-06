"""Script to evaluate on Epic."""
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
import video_language.datasets.epic as epic
import shared.utils as su

# from multi_modality.demo.config import (
#     Config,
#     eval_dict_leaf,
# )
from multi_modality.demo.demo_utils import (
    # _frame_from_video,
    compute_video_text_features,
    get_video_features,
    get_text_features,
)
# from api.api_utils import num_params
from api.api_utils import load_model


def _frame_from_video(video, start_time=None, end_time=None):
    fps = video.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success or current_frame > end_frame:
            break
        
        if current_frame >= start_frame:
            yield frame
        
        current_frame += 1


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


def get_check_for_row(row):
    id_a = row["id_a"]
    video_id_a = "_".join(id_a.split("_")[:-1])
    video_path_a = epic.get_video_path_basic(video_id_a)
    assert os.path.exists(video_path_a)
    row_a = df_main[df_main.narration_id == str(id_a)].iloc[0].to_dict()
    label_a = f"{row_a['verb']} {row_a['noun']}"
    start_a = row_a["start_sec"]
    end_a = row_a["stop_sec"]

    id_b = row["id_b"]
    video_id_b = "_".join(id_b.split("_")[:-1])
    video_path_b = epic.get_video_path_basic(video_id_b)
    assert os.path.exists(video_path_b)
    row_b = df_main[df_main.narration_id == str(id_b)].iloc[0].to_dict()
    label_b = f"{row_b['verb']} {row_b['noun']}"
    start_b = row_b["start_sec"]
    end_b = row_b["stop_sec"]

    texts = [label_a, label_b]
    frames_a = load_frames_decord(video_path_a, start_a, end_a)
    frames_b = load_frames_decord(video_path_b, start_b, end_b)
    # import ipdb; ipdb.set_trace()
    # # NOTE: only pick frames between start and end for each video
    # video_a = cv2.VideoCapture(video_path_a)
    # video_b = cv2.VideoCapture(video_path_b)
    # import ipdb; ipdb.set_trace()
    # frames_a = [x for x in _frame_from_video(video_a, start_a, end_a)]
    # frames_b = [x for x in _frame_from_video(video_b, start_b, end_b)]
    frames = [frames_a, frames_b]
    # frames = [
    #     [x for x in _frame_from_video(cv2.VideoCapture(y))] \
    #         for y in [video_path_a, video_path_b]
    # ]
    z_v, z_t = compute_video_text_features(
        model=model, texts=texts, video_frames=frames, config=config,
    )
    sim = (z_v @ z_t.T).cpu().numpy()
    vc = video_check(sim)
    tc = text_check(sim)
    gc = group_check(sim)
    return dict(video_score=vc, text_score=tc, group_score=gc)


if __name__ == "__main__":

    # Load data
    split = "validation"
    paths = epic.get_paths()
    df_main = epic.load_main_csv(paths, split=split)
    _, df_time = epic.load_time_csv(paths, split=split)
    df_pair = pd.read_csv(
        os.path.join(
            paths["meta_dir"],
            "video_single_antonym_pairs-validation.csv",
        )
    )

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

    video_features_path = ".internvideo2_epic_video_features.pt"
    text_features_path = ".internvideo2_epic_text_features.pt"
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
            video_id = "_".join(_id.split("_")[:-1])
            video_path = epic.get_video_path_basic(video_id)
            assert os.path.exists(video_path)
            row = df_main[df_main.narration_id == str(_id)].iloc[0].to_dict()
            label = f"{row['verb']} {row['noun']}"
            st = row["start_sec"]
            et = row["stop_sec"]
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
    results.to_csv("results/scores_epic_internvideo2-s2.csv", index=False)
        
    """
    # Evaluate on a single row
    debug = True
    if debug:
        i = 0
        row = df_pair.iloc[i].to_dict()
        result = get_check_for_row(row)
        print(result)

    # Run on entire dataset
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
    scores.to_csv("results/scores_epic_internvideo2-s2.csv", index=False)
    """
