import torch
import numpy as np
import os


def num_params(model):
    n = np.sum([p.numel() for p in model.parameters()]) / 1e6
    print(f"Number of parameters in {type(model).__name__}: {np.round(n, 3)}M")


def load_model(ckpt_name="InternVideo2-stage2_1b-224p-f4.pt"):

    import sys
    repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(repo_path)
    repo_path = os.path.join(repo_path, "multi_modality")
    sys.path.append(repo_path)

    from multi_modality.demo.config import (
        Config,
        eval_dict_leaf,
    )
    from multi_modality.demo.demo_utils import (
        retrieve_text,
        _frame_from_video,
        setup_internvideo2,
    )

    # Load model
    config = Config.from_file('InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)

    ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/InternVideo/"
    model_pth = os.path.join(ckpt_root, ckpt_name)
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
    return intern_model, tokenizer, config