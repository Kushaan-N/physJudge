"""Judge A: VideoPhy-2-AutoEval (mPLUG-Owl-Video), physical-commonsense task.

Uses the VideoPhy repo's model code, PROMPT_PHYSICS template and output parsing. One
deviation: the 16 condition frames go straight into its image processor at 224x224,
which is what the repo's load_video does after decord. They are not written to mp4 and
re-read, so every condition gets identical pixels and no codec or resampling
differences. Run in the env pinned to transformers==4.28.1.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import CONDITIONS, CLIPS_CSV, RESULTS, VIDEOS, WS, conditions, digit_expectation, done_keys, load_frames

VP2 = WS / "videophy" / "VIDEOPHY2"
sys.path.insert(0, str(VP2))
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration  # noqa: E402
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor, get_index  # noqa: E402
from template import PROMPT_PHYSICS  # noqa: E402
from transformers.models.llama.tokenization_llama import LlamaTokenizer  # noqa: E402

JUDGE = "autoeval"
NUM_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def parse(output):
    """Same rule as VIDEOPHY2/inference.py, but None instead of a silent default of 0."""
    o = output.lower().strip()
    for key, val in NUM_MAP.items():
        if key in o:
            return val
    return None


def video_tensor(frames, image_processor):
    imgs = [Image.fromarray(f).convert("RGB").resize((224, 224), Image.BILINEAR) for f in frames]
    pv = image_processor(imgs, return_tensors="pt")["pixel_values"]  # T,3,224,224
    return pv[None].permute(0, 2, 1, 3, 4)  # 1,3,T,224,224


NATIVE_DIR = WS / "data" / "mp4_native32"
NATIVE_FPS = 8


def native_frames(clip_id, k=32):
    """Robustness R3: k frames on the repo's own get_index grid, plus the same grid offset
    by half a stride. For clips with fewer than 2k frames the two sets partly overlap."""
    import av
    with av.open(str(VIDEOS / f"{clip_id}.mp4")) as c:
        allf = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    n = len(allf)
    seg = float(n - 1) / k
    start = int(seg / 2)
    idx = get_index(n, k)
    alt = np.clip(np.array([start + int(np.round(seg * (i + 0.5))) for i in range(k)]), 0, n - 1)
    return allf[idx], allf[alt], n, int(len(set(idx) & set(alt)))


def write_mp4(frames, path):
    import av
    h, w = frames.shape[1:3]
    s = min(1.0, 640 / max(h, w))
    W, H = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=NATIVE_FPS)
        st.width, st.height, st.pix_fmt, st.options = W, H, "yuv420p", {"crf": "18"}
        for f in frames:
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24").reformat(W, H)):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(WS / "models" / "videophy_2_auto"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(RESULTS / "scores_autoeval.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--native32", action="store_true",
                    help="robustness R3: 32 frames, written to mp4 and read by the repo's own load_video")
    args = ap.parse_args()
    judge = JUDGE + ("_native32" if args.native32 else "")
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(args.checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    cuda = args.device == "cuda"
    # No fp16: mPLUG's temporal conv3d has no Half kernel in torch 2.1. Without bf16, use fp32.
    dtype = torch.bfloat16 if (not cuda or torch.cuda.is_bf16_supported()) else torch.float32
    print("device", args.device, "dtype", dtype, torch.cuda.get_device_name() if cuda else "", flush=True)
    model = MplugOwlForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model = model.to(args.device).eval()

    # Digit tokens; LLaMA may emit '4' or '▁4'. Merge both spellings per digit.
    digit_ids = {d: [i for i in {tokenizer.convert_tokens_to_ids(d), tokenizer.convert_tokens_to_ids("▁" + d)}
                     if i is not None and i != tokenizer.unk_token_id] for d in "12345"}
    all_digit = {i for ids in digit_ids.values() for i in ids}

    text = processor(text=[PROMPT_PHYSICS], return_tensors="pt")
    clips = pd.read_csv(CLIPS_CSV)
    if args.limit:
        clips = clips.groupby("group").head(args.limit)
    done = done_keys(args.out)
    out = open(args.out, "a")
    for r in clips.itertuples():
        names = CONDITIONS if r.group == "plausible" else ["original"]
        if all((r.clip_id, c, judge) in done for c in names):
            continue
        extra = {}
        if args.native32:
            f32, a32, n, overlap = native_frames(r.clip_id)
            conds = conditions(f32, a32, r.clip_id)
            extra = {"n_total": n, "alt_overlap": overlap}
        else:
            frames, frames_alt = load_frames(r.clip_id)
            conds = conditions(frames, frames_alt, r.clip_id)
        for c in names:
            if (r.clip_id, c, judge) in done:
                continue
            if args.native32:
                path = NATIVE_DIR / f"{r.clip_id}_{c}.mp4"
                write_mp4(conds[c], path)
                inputs = processor(text=[PROMPT_PHYSICS], videos=[str(path)], num_frames=32, return_tensors="pt")
                inputs = dict(inputs)
            else:
                inputs = dict(text)
                inputs["video_pixel_values"] = video_tensor(conds[c], image_processor)
            inputs = {k: (v.to(dtype) if v.dtype == torch.float else v).to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                g = model.generate(**inputs, do_sample=False, top_k=1, max_length=256,
                                   output_scores=True, return_dict_in_generate=True)
            seq = g.sequences[0].tolist()
            raw = tokenizer.decode(seq, skip_special_tokens=True)
            # Answer position: the first generated step whose greedy token is a digit 1-5.
            probs, step = {d: 0.0 for d in "12345"}, None
            for s, sc in enumerate(g.scores):
                if int(sc[0].argmax()) in all_digit:
                    p = torch.softmax(sc[0].float(), -1)
                    probs = {d: float(sum(p[i] for i in ids)) for d, ids in digit_ids.items()}
                    step = s
                    break
            rec = {"clip_id": int(r.clip_id), "group": r.group, "condition": c, "judge": judge, **extra,
                   "score": parse(raw), "score_cont": digit_expectation(probs) if step is not None else None,
                   "digit_probs": probs, "digit_mass": sum(probs.values()), "answer_step": step, "raw": raw}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(r.clip_id, c, rec["score"], rec["score_cont"], repr(raw), flush=True)


if __name__ == "__main__":
    main()
