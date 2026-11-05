"""Judge B: Qwen2.5-VL-Instruct scoring the 16 frames sent as separate images.
Resumable: skips (clip, condition, judge) keys already in the output jsonl."""
import argparse
import json
import re

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import CONDITIONS, CLIPS_CSV, PC_INSTRUCTION, RESULTS, conditions, digit_expectation, done_keys, load_frames

MAX_PIXELS = 256 * 28 * 28  # ~256 visual tokens per frame


INTRO = {"default": "The following {n} images are frames from a video, in order.",
         "no_order": "The following {n} images are frames from a video."}  # robustness R4


def build_messages(frames, intro="default"):
    content = [{"type": "text", "text": INTRO[intro].format(n=len(frames))}]
    content += [{"type": "image", "image": Image.fromarray(f), "max_pixels": MAX_PIXELS} for f in frames]
    content.append({"type": "text", "text": PC_INSTRUCTION + " Answer with a single number from 1 to 5 and nothing else."})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N clips (smoke test)")
    ap.add_argument("--out", default=str(RESULTS / "scores_qwen.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--intro", default="default", choices=["default", "no_order"])
    args = ap.parse_args()

    from qwen_vl_utils import process_vision_info

    judge = args.model.split("/")[-1] + ("" if args.intro == "default" else f"_{args.intro}")
    cuda = args.device == "cuda"
    dtype = torch.bfloat16 if (not cuda or torch.cuda.is_bf16_supported()) else torch.float16
    print("device", args.device, "dtype", dtype, torch.cuda.get_device_name() if cuda else "", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    digit_ids = {d: proc.tokenizer.convert_tokens_to_ids(d) for d in "12345"}

    clips = pd.read_csv(CLIPS_CSV)
    if args.limit:
        clips = clips.groupby("group").head(args.limit)
    done = done_keys(args.out)
    out = open(args.out, "a")
    for r in clips.itertuples():
        frames, frames_alt = load_frames(r.clip_id)
        conds = conditions(frames, frames_alt, r.clip_id)
        names = CONDITIONS if r.group == "plausible" else ["original"]
        for c in names:
            if (r.clip_id, c, judge) in done:
                continue
            msgs = build_messages(conds[c], args.intro)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images, _ = process_vision_info(msgs)
            inputs = proc(text=[text], images=images, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   output_scores=True, return_dict_in_generate=True)
            raw = proc.batch_decode(g.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            p = torch.softmax(g.scores[0][0].float(), -1)
            probs = {d: float(p[i]) for d, i in digit_ids.items()}
            m = re.search(r"[1-5]", raw)
            rec = {"clip_id": int(r.clip_id), "group": r.group, "condition": c, "judge": judge,
                   "score": int(m.group()) if m else None, "score_cont": digit_expectation(probs),
                   "digit_probs": probs, "digit_mass": sum(probs.values()), "raw": raw}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(r.clip_id, c, rec["score"], round(rec["score_cont"] or -1, 3), repr(raw), flush=True)


if __name__ == "__main__":
    main()
