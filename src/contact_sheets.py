"""Contact sheets for the eyeball check: 20 plausible clips (seeded), each shown as the
16 frames in reversed order (read left to right, top to bottom), with the caption.
Writes results/eyeball/<clip_id>.jpg and a template results/eyeball_reversed.csv to fill
in by hand (plausible_backward: 1 if the reversed clip looks physically plausible)."""
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import CLIPS_CSV, RESULTS, load_frames

N = 20
TILE_W = 240


def sheet(frames, caption):
    h, w = frames.shape[1:3]
    th = round(TILE_W * h / w)
    tiles = [Image.fromarray(f).resize((TILE_W, th)) for f in frames]
    head = 40
    img = Image.new("RGB", (TILE_W * 8, th * 2 + head), "white")
    ImageDraw.Draw(img).text((6, 4), "\n".join(textwrap.wrap("REVERSED: " + caption, 150)[:2]), fill="black")
    for i, t in enumerate(tiles):
        img.paste(t, ((i % 8) * TILE_W, head + (i // 8) * th))
        ImageDraw.Draw(img).text(((i % 8) * TILE_W + 4, head + (i // 8) * th + 4), str(i + 1), fill="yellow")
    return img


def main():
    out = RESULTS / "eyeball"
    out.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(CLIPS_CSV)
    pl = clips[clips.group == "plausible"]
    pick = pl.sample(N, random_state=1).sort_values("clip_id")
    for r in pick.itertuples():
        frames, _ = load_frames(r.clip_id)
        sheet(frames[::-1], r.caption).save(out / f"{r.clip_id}.jpg", quality=85)
    tmpl = RESULTS / "eyeball_reversed.csv"
    if not tmpl.exists():
        pick.assign(plausible_backward=np.nan, note="")[["clip_id", "action", "caption", "plausible_backward", "note"]] \
            .to_csv(tmpl, index=False)
    print(pick[["clip_id", "action"]].to_string(index=False))


if __name__ == "__main__":
    main()
