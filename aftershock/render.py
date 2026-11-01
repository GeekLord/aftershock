import numpy as np
from PIL import Image, ImageDraw, ImageFont
COL = {"no_damage": (34, 197, 94), "minor": (234, 179, 8), "major": (249, 115, 22), "destroyed": (220, 38, 38), "unknown": (148, 163, 184)}

def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def overlay(img, result, masks, max_side=1800):
    base = img.convert("RGB").copy(); s = min(1.0, max_side / max(base.size)); base = base.resize((int(base.size[0] * s), int(base.size[1] * s)))
    lay = Image.new("RGBA", base.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for b, grade in masks:
        x1, y1, x2, y2 = [int(v * s) for v in b["box"]]
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        m = Image.fromarray((b["mask"] * 140).astype(np.uint8)).resize((x2 - x1, y2 - y1))
        lay.paste(Image.new("RGBA", m.size, COL.get(grade, COL["unknown"]) + (255,)), (x1, y1), m)
        d.rectangle([x1, y1, x2, y2], outline=COL.get(grade, COL["unknown"]) + (255,), width=3)
    f = _font(max(18, base.size[0] // 45))
    for c in result["cells"]:
        x1, y1, x2, y2 = [int(v * s) for v in c["box"]]
        d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 200), width=2)
        tag = f"#{c['rank']}" + (" ⚑" if c["review"] else "")
        tw = d.textlength(tag, font=f); d.rectangle([x1 + 6, y1 + 6, x1 + 18 + tw, y1 + 14 + f.size], fill=(15, 23, 42, 210))
        d.text((x1 + 12, y1 + 8), tag, font=f, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base.convert("RGBA"), lay).convert("RGB")
