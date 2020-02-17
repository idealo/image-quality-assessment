import glob
import os

dispatcher = {
    "back-strip-only": "back_strip_only",
    "left-strip-only": "left_strip_only",
    "right-strip-only": "right_strip_only",
    "full-strip": "full_strip",
    "left-strip-linear-fade": "left_strip_linear_fade",
    "left-strip-linear-fade-back-dim": "left_strip_linear_fade",
    "right-strip-linear-fade": "right_strip_linear_fade",
    "right-strip-linear-fade-back-dim": "right_strip_linear_fade",
    "left-right-linear-fade": "left_right_linear_fade",
    "left-right-linear-fade-back-dim": "left_right_linear_fade",
    "linear-fade-towards-back-middle": "linear_fade_towards_back_middle",
}

for f in glob.glob("*.jpg"):
    new_f = f.replace("led_brightness", "led-brightness")
    for new, old in dispatcher.items():
        if old in new_f:
            new_f = new_f.replace(old, new)

    print(new_f)
    os.rename(f, new_f)
