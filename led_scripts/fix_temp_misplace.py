import glob
import os

for f in glob.glob("*color-tempauto.jpg"):
    print("-"*25)
    print(f)
    parts = f.rstrip(".jpg").split("_")
    parts.insert(-2, parts[-1])
    print(parts)
    new_f = "_".join(parts[:-1]) + ".jpg"
    print(new_f)
    os.rename(f, new_f)
