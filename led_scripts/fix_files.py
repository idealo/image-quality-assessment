import glob

for f in glob.glob("./test_images/*no-ambient*"):
    print(f)
    new_name = f.replace("no-ambient_", "")
    print(new_name)

