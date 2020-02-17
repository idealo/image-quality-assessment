import json
import pandas
import os
import cv2
import subprocess
import sys
import zipfile
import shutil
import PIL.Image
import glob

import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt

# import qualipy as qp
# import qualipy.filters as qpf
import imquality.brisque as brisque

from IPython.display import Image, display, TextDisplayObject
from pprint import pprint
from ipywidgets import fixed, interactive_output
from scipy import stats as scistats
from textwrap import dedent
from joblib import Parallel, delayed

# from brisque import BRISQUE

font = cv2.FONT_HERSHEY_COMPLEX


def subprocess_wrap(fname, cmd, log=True, stream=False, **kwargs):
    """
    Write output of command to a file for later processing, stream it to stdout, and return the process object
    
    **kwargs are passedto subprocess.Popen
    """
    with open(fname, "wb") as f:  # replace 'w' with 'wb' for Python 3
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, **kwargs)
        for line in iter(
            process.stdout.readline, b""
        ):  # replace '' with b'' for Python 3
            if stream:
                sys.stdout.write(line)
            if log:
                f.write(line)
    return process


def run_idealo_iqa(paths, model, build_log="docker-build.log", rebuild=False, gpu=False):
    if model not in {"technical", "aesthetic"}:
        raise ValueError("Model arg must be either 'technical' or 'aesthetic'")
    if gpu:
        suffix = "gpu"
    else:
        suffix = "cpu"
    if rebuild:
        print(f"Building idealo container from Dockerfile.{suffix} ...")
        build_cmd = f"docker build -t nima-{suffix} . -f Dockerfile.{suffix}"
        proc = subprocess_wrap(build_log, build_cmd, log=True, stream=True, shell=True)

    weights = (
        "weights_mobilenet_technical_0.11.hdf5"
        if model == "technical"
        else "weights_mobilenet_aesthetic_0.07.hdf5"
    )
    predict_cmd = dedent(
        f"""
    ./predict  \
    --docker-image nima-{suffix} \
    --base-model-name MobileNet \
    --weights-file $(pwd)/models/MobileNet/{weights} \
    --image-source $(pwd)/idealo_images
    """
    )
    iqa_log = f"output-{model}.log"

    # Bit of a hack but easiest way to deal with interface to idealo predictor.
    # Make a tmp directory and move the desired images into tmp dir temporarily, then
    # move them back to their original location when done. Had to do this
    # because symlinks don't really work in containers
    os.makedirs("idealo_images")
    for p in paths:
        shutil.move(p, "idealo_images/")

    print(f"Running {model} idealo model ...")
    proc = subprocess_wrap(iqa_log, predict_cmd, log=True, stream=False, shell=True)

    for p in glob.glob("idealo_images/*"):
        shutil.move(p, "test_images/")
    os.rmdir("idealo_images")

    with open(iqa_log, "r") as f:
        lines = [l for l in f.readlines() if not ("ETA:" in l or "s/step" in l)]
        json_txt = "".join(lines)
    scores = json.loads(json_txt)

    return scores


def run_qualipy_analysis(
    paths,
    run_log="docker-qualipy-run.log",
    build_log="docker-qualipy-build.log",
    rebuild=False,
):
    with open("qualipy_images.txt", "w") as f:
        for p in paths:
            container_path = os.path.join(
                "/opt",
                "app",
                "image-quality-assessment",
                "test_images",
                os.path.split(p)[-1],
            )
            f.write(container_path + "\n")
    if rebuild:
        print("Rebuilding Qualipy Docker image ...")
        build_cmd = "docker build -t qualipy-runner -f DockerfileQualipy ."
        proc = subprocess_wrap(build_log, build_cmd, log=True, stream=True, shell=True)
    run_cmd = "docker run -v /home/kyle/WEI/image-quality-assessment:/opt/app/image-quality-assessment qualipy-runner"
    print("Running qualipy analysis ...")
    proc = subprocess_wrap(run_log, run_cmd, log=True, stream=False, shell=True)
    with open("qualipy_analysis_results.json", "r") as f:
        results = json.load(f)
    cleaned = {}
    for d in results:
        cleaned.update(d)
    return {get_image_id_from_path(k): v for k, v in cleaned.items()}


def get_opencv_converter(input_space, output_space):
    return getattr(cv2, f"COLOR_{input_space}2{output_space}")


def compute_colorspace_stats(image, in_space, stat_space=None, channels=None):
    in_space = in_space.upper()
    if stat_space is None:
        stat_space = in_space
    stat_space = stat_space.upper()
    allowed_spaces = {
        "BGR": {0: "blue", 1: "green", 2: "red"},
        "HSV": {0: "hue", 1: "saturation", 2: "value"},
    }
    if in_space not in allowed_spaces:
        raise ValueError(f"Input color space must be one of {allowed_spaces}")
    if stat_space not in allowed_spaces:
        raise ValueError(f"Stats color space must be one of {allowed_spaces}")
    channel_map = allowed_spaces[stat_space]

    if isinstance(image, str):
        image = cv2.imread(image)
    if in_space != stat_space:
        image = cv2.cvtColor(image, get_opencv_converter(in_space, stat_space))
    stats = {}
    for ind, chan in channel_map.items():
        chan_arr = image[:, :, ind]
        k = f"{chan}_max"
        chan_max = np.amax(chan_arr)
        stats[k] = chan_max

        k = f"{chan}_mean"
        stats[k] = np.mean(chan_arr)

        k = f"{chan}_mode"
        mode_result = scistats.mode(chan_arr, axis=None)
        stats[k] = mode_result.mode[0]
        stats[k + "_count"] = mode_result.count[0]
    return stats


def compute_qualipy_stats(image):
    results = qp.process(
        image,
        [
            qpf.WholeBlur(),
            qpf.BlurredContext(),
            qpf.Exposure(negative_under_exposed=True),
            # qpf.Highlights(),
            # qpf.CrossProcessed(),
        ],
        combine_results=False,
        return_predictions=True,
    )
    return results


def compute_pybrisque_score(image):
    brisq = BRISQUE()
    return {"pyb_brisque_score": brisq.get_score(img)}


def compute_imquality_brisque_score(image):
    img = PIL.Image.open(image)
    return {"imq_brisque_score": brisque.score(img)}


def fix_image_size(image, expected_pixels=2e6):
    if isinstance(image, str):
        image = cv2.imread(image)
    ratio = expected_pixels / (image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image, threshold=100):
    if isinstance(image, str):
        image = cv2.imread(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)


def compute_blur_score(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = fix_image_size(image)
    blur_map, score, is_blurred = estimate_blur(image)
    return {"laplacian_blur_score": score}


def compute_mean_intensity_greyscale(image, in_space="BGR"):
    if isinstance(image, str):
        image = cv2.imread(image)
    in_space = in_space.upper()
    if image.ndim == 3 or in_space != "GRAY":
        image = cv2.cvtColor(image, get_opencv_converter(in_space, "GRAY"))
    return {"grayscale_mean_intensity": np.mean(image)}


def get_image_id_from_path(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def compute_image_statistics_parallel(
    paths, threads=True, n_jobs=-2, funcs=None
):
    """
    Compute a bunch of image statistics and return a new dataframe containing them
    """


    allowed_funcs = {
        "grayscale_mean_intensity": (compute_mean_intensity_greyscale, tuple(), dict()),
        "laplacian_blur_score": (compute_blur_score, tuple(), dict()),
        "brisque_score": (compute_imquality_brisque_score, tuple(), dict()),
        "bgr_stats": (compute_colorspace_stats, ("BGR",), {"stat_space": "BGR"}),
        "hsv_stats": (compute_colorspace_stats, ("BGR",), {"stat_space": "HSV"}),
        "qualipy": ("", "", ""),
        "idealo": ("", "", ""),
    }
    if funcs is None:
        funcs = allowed_funcs
    if any(f not in allowed_funcs for f in funcs):
        names = list(allowed_funcs.keys())
        raise ValueError(f"Provided funcs {funcs} not a subset of {names}")


    # This returns results in the same order as input args, so don't need to worry about
    # sorting
    if threads:
        par = Parallel(n_jobs=n_jobs, prefer="threads")
    else:
        par = Parallel(n_jobs=n_jobs, prefer="processes")

    all_data = [{"image_id": get_image_id_from_path(p), "path": p} for p in paths]

    for k, (func, args, kwargs) in allowed_funcs.items():
        # qualipy and idealo need special treatment down below
        if k in {"qualipy", "idealo"} or k not in funcs:
            continue
        print(f"Running {func.__name__} ...")
        results = par(delayed(func)(path, *args, **kwargs) for path in paths)
        for i, result in enumerate(results):
            all_data[i].update(result)

    all_data = {d["image_id"]: d for d in all_data}
    if "qualipy" in funcs:
        qp_results = run_qualipy_analysis(paths)
        for image_id, result in qp_results.items():
            all_data[image_id].update(result)


    if "idealo" in funcs:
        technical_scores = run_idealo_iqa(paths, "technical")
        for d in technical_scores:
            if d["image_id"] in all_data:
                all_data[d["image_id"]]["technical_score"] = d["mean_score_prediction"]
        aesthetic_scores = run_idealo_iqa(paths, "aesthetic")
        for d in aesthetic_scores:
            if d["image_id"] in all_data:
                all_data[d["image_id"]]["aesthetic_score"] = d["mean_score_prediction"]
    return list(all_data.values())


def compute_image_statistics_serial(df, base="", channels=None):
    """
    Compute a bunch of image statistics and return a new dataframe containing them
    """

    if channels is None:
        channels = {0: "hue", 1: "saturation", 2: "value"}
    stats = {}
    for img_id in df["image_id"]:
        if base:
            img_path = os.path.join(base, img_id + ".jpg")
        else:
            img_path = img_id + ".jpg"
        img = cv2.imread(img_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for ind, chan in channels.items():
            chan_arr = img_hsv[:, :, ind]
            k = f"{chan}_mean"
            # print(k)
            chan_max = np.amax(chan_arr)
            # print(chan_arr.shape)
            # print(f"{chan} max = {chan_max}")
            if k in stats:
                stats[k].append(np.mean(chan_arr))
            else:
                stats[k] = [np.mean(chan_arr)]
            k = f"{chan}_mode"
            # mode_result = scistats.mode(chan_arr, axis=None)
            # if k in stats:
            #    stats[k].append(mode_result.mode[0])
            #    stats[k+"_count"].append(mode_result.count[0])
            # else:
            #    stats[k] = [mode_result.mode[0]]
            #    stats[k+"_count"] = [mode_result.count[0]]
    # for k, v in stats.items():
    #    df[k] = v
    return pandas.DataFrame(stats)


def compute_histogram(image, in_space, hist_space=None, bins=None):
    in_space = in_space.upper()
    if hist_space is None:
        hist_space = in_space
    hist_space = hist_space.upper()
    allowed_spaces = {
        "GRAY": {0: ("gray", 255)},
        "BGR": {0: ("blue", 255), 1: ("green", 255), 2: ("red", 255)},
        "HSV": {0: ("hue", 179), 1: ("saturation", 255), 2: ("value", 255)},
    }

    if in_space not in allowed_spaces:
        raise ValueError(f"Input color space must be one of {allowed_spaces}")
    if hist_space not in allowed_spaces:
        raise ValueError(f"Histogram color space must be one of {allowed_spaces}")
    channel_map = allowed_spaces[hist_space]
    # By default use number of bins equal to number of allowable values in
    # channel, and update with user provided values
    default_bins = {tup[0]: tup[1] for tup in allowed_spaces[hist_space].values()}
    if any(k not in default_bins for k in bins.keys()):
        bin_keys = list(bins.keys())
        allowed_keys = list(default_bins.keys()) 
        msg = f"Bin keys {bin_keys} not a subnets of {allowed_keys}"
        raise ValueError(msg)
    if bins is not None:
        default_bins.update(bins)

    if isinstance(image, str):
        image = cv2.imread(image)

    if in_space != hist_space:
        image = cv2.cvtColor(image, get_opencv_converter(in_space, hist_space))

    hists = {}
    for chan_idx, (chan_name, max_val) in channel_map.items():
        histr = cv2.calcHist(
            [image], [chan_idx], None, [default_bins[chan_name]], [0, max_val]
        )
        hists[chan_name] = histr
    return hists


def plot_hists(hist_dict, in_space):
    in_space = in_space.upper()
    allowed_spaces = {
        "GRAY": {0: ("gray", 255)},
        "BGR": {"blue": (0, 255), "green": (1, 255), "red": (2, 255)},
        "HSV": {"hue": (0, 179), "saturation": (1, 255), "value": (2, 255)},
    }
    if in_space not in allowed_spaces:
        allowed_keys = list(allowed_spaces.keys())
        msg = f"Invalid color space {in_space}. Must be on of {allowed_keys}"
        raise ValueError(msg)
    channel_map = allowed_spaces[in_space]
    fig, axes = plt.subplots(1, len(hist_dict), figsize=(12, 6))
    axes = axes.flatten()
    for chan_name, (chan_ind, max_val) in channel_map.items():
        hist = hist_dict[chan_name]
        x = np.linspace(0, max_val, len(hist.flatten()) + 1)
        width = x[1] - x[0]
        x = x[1:] - width/2
        ax = axes[chan_ind]
        ax.set_title(chan_name)
        # ax.plot(histr)
        ax.bar(x, hist.flatten(), width=width)
    plt.tight_layout()
    plt.show()


def threshold_hue_in_range_hsv(img_hsv, min_val, max_val, invert=False):
    """
    Sets all pixels outside of a particular hue range to black by setting value channel to zero
    
    Inverted keyword inverts the logic. So everything *inside* the range is set to black and
    everything outside is left alone
    """
    hue = img_hsv[:, :, 0]
    value = img_hsv[:, :, 2]
    # threshed_hue = np.where((hue < min_val) | (hue > max_val), 0, hue)
    if invert:
        threshed_value = np.where((hue > min_val) & (hue < max_val), 0, value)
    else:
        threshed_value = np.where((hue < min_val) | (hue > max_val), 0, value)
    threshed_img_hsv = np.copy(img_hsv)
    # threshed_img_hsv[:, :, 0] = threshed_hue
    threshed_img_hsv[:, :, 2] = threshed_value
    return threshed_img_hsv


def display_images_matching_criteria(df, base, criteria=None, count=None):
    """
    Opens all images in df. 
    
    If `criteria` is not None, only open images matching `criteria`. 
    
    If count is not None, only opens the first `count` images  sorted by descending score
    """

    if criteria is not None:
        cleaned = df[criteria]
    else:
        cleaned = df
    if count is not None:
        row_iter = cleaned.nlargest(count, "score").iterrows()
    else:
        row_iter = cleaned.iterrows()
    for idx, row in row_iter:
        print(row["image_id"])
        img_file = os.path.join(base, row["image_id"] + ".jpg")
        img = Image(img_file)
        display(img)


def display_images(paths):
    """
    Displays images from a list of image paths
    """

    for path in paths:
        img = Image(path)
        #print(path)
        display(path, img)


def get_images_matching_criteria(
    df, criteria=None, count=None, sort_by="", ascending=True
):
    """
    Returns paths of all images in df.
    
    If `criteria` is not None, only returns images matching `criteria`. 
    
    If count is not None, only opens the first `count` images  sorted by descending score
    """

    if criteria is not None:
        cleaned = df[criteria]
    else:
        cleaned = df
    if sort_by:
        #cleaned = cleaned.sort_values(sort_by, ignore_index=True, ascending=ascending)
        cleaned = cleaned.sort_values(sort_by, ascending=ascending)
    if count is not None:
        row_iter = cleaned.head(count).iterrows()
    else:
        row_iter = cleaned.iterrows()
    return [row["path"] for idx, row in row_iter]


def create_zip_from_paths(paths, zip_name):
    # Make a temp dir to store things
    tmp_dir = os.path.splitext(zip_name)[0]
    os.makedirs(tmp_dir)
    # Copy images to tmp dir
    for path in paths:
        shutil.copy2(path, tmp_dir)
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                zipf.write(os.path.join(root, file))
    shutil.rmtree(tmp_dir)


def join_and(l):
    """
    Joins a list of boolean series with &
    """

    criteria = l.pop()
    while l:
        new_crit = l.pop()
        criteria = criteria & new_crit
    return criteria


def get_params_from_image_id(image_id):
    vars_to_extract = {
        "color": str,
        "led-brightness": float,
        "exposure": int,
        "color-temp": int,
    }
    data = {}
    params = image_id.split("_")
    data["location"] = params[0]
    data["orientation"] = params[-1]
    for p in params[1:-1]:
        for k, converter in vars_to_extract.items():
            if p.startswith(k):
                val = p.replace(k, "")
                if val[0] == "-":
                    continue
                try:
                    data[k] = converter(val)
                except ValueError:
                    data[k] = val
    
    if data["color-temp"] == "auto":
        data["autobalance"] = True
        data["color-temp"] = 0
    else:
        data["autobalance"] = False
    return data


def get_record_matching_dict(df, d, unique=True):
    """
    Get a record from a pandas dataframe where column values match values in a dict. 
    Keys in dict must be valid column names in dataframe.
    
    If no records are found matching condition, returns False
    If unique=True, assumes dict will select a single, unique record and throws 
    ValueError if more than one record is selected 
    """
    print(d)
    record = df[join_and([df[k] == v for k, v in d.items()])]
    if len(record) == 0:
        return False
    if unique and len(record) != 1:
        raise ValueError(f"Returned multiple records: {record}")
    return record.iloc[0]


def display_image(df, base=None, plot=False, **kwargs):
    record = get_record_matching_dict(df, kwargs, unique=True)
    if isinstance(record, bool) and not record:
        display(f"####\nNO RECORD MATCHING CONDITIONS: {kwargs}")
        return
    if base is not None:
        img_path = os.path.join(base, record["image_id"] + ".jpg")
    else:
        img_path = record["image_id"] + ".jpg"
    display(img_path, Image(img_path))#, f"Value Average: {record['value_mean']}")
    if plot:
        img = cv2.imread(img_path)
        max_bgr = plot_hist_bgr(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        max_hsv = plot_hist_hsv(img_hsv)


def build_widgets(df, img_dir="./test_images/", plot=False):
    widgets = {}
    # Maps dataframe keys to the widget we want to use
    widget_map = {
        "location": (ipyw.Dropdown, str),
        "autobalance": (ipyw.Checkbox, bool),
        "color": (ipyw.SelectionSlider, str),
        "led-brightness": (ipyw.SelectionSlider, float),
        "exposure": (ipyw.SelectionSlider, int),
        "color-temp": (ipyw.SelectionSlider, int),
        "orientation": (ipyw.SelectionSlider, str),
    }
    style = {"description_width": "initial", "value_width": "initial"}
    layout = ipyw.Layout(width="400px")
    for k in df.columns:
        if k not in widget_map:
            continue
        dtype_str = str(df[k].dtype)
        widget_class, converter = widget_map[k]
        value = converter(df[k].iloc[0])
        if "int" in dtype_str or dtype_str == "float64":
            widget_kwargs = {
                "options": sorted(df[k].unique()),
                "description": k,
                "disabled": False,
                "value": value,
            }
        elif dtype_str == "bool":
            widget_kwargs = {
                "value": True,
                "description": k,
                "indent": True,
                "disabled": False,
                "value": value,
            }
        else:
            widget_kwargs = {
                "options": sorted(df[k].unique()),
                "description": k,
                "disabled": False,
                "value": value,
            }
        widgets[k] = widget_class(**widget_kwargs, layout=layout)

    # Build buttons for tagging images
    good_button = ipyw.Button(description="Good", button_style="success")
    bad_button = ipyw.Button(description="Bad", button_style="danger")

    left_box = ipyw.VBox(
        [widgets["location"], widgets["autobalance"], widgets["color"]]
    )
    middle_box = ipyw.VBox(
        [widgets["led-brightness"], widgets["exposure"], widgets["orientation"]]
    )
    right_box = ipyw.VBox([widgets["color-temp"], good_button, bad_button])
    ui = ipyw.HBox([left_box, middle_box, right_box])
    out = interactive_output(
        display_image,
        {"df": fixed(df), "base": fixed(img_dir), "plot": fixed(plot), **widgets},
    )

    @out.capture()
    def mark_img_good(b):
        with out:
            record = get_record_matching_dict(
                df, {k: widg.value for k, widg in widgets.items()}
            )
            img_name = record["image_id"] + ".jpg"
            with open("image_results.txt", "a") as f:
                f.write(f"{img_name},usable\n")

    @out.capture()
    def mark_img_bad(b):
        with out:
            record = get_record_matching_dict(
                df, {k: widg.value for k, widg in widgets.items()}
            )
            img_name = record["image_id"] + ".jpg"
            with open("image_results.txt", "a") as f:
                f.write(f"{img_name},unusable\n")

    good_button.on_click(mark_img_good)
    bad_button.on_click(mark_img_bad)
    return ui, out

def manually_verify_images(df, skip_existing=True, write=True):
    if os.path.isfile("manually_verified_images.txt"):
        with open("manually_verified_images.txt", "r") as f:
            results = {}
            for line in f.readlines():
                k, v = line.strip().split(",")
                results[k] = v
    else:
        results = {}
    print("Number of existing results: {}".format(len(results)))
    with open("manually_verified_images.txt", "a") as f:
        for idx, row in df.iterrows():
            print(row["path"])
            if skip_existing:
                if row["path"] in results:
                    continue
            display(Image(row["path"]))
            result = yes_or_no("")
            if write:
                if result:
                    line = "{},{}\n".format(row["path"], "good")
                    results[row["path"]] = "good"
                else:
                    line = "{},{}\n".format(row["path"], "bad")
                    results[row["path"]] = "bad"
                f.write(line)
            clear_output()