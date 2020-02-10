import json
import pandas
import os
import cv2
import subprocess
import sys
import zipfile
import shutil

import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image, display, TextDisplayObject
from pprint import pprint
from ipywidgets import fixed, interactive_output
from scipy import stats as scistats
from textwrap import dedent
from joblib import Parallel, delayed
    
font = cv2.FONT_HERSHEY_COMPLEX

def subprocess_wrap(fname, cmd, log=True, stream=False, **kwargs):
    """
    Write output of command to a file for later processing, stream it to stdout, and return the process object
    
    **kwargs are passedto subprocess.Popen
    """
    with open(fname, 'wb') as f:  # replace 'w' with 'wb' for Python 3
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, **kwargs)
        for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
            if stream:
                sys.stdout.write(line)
            if log:
                f.write(line)
    return process

def run_idealo_iqa(model, build_log="docker-build.log", rebuild=False, gpu=False):
    if model not in {"technical", "aesthetic"}:
        raise ValueError("Model arg must be either 'technical' or 'aesthetic'")
    if gpu:
        suffix = "gpu"
    else:
        suffix = "cpu"
    if rebuild:
        build_cmd = f"docker build -t nima-{suffix} . -f Dockerfile.{suffix}"
        proc = subprocess_wrap(build_log, build_cmd, log=True, stream=True, shell=True)
        
    weights = "weights_mobilenet_technical_0.11.hdf5" if model == "technical" else "weights_mobilenet_aesthetic_0.07.hdf5"
    predict_cmd = dedent(f"""
    ./predict  \
    --docker-image nima-{suffix} \
    --base-model-name MobileNet \
    --weights-file $(pwd)/models/MobileNet/{weights} \
    --image-source $(pwd)/test_images
    """)
    iqa_log = f"output-{model}.log"
    proc = subprocess_wrap(iqa_log, predict_cmd, log=True, stream=False, shell=True)


def _compute_image_stats_workers(img_path, channels):
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    stats = {}
    for ind, chan in channels.items():
        chan_arr = img_hsv[:, :, ind] 
        #print(k)
        #print(chan_arr.shape)
        #chan_max = np.amax(chan_arr)
        #print(f"{chan} max = {chan_max}")
        
        k = f"{chan}_mean"
        stats[k] = np.mean(chan_arr)
        
        #k = f"{chan}_mode"
        #mode_result = scistats.mode(chan_arr, axis=None)
        #stats[k] = mode_result.mode[0]
        #stats[k+"_count"] = mode_result.count[0]
    return stats


def compute_image_statistics_parallel(df, base="", channels=None, threads=True, n_jobs=-2):
    """
    Compute a bunch of image statistics and return a new dataframe containing them
    """
    
    if channels is None: 
        channels = {0: "hue", 1: "saturation", 2: "value"}
    paths = [os.path.join(base, img_id + ".jpg") for img_id in df["image_id"]]
    # This returns results in the same order as input args, so don't need to worry about
    # sorting
    if threads:
        par = Parallel(n_jobs=n_jobs, prefer="threads")
    else:
        par = Parallel(n_jobs=n_jobs, prefer="processes")
    results = par(delayed(_compute_image_stats_workers)(path, channels) for path in paths)
    stats = {k: [v] for k, v in results[0].items()}
    for d in results[1:]:
        for k, v in d.items():
            stats[k].append(v)
    return stats


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
            #print(k)
            chan_max = np.amax(chan_arr)
            #print(chan_arr.shape)
            #print(f"{chan} max = {chan_max}")
            if k in stats:
                stats[k].append(np.mean(chan_arr))
            else:
                stats[k] = [np.mean(chan_arr)]
            k = f"{chan}_mode"
            #mode_result = scistats.mode(chan_arr, axis=None)
            #if k in stats:
            #    stats[k].append(mode_result.mode[0])
            #    stats[k+"_count"].append(mode_result.count[0])
            #else:
            #    stats[k] = [mode_result.mode[0]]
            #    stats[k+"_count"] = [mode_result.count[0]]
    #for k, v in stats.items():
    #    df[k] = v
    return pandas.DataFrame(stats)
        

def plot_hist_hsv(img, bins=None):
    channel_map = {0: ("hue", 179), 1: ("saturation", 255), 2: ("value", 255)}
    if bins is None:
        bins = [360, 100, 100]
    if len(bins) != 3:
        raise ValueError("Must provide number of bins for all 3 channels")
    fig, axes = plt.subplots(1, 3, figsize=(12,6))
    axes = axes.flatten()
    max_vals = np.zeros(3)
    for channel in range(0, 3):
        title, max_val = channel_map[channel]
        x = np.linspace(0, max_val, bins[channel])
        histr = cv2.calcHist([img],[channel],None,[bins[channel]],[0,max_val]) 
        ind = np.argmax(histr)
        print("Most common {} value: {}".format(title, x[ind]))
        ax = axes[channel]
        ax.set_title(channel_map[channel])
        #ax.plot(histr)
        ax.bar(x, histr.flatten())
        max_vals[channel] = x[ind]
    plt.tight_layout()
    plt.show()
    return max_vals
    
def plot_hist_bgr(img, bins=256):
    channel_map = {0: "blue",1: "green",2: "red"}
    fig, axes = plt.subplots(1, 3, figsize=(12,6))
    axes = axes.flatten()
    x = np.linspace(0, 256, bins)
    max_vals = np.zeros(3)
    for channel in range(0, 3):
        histr = cv2.calcHist([img],[channel],None,[bins],[0,256]) 
        ind = np.argmax(histr)
        print("Most common {} value: {}".format(channel_map[channel], x[ind]))
        ax = axes[channel]
        ax.set_title(channel_map[channel])
        #ax.plot(histr)
        ax.bar(x, histr.flatten())
        max_vals[channel] = x[ind]
    plt.tight_layout()
    plt.show()
    return max_vals

def threshold_hue_in_range_hsv(img_hsv, min_val, max_val, invert=False):
    """
    Sets all pixels outside of a particular hue range to black by setting value channel to zero
    
    Inverted keyword inverts the logic. So everything *inside* the range is set to black and
    everything outside is left alone
    """
    hue = img_hsv[:, :, 0]
    value = img_hsv[:, :, 2]
    #threshed_hue = np.where((hue < min_val) | (hue > max_val), 0, hue)
    if invert:
        threshed_value = np.where((hue > min_val) & (hue < max_val), 0, value)
    else:
        threshed_value = np.where((hue < min_val) | (hue > max_val), 0, value)
    threshed_img_hsv = np.copy(img_hsv)
    #threshed_img_hsv[:, :, 0] = threshed_hue
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
        print(path)
        display(img)


def get_images_matching_criteria(df, base, criteria=None, count=None, sort_by="", ascending=True):
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
        cleaned = cleaned.sort_values(sort_by, ignore_index=True, ascending=ascending)
    if count is not None:
        row_iter = cleaned.head(count).iterrows()
    else:
        row_iter = cleaned.iterrows()
    paths = [] 
    for idx, row in row_iter:
        path = os.path.join(base, row["image_id"] + ".jpg")
        paths.append(path)
    return paths

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

def load_data(fname):
    with open(fname, 'r') as f:
        lines = [l for l in f.readlines() if not ("ETA:" in l or "ms/step" in l)]
        json_txt = ''.join(lines)
    scores = json.loads(json_txt)
    vars_to_extract = {
        "color": str,
        "led-brightness": float,
        "exposure": int,
        "color-temp": int,
    }
    for score in scores:
        params = score['image_id'].split("_")
        score["location"] = params[0]
        score["score"] = score["mean_score_prediction"]
        score["orientation"] = params[-1]
        del score["mean_score_prediction"]
        for p in params[1:-1]:
            for k, converter in vars_to_extract.items():
                if p.startswith(k):
                    val = p.replace(k, "")
                    if val[0] == "-":
                        continue
                    try:
                        score[k] = converter(val) 
                    except ValueError:
                        score[k] = val 
            
        if score["color-temp"] == "auto":
            score["autobalance"] = True
            score["color-temp"] = 0
        else:
            score["autobalance"] = False
    return scores

def get_record_matching_dict(df, d, unique=True):
    """
    Get a record from a pandas dataframe where column values match values in a dict. 
    Keys in dict must be valid column names in dataframe.
    
    If no records are found matching condition, returns False
    If unique=True, assumes dict will select a single, unique record and throws 
    ValueError if more than one record is selected 
    """
    print(d)
    record = df[join_and([df[k]==v for k, v in d.items()])]
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
    display(img_path, Image(img_path), f"Value Average: {record['value_mean']}")
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
            widget_kwargs = {"options": sorted(df[k].unique()), "description": k, "disabled": False, "value": value}
        elif dtype_str == "bool":
            widget_kwargs = {"value": True, "description": k, "indent": True, "disabled": False, "value": value}
        else:
            widget_kwargs = {"options": sorted(df[k].unique()), "description": k, "disabled": False, "value": value}
        widgets[k] = widget_class(**widget_kwargs, layout=layout) 
        
    # Build buttons for tagging images
    good_button = ipyw.Button(description="Good", button_style="success")
    bad_button = ipyw.Button(description="Bad", button_style="danger")
    
    left_box = ipyw.VBox([widgets["location"], widgets["autobalance"], widgets["color"]])
    middle_box = ipyw.VBox([widgets["led-brightness"], widgets["exposure"], widgets["orientation"]])
    right_box = ipyw.VBox([good_button, bad_button])
    ui = ipyw.HBox([left_box, middle_box, right_box])
    out = interactive_output(display_image, {"df": fixed(df), "base": fixed(img_dir), "plot": fixed(plot), **widgets})
    
    @out.capture()
    def mark_img_good(b):
        with out:
            record = get_record_matching_dict(df, {k: widg.value for k, widg in widgets.items()})
            img_name = record["image_id"] + ".jpg"
            with open("image_results.txt", "a") as f:
                f.write(f"{img_name},usable\n")
            
    @out.capture()
    def mark_img_bad(b):
        with out:
            record = get_record_matching_dict(df, {k: widg.value for k, widg in widgets.items()})
            img_name = record["image_id"] + ".jpg"
            with open("image_results.txt", "a") as f:
                f.write(f"{img_name},unusable\n")
    good_button.on_click(mark_img_good) 
    bad_button.on_click(mark_img_bad)
    return ui, out