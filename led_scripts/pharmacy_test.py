# Camera settings
# Brightness
#          value: 0
#          min: -56
#          max: 56
#          step: 1
#          default: 0
# Contrast
#          value: 0
#          min: 0
#          max: 95
#          step: 1
#          default: 0
# Hue
#          value: 0
#          min: -2000
#          max: 2000
#          step: 1
#          default: 0
# Saturation
#          value: 25
#          min: 0
#          max: 100
#          step: 1
#          default: 25
# Sharpness
#          value: 2
#          min: 1
#          max: 7
#          step: 1
#          default: 2
# Gamma
#          value: 100
#          min: 100
#          max: 300
#          step: 1
#          default: 100
# White Balance temperature
#          value: 4600
#          min: 2200
#          max: 7500
#          step: 1
#          default: 4600

import board
import neopixel
import random
import requests
import time
import glob

from pprint import pprint
from itertools import product, repeat

#NUM_LEDS = 13
#pixels[0] = (255, 0, 0)

#import code
#code.interact(local=locals())


class LEDControl:
    """
    Create strip object with `num_leds` LEDs and `brightness` brightness level

    brightness between 0 and 1
    """

    # http://planetpixelemporium.com/tutorialpages/light.html
    COLOR_DICT = {
            "yellow": (252, 202, 3),
            "light-yellow": (255, 202, 45),
            # yellow-*k RGB values computed using:
            # https://academo.org/demos/colour-temperature-relationship/
            # Inspired by this reddit post:
            # https://www.reddit.com/r/Damnthatsinteresting/comments/eyuckf/a_comparison_of_light_temperatures/
            "yellow-4000k": (255, 206, 166), 
            "yellow-5000k": (255, 228, 206),
            "yellow-5500k": (255, 237, 222),
            "tungsten": (255, 214, 170), 
            "white": (255, 255, 255),
            "high-noon-sun": (255, 255, 251),
            "carbon-arc": (255, 250, 244),
            "halogen": (255, 241, 224),
            "off": (0, 0, 0),
            "warm-flourescent": (255, 244, 229),
            "standard-flourescent": (244, 255, 250),
            "full-spectrum-flourescent": (255, 244, 242),
        }
    

    def __init__(self, num_leds, brightness):

        self.num_leds = num_leds
        self.brightness = brightness
        self.strip = neopixel.NeoPixel(board.D18, num_leds, brightness=brightness)#, auto_write=True)
        self.lslice = slice(0, 14)
        self.bslice = slice(14, 33)
        self.rslice = slice(33, num_leds)
        self.full_slice = slice(0, num_leds)

    def deinit(self):
        self.strip.deinit()

    def full_strip(self, color):
        rgb_tup = self.COLOR_DICT[color]
        self.strip[self.full_slice] = [rgb_tup for i in range(0, self.num_leds)]
        self.strip.show()

    def clear_strip(self):
        self.full_strip("off")
        self.strip.show()

    def left_strip_only(self, color):
        self.clear_strip()
        rgb_tup = self.COLOR_DICT[color]
        self.strip[self.lslice] =  [rgb_tup for i in range(0, self.lslice.stop)]
        self.strip.show()

    def right_strip_only(self, color):
        self.clear_strip()
        rgb_tup = self.COLOR_DICT[color]
        self.strip[self.rslice] =  [rgb_tup for i in range(self.rslice.start, self.rslice.stop)]
        self.strip.show()

    def back_strip_only(self, color):
        self.clear_strip()
        rgb_tup = self.COLOR_DICT[color]
        self.strip[self.bslice] =  [rgb_tup for i in range(self.bslice.start, self.bslice.stop)]
        self.strip.show()

    def _generate_tuples(self, color, N):
        """
        Generate a list of tuples that linearly fades from `color` to off in N steps
        """
        brightness_fraction = 1.0 / N
        color_tup = self.COLOR_DICT[color]
        tuples = []
        for i in range(N):
            local_factor = i*brightness_fraction
            tup = tuple(int(el*local_factor) for el in color_tup)
            tuples.append(tup)
        return tuples

    def left_strip_linear_fade(self, color, back_dimmest=False):
        num_leds = self.lslice.stop - self.lslice.start
        tuples = self._generate_tuples(color, num_leds)
        if back_dimmest:
            tuples.reverse()
        self.strip[self.lslice] = tuples
        self.strip.show()


    def right_strip_linear_fade(self, color, back_dimmest=False):
        num_leds = self.rslice.stop - self.rslice.start
        tuples = self._generate_tuples(color, num_leds)
        if not back_dimmest:
            tuples.reverse()
        self.strip[self.rslice] = tuples
        self.strip.show()

    def left_right_linear_fade(self, color, back_dimmest=False):
        self.left_strip_linear_fade(color, back_dimmest=back_dimmest)
        self.right_strip_linear_fade(color, back_dimmest=back_dimmest)
        self.strip.show()

    def linear_fade_towards_back_middle(self, color):
        middle_ind = self.num_leds // 2
        left_half_tups = self._generate_tuples(color, middle_ind)
        left_half_tups.reverse()
        if self.num_leds % 2 != 0:
            right_half_tups = self._generate_tuples(color, middle_ind + 1)
        else:
            right_half_tups = self._generate_tuples(color, middle_ind)
        self.strip[self.full_slice] = left_half_tups + right_half_tups
        self.strip.show()

    def update_brightness(self, brightness):
        self.strip.brightness = brightness
        self.strip.show()


def focus_camera():
    print("Focusing camera ...")
    resp = requests.get("http://127.0.0.1:5000/focus_camera")
    print(f"Camera response: {resp}")
    print("Sleeping while focusing completes ...")
    time.sleep(FOCUS_SLEEP_SECONDS)


def take_snapshot(fname):
    print(f"Taking picture {fname} ...")
    resp = requests.get("http://127.0.0.1:5000/get_picture")
    print(f"Snapshot response: {resp}")
    with open(fname, "wb") as f:
        f.write(resp.content)

def change_camera_setting(name, val):
    payload = {"control_name": name, "control_value": val}
    resp = requests.post("http://127.0.0.1:5000/set_control", json=payload, headers={"Content-Type": "application/json"})
    return resp

def set_manual_exposure_mode():
    """
    See https://ken.tossell.net/libuvc/doc/group__ctrl.html#gafd6f20d317eb6793fd3555637eb8437a

    UVC_AUTO_EXPOSURE_MODE_MANUAL (1) - manual exposure time, manual iris
    UVC_AUTO_EXPOSURE_MODE_AUTO (2) - auto exposure time, auto iris
    UVC_AUTO_EXPOSURE_MODE_SHUTTER_PRIORITY (4) - manual exposure time, auto iris
    UVC_AUTO_EXPOSURE_MODE_APERTURE_PRIORITY (8) - auto exposure time, manual iris
    """
    
    return change_camera_setting("Auto Exposure Mode", 1)


def disable_auto_white_balance():
    """
    This setting takes values of 0 or 1. Assuming 0 is false. 1 is the default
    """

    print("******* DISABLING AUTOMATIC WHITE BALANCE ***********")
    return change_camera_setting("White Balance temperature,Auto", 0)

def enable_auto_white_balance():
    """
    This setting takes values of 0 or 1. Assuming 0 is false. 1 is the default
    """

    return change_camera_setting("White Balance temperature,Auto", 1)

def set_color_temperature(val):
    if not (2200 <= val <= 7500):
        raise ValueError(f"Value {val} outside of allowable range [2200, 7500]")
    return change_camera_setting("White Balance temperature", val)

def set_absolute_exposure(val):
    if not (3 <= val <= 2047):
        raise ValueError(f"Value {val} outside of allowable range [3, 2047]")
    return change_camera_setting("Absolute Exposure Time", val)


def set_brightness(val):
    if not (-56 <= val <= 56):
        raise ValueError(f"Value {val} outside of allowable range [-56, 56]")
    return change_camera_setting("Brightness", val)



def take_picture_with_params(strip, params, file_prefix):
    dispatcher = {
            "back-strip-only": strip.back_strip_only,
            "left-strip-only": strip.left_strip_only,
            "right-strip-only": strip.right_strip_only,
            "full-strip": strip.full_strip,
            "left-strip-linear-fade": strip.left_strip_linear_fade, 
            "left-strip-linear-fade-back-dim": strip.left_strip_linear_fade, 
            "right-strip-linear-fade": strip.right_strip_linear_fade, 
            "right-strip-linear-fade-back-dim": strip.right_strip_linear_fade, 
            "left-right-linear-fade": strip.left_right_linear_fade, 
            "left-right-linear-fade-back-dim": strip.left_right_linear_fade, 
            "linear-fade-towards-back-middle": strip.linear_fade_towards_back_middle,
            }
    print(f"Parameter set: {params}")
    for k, v in params.items():
        if k == "orientation":
            file_prefix += f"_{v}"
        else:
            file_prefix += f"_{k}{v}"
    print(f"file prefix = {file_prefix}")
    strip.clear_strip()
    orientation = params["orientation"]
    led_setup = dispatcher[orientation]
    # Update camera environment
    set_absolute_exposure(params["exposure"]) 
    strip.update_brightness(params["led-brightness"])
    if params["color-temp"] != "auto":
        set_color_temperature(params["color-temp"])
    print(f"Setting LED to {orientation}")
    if "back-dim" in orientation:
        led_setup(params["color"], back_dimmest=True)
    else:
        led_setup(params["color"])
    # Sometimes you need to focus a second time
    focus_camera()
    #focus_camera()
    pic_name = file_prefix + ".jpg"
    take_snapshot(pic_name)
    #input("Check orientation then press enter ...")


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
    return data


def get_combos_in_current_dir():
    imgs = glob.glob("*.jpg")
    params = [get_params_from_image_id(img.rstrip(".jpg")) for img in imgs]
    for p in params:
        del p["location"]
    return params


def get_bad_params_from_manually_verified_file(path):
    all_data = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [el.strip() for el in line.split(",")]
            print(line)
            if line[1] == "bad":
                image_id = os.path.split(line[0])[-1].rstrip(".jpg")
                print(image_id)
                all_data.append(get_params_from_image_id(image_id))
    return all_data


def get_bad_params_from_json_files(base):
    json_files = glob.glob(os.path.join(base, "*.json"))
    desired_keys = {'color', 'color-temp', 'exposure', 'led-brightness', 'location', 'orientation'}
    all_data = []
    for jsf in json_files:
        with open(jsf, "r") as f:
            data = json.load(f)
        for record in data:
            new_record = {k: v for k, v in record.items() if k in desired_keys}
            all_data.append(new_record)
    return all_data

    
NUM_LEDS = 46
FILE_PREFIX = "bottom"
FOCUS_SLEEP_SECONDS = 2.8

def main():

    set_manual_exposure_mode()
    strip = LEDControl(NUM_LEDS, 1)
    settings = {
            "color": 
                (#"yellow",
                 "light-yellow",
                 # Soft yellow colors
                 #"yellow-4000k", "yellow-5000k", "yellow-5500k",
                 # White colors
                 #"white", #"carbon-arc", "halogen", "warm-flourescent", "standard-flourescent", "full-spectrum-flourescent",
                 #"off"
                 ),
            "led-brightness": (1, .75, .5), 
            "exposure": (500, 750, 1000, 1500, 2000), 
            "color-temp": tuple(range(2200, 3800, 200)),
            "orientation": ("back-strip-only", "left-strip-only", "right-strip-only", "full-strip", "left-strip-linear-fade", "left-strip-linear-fade-back-dim", "right-strip-linear-fade", "right-strip-linear-fade-back-dim", "left-right-linear-fade", "left-right-linear-fade-back-dim", "linear-fade-towards-back-middle")
        }
    # First we auto white balance, so get all combos without sweeping through color temperature
    autobal_combos = list(product(*[zip(repeat(k), v) for k, v in settings.items() if k != "color-temp"]))
    all_combos = list(product(*[zip(repeat(k), v) for k, v in settings.items()]))
    # Make each unique param set a dict so access is more readable
    autobal_combos = [dict(c) for c in autobal_combos]
    for c in autobal_combos:
        c["color-temp"] = "auto"
    all_combos = [dict(c) for c in all_combos]
    # Some post-analysis I did indicated that all images taken with exposure at 2000 are over exposed 
    # except when only the right side is illuminated. Below filters out overexposed param sets
    #autobal_combos = [c for c in autobal_combos if not ((c["exposure"] == 2000) and ("right-strip" not in c["orientation"]))] 
    #all_combos = [c for c in all_combos if not ((c["exposure"] == 2000) and ("right-strip" not in c["orientation"]))] 
    # brightness = 1 and exposure = 1000 and "right-strip" not in orientation are bad
    #autobal_combos = [c for c in autobal_combos if not ((c["exposure"] == 1000) and (c["led-brightness"] == 1) and ("right-strip" not in c["orientation"]))] 
    #all_combos = [c for c in all_combos if not ((c["exposure"] == 1000) and (c["led-brightness"] == 1) and ("right-strip" not in c["orientation"]))] 

    # If we already tried some params and there are pics in the current directory, skip them
    existing_combos = get_combos_in_current_dir()
    pprint(existing_combos)
    #pprint(existing_combos)
    print("Initial number of combos to try: {}".format(len(all_combos)))
    print("Initial number of autobal combos to try: {}".format(len(autobal_combos)))
    all_combos = [c for c in all_combos if c not in existing_combos]
    autobal_combos = [c for c in autobal_combos if c not in existing_combos]
    print("Remaining number of combos after skipping existing: {}".format(len(all_combos)))
    print("Remaining number of autobal combos after skipping existing: {}".format(len(autobal_combos)))
    quit()

    # First test param set with auto white balance
    total_combos = len(all_combos)
    num_completed = 1
    #enable_auto_white_balance()
    #for combo in autobal_combos:
    #    print("#"*40)
    #    time_remaining = (total_combos - num_completed)*FOCUS_SLEEP_SECONDS
    #    print(f"Testing parameter set {num_completed} of {total_combos}. ETA: {time_remaining} seconds")
    #    take_picture_with_params(strip, combo, FILE_PREFIX) 
    #    num_completed += 1

    disable_auto_white_balance()
    for combo in all_combos:
        print("#"*40)
        time_remaining = (total_combos - num_completed)*FOCUS_SLEEP_SECONDS
        print(f"Testing parameter set {num_completed} of {total_combos}. ETA: {time_remaining} seconds")
        take_picture_with_params(strip, combo, FILE_PREFIX) 
        num_completed += 1
    strip.deinit()

if __name__ == "__main__":
    main()
