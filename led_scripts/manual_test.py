import test
import argparse
import requests

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--brightness", required=True, type=float, help="Brightness of LEDs")
    p.add_argument("-o", "--orientation", required=True, type=str, help="Orientation of LEDs")
    p.add_argument("-e", "--exposure", required=True, type=int, help="Exposure of camera")
    p.add_argument("-c", "--color", required=True, type=str, help="Color of LEDs")
    p.add_argument("-t", "--temp", required=True, type=int, help="Color temperature of camera")
    p.add_argument("-n", "--name", type=str, help="Name of snapshot")
    args = p.parse_args()



    strip = test.LEDControl(46, args.brightness)
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

    strip.clear_strip()
    func = dispatcher[args.orientation]
    func_name = func.__name__
    print(f"Setting config {func_name}")
    if "back-dim" in func_name:
        func(args.color, back_dimmest=True)
    else:
        func(args.color)
    disable_auto_white_balance()
    set_absolute_exposure(args.exposure)
    set_color_temperature(args.temp)
    input("Press enter to continue to take snapshot")

    if args.name:
        fname = args.name
    else:
        fname = "manual_led-brightness{}_exposure{}_color-temp{}_color{}_{}.jpg".format(args.brightness, args.exposure, args.temp, args.color, args.orientation)
    take_snapshot(fname)


    print("Clearing and deinitializing strip")
    strip.clear_strip()
    strip.deinit()

if __name__ == "__main__":
    main()
