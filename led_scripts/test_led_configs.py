import test
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("color", help="Color of LEDs")
    p.add_argument("-b", "--brightness", type=float, help="Brightness of LEDs")
    args = p.parse_args()



    strip = test.LEDControl(46, 1)
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

    for func_name, func in dispatcher.items():
        strip.clear_strip()
        print(f"Setting config {func_name}")
        print(func)
        print(func.__name__)
        if "back-dim" in func_name:
            func(args.color, back_dimmest=True)
        else:
            func(args.color)
        input("Press enter to continue to next configuration")


    print("Clearing and deinitializing strip")
    strip.clear_strip()
    strip.deinit()

if __name__ == "__main__":
    main()
