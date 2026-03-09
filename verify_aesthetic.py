import os
import sys

try:
    from hooks import create_hook_image
except ImportError:
    print("PIL/Pillow not found locally. Run inside Docker or install requirements.")
    sys.exit(1)


def verify():
    print("Verifying hook aesthetics...")

    test_text = "POV: You are testing\nthe new aesthetic feature\nwith explicit lines."
    output_path = "aesthetic_hook.png"
    target_width = 800

    try:
        path, width, height = create_hook_image(test_text, target_width, output_image_path=output_path)
        print(f"PASS: image generated at {path}")
        print(f"      dimensions (with shadow): {width}x{height}")

        if not os.path.exists(path):
            print("FAIL: output file does not exist")
            return False

        print("PASS: basic generation check complete (inspect image visually)")
        return True
    except Exception as exc:
        print(f"FAIL: verification error: {exc}")
        return False


if __name__ == "__main__":
    verify()
