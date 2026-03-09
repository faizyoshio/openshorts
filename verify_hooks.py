import os
import sys

try:
    from hooks import create_hook_image
except ImportError:
    print("PIL/Pillow not found locally. Run inside Docker or install requirements.")
    sys.exit(1)


def verify():
    print("Verifying hook image generation...")

    test_text = "POV: You are testing the viral hook feature\nand it works perfectly."
    output_path = "test_hook.png"
    target_width = 800

    try:
        path, width, height = create_hook_image(test_text, target_width, output_image_path=output_path)
        print(f"PASS: image generated at {path}")
        print(f"      dimensions: {width}x{height}")

        if not os.path.exists(path):
            print("FAIL: file does not exist")
            return False

        if os.path.getsize(path) == 0:
            print("FAIL: file is empty")
            return False

        print("PASS: verification successful")
        return True
    except Exception as exc:
        print(f"FAIL: verification failed: {exc}")
        return False
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    verify()
