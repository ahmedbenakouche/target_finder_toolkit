import time
from target_finder_toolkit.mouse_utils import disable_mouse_acceleration, restore_mouse_acceleration

print("Phase A: before calling mouse_utils")
print("Test the mouse now for 15 seconds.")
time.sleep(15)

disable_mouse_acceleration()
print("Phase B: mouse_utils is now active")
print("Test the mouse again for 15 seconds.")
time.sleep(15)

restore_mouse_acceleration()
print("Restored.")