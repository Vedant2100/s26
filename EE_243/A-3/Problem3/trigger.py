import modal
import sys
from pathlib import Path

# Look up the deployed function
try:
    f = modal.Function.lookup("ee243-problem3", "run_pipeline")
except Exception as e:
    print(f"Error looking up function: {e}")
    sys.exit(1)

# Read the zip file
zip_path = Path("/Users/EndUser/Downloads/Repos/s26/EE_243/A-3/Problem3/problem3_data.zip")
if not zip_path.exists():
    print(f"Error: Could not find {zip_path}")
    sys.exit(1)

zip_bytes = zip_path.read_bytes()

print("Triggering the pipeline asynchronously...")
# Spawn it in the background
call = f.spawn(zip_bytes, iterations=30000)

print(f"Success! Task spawned asynchronously with ID: {call.object_id}")
print("You can safely close your laptop or turn off your computer.")
print("To download the results later, run:")
print("modal volume get ee243-results problem3_results.zip .")
