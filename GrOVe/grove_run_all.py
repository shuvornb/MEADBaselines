
import os
import subprocess
from datetime import datetime

# === Helper Function to Run Scripts ===
def run_script(script_path):
    print(f"🚀 Running: {script_path}")
    result = subprocess.run(['python3', script_path], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print(f"✅ Completed: {script_path}\\n")

# === List of Scripts (in Sequence) ===
scripts = [
    "grove_train_target.py",
    "grove_train_surrogates.py",
    "grove_train_independents.py",
    "grove_train_fingerprint.py"
]

# === Run All Scripts in Sequence ===
print("🎉 Starting Full GROVE Pipeline...")
for script in scripts:
    if os.path.exists(script):
        run_script(script)
    else:
        print(f"❌ Script not found: {script}")

print("🎉 All GROVE Experiments Completed.")

# === Save log of the run ===
log_path = f"grove_full_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_path, "w") as log_file:
    log_file.write("🎉 Full GROVE Pipeline Completed Successfully.\n")
print(f"✅ Log saved at: {log_path}")
