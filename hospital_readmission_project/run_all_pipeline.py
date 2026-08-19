# -*- coding: utf-8 -*-
"""
run_all_pipeline.py
===================
MASTER RUNNER for the Hospital Readmission Risk Scorer.
Executes all steps of the pipeline in the correct order.
"""

import subprocess
import os
import sys
import time

# Set encoding for Windows console
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

def run_script(script_name):
    print(f"\n" + "="*70)
    print(f"  EXECUTING: {script_name}")
    print("="*70)
    
    # Use the local venv python
    # This assumes the script is run from the project root or the same folder as venv
    # Prefer the current project environment.  ``venv`` is retained as a
    # compatibility fallback for existing installations.
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python" # Fallback to system python

    try:
        start_time = time.time()
        # Use subprocess to run the script and capture output in real-time
        process = subprocess.Popen(
            [python_exe, script_name],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()
        
        duration = time.time() - start_time
        if process.returncode == 0:
            print(f"\n✅ {script_name} COMPLETED SUCCESSFULY ({duration:.1f}s)")
        else:
            print(f"\n❌ {script_name} FAILED with exit code {process.returncode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR running {script_name}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    pipeline_steps = [
        "data_preprocessing.py",
        "exploratory_analysis.py",
        "feature_engineering.py",
        "data_splitting.py",
        "model_training.py",
        "model_evaluation.py",
        "feature_importance.py",
        "demo_prediction.py"
    ]
    
    print("\n" + "#"*70)
    print("  HOSPITAL READMISSION RISK PIPELINE — FULL EXECUTION")
    print("#"*70)
    
    for step in pipeline_steps:
        run_script(step)
        
    print("\n" + "#"*70)
    print("  🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("  Check the 'reports/' and 'visualizations/' folders for results.")
    print("#"*70 + "\n")
