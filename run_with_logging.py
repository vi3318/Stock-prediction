"""
Wrapper script to run train_full_system.py with simultaneous console and file logging
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log file path
    log_file = log_dir / "full_training.log"
    
    print(f"Starting training with logging to: {log_file}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Build command with unbuffered output
    cmd = [
        sys.executable,
        "-u",  # Unbuffered output
        "scripts/train_full_system.py",
        "--ticker", "AAPL",
        "--days", "1000",
        "--n-trials", "100"
    ]
    
    # Run with logging
    with open(log_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()
        
        # Run process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0  # No buffering
        )
        
        # Read and display/log output line by line
        for line in process.stdout:
            # Print to console with immediate flush
            print(line, end='', flush=True)
            # Write to file with immediate flush
            f.write(line)
            f.flush()
        
        # Wait for completion
        return_code = process.wait()
    
    print()
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")
    
    if return_code == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with exit code {return_code}")
        print(f"Check {log_file} for details")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())
