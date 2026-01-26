import psutil
import time
import logging
import os
from datetime import datetime

def monitor_resource_usage(log_dir: str = "logs", interval: int = 5):
    """
    Monitors system CPU and memory usage and logs it to a file.

    Args:
        log_dir (str): The directory where logs will be saved.
        interval (int): The time interval (in seconds) between measurements.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(log_dir, f"resource_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print to console
        ]
    )

    logging.info("--- Starting Resource Monitor ---")
    logging.info(f"Logging CPU and Memory usage every {interval} seconds...")
    logging.info("Press Ctrl+C to stop.")

    try:
        while True:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Log the information
            logging.info(
                f"CPU: {cpu_percent:>5.1f}% | "
                f"Memory Used: {memory.percent:>5.1f}% | "
                f"Available Memory: {memory.available / (1024**3):.2f} GB"
            )
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("\n--- Stopping Resource Monitor ---")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    monitor_resource_usage()