import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_dir,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_dir,LOG_FILE)

# Configure logging with rotation (max 5MB per file, keep 5 backup files)
handler = RotatingFileHandler(
    LOG_FILE_PATH,
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5
)

logging.basicConfig(
    handlers=[handler],
    format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s",
    level=logging.INFO
)