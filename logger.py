import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tạo handler để ghi log vào console
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Định dạng log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Thêm handler vào logger
logger.addHandler(handler)
