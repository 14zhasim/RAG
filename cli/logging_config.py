import logging
from lib.search_utils import LOGS_PATH

def setup_logging(debug: bool = False):
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=f'{LOGS_PATH}/RAG.log',
            filemode='w',
            encoding='utf-8',
            format="{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )

