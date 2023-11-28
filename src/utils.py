import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # set logger level

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# fh = logging.FileHandler("run.log",mode='a',encoding='utf-8')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

SUMMARY_INSTRUCTION = "为这段摘要生成表示以用于检索相关文章"
QUERY_INSTRUCTION= "为这个问题生成表示以用于检索相关文章"
ANSWER_INSTRUCTION= "为这个答案生成表示以用于检索相关文章"

INSTRUCTION_DICT = {
    "content_query": QUERY_INSTRUCTION,
    "content_answer": ANSWER_INSTRUCTION,
    "content_summary": SUMMARY_INSTRUCTION
}

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_PATH,"data")
MODEL_PATH = os.path.join(PROJECT_PATH,"model_output")
