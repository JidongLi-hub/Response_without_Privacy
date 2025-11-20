import json
import pandas as pd
import requests
import urllib.parse
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from transformers import AutoTokenizer
import numpy as np

def json_to_jsonl(json_file_path):
    """
    将包含JSON对象数组的JSON文件转换为JSONL文件。

    :param json_file_path: 输入的JSON文件路径。
    :param jsonl_file_path: 输出的JSONL文件路径。
    """
    try:
        with open(json_file_path, 'r') as json_file:
            # 从JSON文件中加载数据
            data = json.load(json_file)

            # 确保数据是一个列表
            if not isinstance(data, list):
                print("错误：JSON文件的内容不是一个列表 (数组)。")
                return

        with open(json_file_path.replace(".json",".jsonl"), 'w') as jsonl_file:
            # 遍历列表中的每一个JSON对象
            for entry in data:
                # 将每个对象转换为JSON字符串并写入文件，然后添加换行符
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"成功将 '{json_file_path}' 转换")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{json_file_path}'。")
    except json.JSONDecodeError:
        print(f"错误：无法解析 '{json_file_path}' 的内容。请确保它是一个有效的JSON文件。")
    except Exception as e:
        print(f"发生未知错误: {e}")

def extract_new_sequences_from_VM():
    # 提取逐字记忆文中提供的新序列
    file_path = "./data/injection_data_url.csv"
    output_path = "./data/injection_data.csv"
    df = pd.read_csv(file_path, header=None)

    # 启动 Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 无界面模式
    driver = webdriver.Chrome(options=options)
    print("Web is ready!")

    seqs = []
    for index, row in df.iterrows():
        print(f"Processing: No.{index}")
        driver.get(row[0])
        driver.refresh()  # 🔥 强制重新加载，因为#后面的哈希部分在浏览器看来是同一个页面，替换后不会触发重复加载，所以强制刷新
        time.sleep(2)
        # 等待加载并提取输入框内容（假设输入框是 <textarea>）
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        text = textarea.get_attribute("value")
        seqs.append(text)

    driver.quit()

    od = pd.DataFrame({"seqs":seqs})
    od.to_csv(
        output_path,
        header=True,
        index=False, 
        encoding="utf-8"
    )

def extract_Seqs_to_CSV(input_path="/model/fangly/mllm/ljd/Memory_or_Hallucination/data/new_WikiFactDiff1000.json"):
    dics = json.load(open(input_path, "r"))
    
    seqs = [dic["text"] for dic in dics]
    df = pd.DataFrame(
        {
            "seqs":seqs
        }
    )
    df.to_csv("data/injection_WIKI_data.csv", header=True,index=False)





if __name__ == "__main__":
    # json_to_jsonl("./new_WikiFactDiff.json")
    # extract_new_sequences_from_VM()
    extract_Seqs_to_CSV()
    