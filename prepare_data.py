from utils import *


def prepare_data(
        tokenizer,
        dataset,
        output_file="data/pii-masking-200k-splited/english_pii_43k-splited-relabel.jsonl"
    ):
    """根据原始数据中的标签，使用我们自己的tokenizer重新标注数据集，生成新的标签文件。
       新增的label包括[PS]（隐私内容开始前的那个token）,[PII]（隐私内容token）以及 
       [FO]（非隐私内容token且在隐私之前出现）[BO](非隐私内容且在隐私之后出现)"""
    def relabel_data(example):
        def find_sub_index(seq1, seq2) -> int:
            len1 = len(seq1)
            len2 = len(seq2)
            if not isinstance(seq1, torch.Tensor):
                seq1 = torch.tensor(seq1)
            if not isinstance(seq2, torch.Tensor):
                seq2 = torch.tensor(seq2)
            if len1 == 0 or len2 > len1:
                return -1
            for index in range(0, len1-len2+1):
                if torch.equal(seq1[index: index+len2], seq2):
                    return index
            return -1
        text = example["source_text"]
        span_labels = json.loads(example["span_labels"])
        inputs = tokenizer(text, return_tensors="pt")
        # print("Text tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        span_labels = [l for l in span_labels if l[2] != "O"]
        new_labels = ["[BO]"] * inputs["input_ids"].shape[1]
        for span in span_labels:
            target_token_indices = find_token_indices_by_char_span(text, span, tokenizer)
            for _, index in enumerate(target_token_indices):
                if index > 0 and _ == 0:
                    new_labels[index - 1] = "[PS]"
                new_labels[index] = "[PII]"
        for i in range(len(new_labels)):
            if new_labels[i] == "[BO]":
                new_labels[i] = "[FO]"
            else:
                break
        return {"new_labels": new_labels}

    dataset = dataset.map(relabel_data, num_proc=1, batched=False)
    dataset.to_json(output_file)
    print("=====Done=====\nExample:\n", json.dumps(dataset[0], indent=4))


def find_token_indices_by_char_span(full_text, span, tokenizer):
    """
    通过字符位置定位 Token 索引，返回的是目标短语对应的 token indices 列表，一般应该是连续的
    """
    # 1. 找到目标短语在原始字符串中的字符起止位置
    # 注意：如果短语出现多次，这里默认找第一个。如果需要找特定的，需要你提供具体位置。
    start_char, end_char = int(span[0]), int(span[1])
    
    # 2. Tokenize 并获取 offset_mapping
    # return_offsets_mapping=True 是关键
    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    
    # 3. 遍历 offset_mapping 找到对应的 tokens
    # offsets 的格式是 [(0, 3), (3, 5)...] 表示每个 token 对应的字符区间
    offsets = encoding.offset_mapping[0] 
    input_ids = encoding.input_ids[0]
    
    target_token_indices = []
    
    for idx, (token_start, token_end) in enumerate(offsets):
        # 跳过特殊字符（如 BOS <s>），它们的 offset 通常是 (0,0)
        if token_start == token_end:
            continue
            
        # 判断当前 token 是否与目标字符区间有重叠
        # 逻辑：token 的结束位置 > 目标的开始位置 AND token 的开始位置 < 目标的结束位置
        if token_end > start_char and token_start < end_char:
            target_token_indices.append(idx)
            # 打印调试看一下找到了什么
            # print(f"Match: Index {idx}, Token: {tokenizer.decode([input_ids[idx]])}, Offset: {token_start}-{token_end}")

    return target_token_indices

def split_sentences(
        dataset_file,
        output_file="data/pii-masking-200k-splited/english_pii_43k-splited.jsonl"
):
    """将原始数据集中的长句子按照句号切分，同时保证标签正确拆分，从而扩充了数据集"""
    with open(dataset_file, "r") as f:
        lines = f.readlines()
    id_num = 0
    
    for l in tqdm(lines):
        piece = json.loads(l)
        source_text = piece["source_text"]
        span_labels = json.loads(piece["span_labels"])
        dot_indexs = []
        for i, c in enumerate(source_text):
            if c == ".":
                if i < len(source_text)-1 and (source_text[i+1] == "." or '0' <= source_text[i+1] <= '9'): # 排除数字小数点和英文省略号的情况
                    continue
                else:
                    dot_indexs.append(i)

        span_labels_run = deepcopy(span_labels)
        tail = 0 # 拆分句子后的长度差，要减掉
        for i, dot_index in enumerate(dot_indexs):
            sen = source_text[dot_indexs[i-1]+1 if i>0 else 0: dot_index+1] # 拆出的新句子
            sen_span_labels = [] # 新句子的spans
            for span in span_labels_run:
                if span[0] <= dot_index < span[1]:
                    new_span1, new_span2 = [span[0] - tail, dot_index+1 - tail, span[2]], [dot_index+1, span[1], span[2]]
                    sen_span_labels.append(new_span1)
                    span_labels.remove(span)
                    span_labels.insert(0, new_span2)
                    tail = new_span2[0]
                    break
                else:
                    sen_span_labels.append([span[0]-tail, span[1]-tail, span[2]])
                    span_labels.remove(span)
            span_labels_run = deepcopy(span_labels)
            if sen.startswith(" "): # 去掉开头的空格
                sen = sen[1:]
                for i in range(len(sen_span_labels)):
                    if i>0:
                        sen_span_labels[i][0] -= 1
                    sen_span_labels[i][1] -= 1

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {
                        "id": id_num,
                        "source_text": sen,
                        "span_labels": json.dumps(sen_span_labels)
                    },
                    ensure_ascii=False
                ) + "\n" )
                id_num += 1



if __name__ == "__main__":

    # relabel 原始数据集
    # model_path="/model/fangly/mllm/ljd/models/Meta-Llama-3-8B"
    # dataset_file="/model/fangly/mllm/ljd/dataset/pii-masking-200k/english_pii_43k.jsonl"
    # dataset = load_dataset("json", data_files=dataset_file, split="train")
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # # dataset = dataset.select(range(5000))
    # prepare_data( tokenizer, dataset,"data/relabeled_data.jsonl")


    # 将原始数据集做拆分
    # dataset_file="/model/fangly/mllm/ljd/dataset/pii-masking-200k/english_pii_43k.jsonl"
    # output_file="data/pii-masking-200k-splited/english_pii_43k-splited.jsonl"
    # split_sentences(dataset_file, output_file)


    # 使用拆分后的数据集进行relabel
    model_path="/model/fangly/mllm/ljd/models/Meta-Llama-3-8B"
    dataset_file="data/pii-masking-200k-splited/english_pii_43k-splited.jsonl"

    dataset = load_dataset("json", data_files=dataset_file, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prepare_data( tokenizer, dataset,output_file="data/pii-masking-200k-splited/english_pii_43k-splited-relabel.jsonl")


