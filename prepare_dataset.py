from transformers import PreTrainedTokenizerBase
from typing import List, Tuple, Optional
from utils import ContrastivePair, return_default_suffixes

class SteeringDataset:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        examples: List, 
        suffixes: List[Tuple[str, str]]=None,
        disable_suffixes:bool=False,
        use_chat_template:bool=True,
        system_message:Optional[Tuple[str, str]]=None
    ):
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.suffixes = suffixes
        self.preformatted_dataset = []
        self.formatted_dataset = []

        print(f"Processing {len(examples)} examples.")

        for ex in examples:
            if self.use_chat_template:
                if system_message:
                    message_a = [{"role": "system", "content": f"{system_message[0]}"}, {"role": "user", "content": f"{self.clean_text(ex[0])}"}]
                    message_b = [{"role": "system", "content": f"{system_message[1]}"}, {"role": "user", "content": f"{self.clean_text(ex[1])}"}]
                else:
                    message_a = [{"role": "user", "content": f"{self.clean_text(ex[0])}"}]
                    message_b = [{"role": "user", "content": f"{self.clean_text(ex[1])}"}]
                positive = tokenizer.apply_chat_template(message_a, tokenize=False, add_generation_prompt=False)
                negative = tokenizer.apply_chat_template(message_b, tokenize=False, add_generation_prompt=False)
            else:
                positive = self.clean_text(ex[0])
                negative = self.clean_text(ex[1])

            self.preformatted_dataset.append(
                ContrastivePair(positive=positive, negative=negative)
            )

        print(f"Processed {len(self.preformatted_dataset)} examples")

        # Handle suffixes
        if suffixes is not None and not disable_suffixes and isinstance(suffixes[0], tuple):
            for pos_suffix, neg_suffix in suffixes:
                for pair in self.preformatted_dataset:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + pos_suffix,
                            negative=pair.negative + neg_suffix
                        )
                    )
        elif suffixes is not None and not disable_suffixes and isinstance(suffixes[0], str):
            for suffix in suffixes:
                for pair in self.preformatted_dataset:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + suffix,
                            negative=pair.negative + suffix
                        )
                    )
        elif suffixes is None and not disable_suffixes:
            default_suffixes = return_default_suffixes()
            for suffix in default_suffixes:
                for pair in self.preformatted_dataset:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + suffix,
                            negative=pair.negative + suffix
                        )
                    )
        else:
            self.formatted_dataset = self.preformatted_dataset

        print(f"Final dataset contains {len(self.formatted_dataset)} examples.")
        print("For example:\n")
        print(f"Positive: {self.formatted_dataset[0].positive}\nNegative: {self.formatted_dataset[0].negative}\n")


    def clean_text(self, text: str) -> str:
        """
        Clean the input text by replacing special tokens.这里的输入还没有tokenize过,说明是对原始文本中可能存在的特殊token进行替换,以便和之后的特殊符号区分

        Args:
            text: The input text to be cleaned.

        Returns:
            The cleaned text with special tokens replaced.
        """
        if not text:
            return text

        def insert_vline(token: str) -> str:
            if len(token) < 2:
                return " "
            elif len(token) == 2:
                return f"{token[0]}|{token[1]}"
            else:
                return f"{token[:1]}|{token[1:-1]}|{token[-1:]}"

        if self.tokenizer.bos_token:
            text = text.replace(self.tokenizer.bos_token, insert_vline(self.tokenizer.bos_token))
        if self.tokenizer.eos_token:
            text = text.replace(self.tokenizer.eos_token, insert_vline(self.tokenizer.eos_token))
        if self.tokenizer.pad_token:
            text = text.replace(self.tokenizer.pad_token, insert_vline(self.tokenizer.pad_token))
        if self.tokenizer.unk_token:
            text = text.replace(self.tokenizer.unk_token, insert_vline(self.tokenizer.unk_token))

        return text