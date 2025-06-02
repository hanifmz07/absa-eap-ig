import os
import json
from typing import Set, List, Dict, Any
from tqdm import tqdm
import langid
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import random

def is_not_english(text: str) -> bool:
    """
    Checks whether the given text is not in English using langid classification.
    
    Args:
        text (str): Input text to classify.
    
    Returns:
        bool: True if the text is not classified as English, or if detection fails.
    """
    try:
        lang, _ = langid.classify(text)
        return lang != "en"
    except Exception:
        return True  # Assume non-English if classification fails

def filter_and_save_non_english_articles(
    input_dir: str,
    output_file: str,
    min_word_count: int = 10
) -> None:
    """
    Filters out English and duplicate Wikipedia articles from a directory and 
    writes the cleaned non-English content to a JSONL file.
    
    Args:
        input_dir (str): Directory containing WikiExtractor JSON files (e.g., wiki_*).
        output_file (str): Path to write the filtered JSONL output.
        min_word_count (int, optional): Minimum number of words to retain an article. Defaults to 10.
    
    Returns:
        None
    """
    seen_texts: Set[str] = set()
    files = sorted(f for f in os.listdir(input_dir) if f.startswith("wiki_"))

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in tqdm(files, desc="Filtering + Deduplicating"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    try:
                        article = json.loads(line)
                        text = article.get("text", "").strip()

                        if len(text.split()) > min_word_count and is_not_english(text):
                            if text not in seen_texts:
                                seen_texts.add(text)
                                out_f.write(json.dumps(article, ensure_ascii=False) + "\n")
                    except Exception:
                        continue


class SundaEmbeddingDataset(Dataset):
    """
    A PyTorch dataset for training token-level language models on Sundanese text.
    
    Each item is a tokenized sequence padded/truncated to a fixed max length.
    """

    def __init__(
        self,
        text_list: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 128,
        shuffle: bool = False,
        seed: int = 42
    ):
        """
        Initializes the dataset.

        Args:
            text_list (List[str]): A list of raw text strings.
            tokenizer (PreTrainedTokenizerBase): A HuggingFace-compatible tokenizer.
            max_len (int): Maximum token length per sequence.
            shuffle (bool): Whether to shuffle the input data.
            seed (int): Random seed used for shuffling.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = text_list

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx].strip()

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        tokens = encoding["input_ids"].squeeze()  # shape: [max_len]

        return {
            "tokens": tokens
        }

def load_sundanese_paragraphs(jsonl_path: str, min_words: int = 5) -> list[str]:
    """
    Loads non-English paragraphs from a JSONL file and filters short ones.

    Args:
        jsonl_path (str): Path to filtered wiki JSONL file.
        min_words (int): Minimum number of words per paragraph to include.

    Returns:
        list[str]: List of filtered paragraph strings.
    """
    all_sentences = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for paragraph in obj["text"].split("\n"):
                paragraph = paragraph.strip()
                if len(paragraph.split()) > min_words:
                    all_sentences.append(paragraph)
    return all_sentences