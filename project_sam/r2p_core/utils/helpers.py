import os
import json
import torch
import re
from typing import Dict
from PIL import Image
import re
from typing import List, Dict, Union
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from defined import myvlm_reverse_category_dict, yollava_reverse_category_dict, osc_reverse_category_dict

@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline."""
    category_name: str
    concept_name: str
    model_name: str = 'mini_cpm'
    device: str = "cuda"
    temperature: float = 0.2
    max_new_tokens: int = 512
    data_name: str = "YoLLaVA"
    seed: int = 42
    topK: int = 40
    refined_k: int = 5
    input_type: str = "text"
    
    # Flags
    rerank_early: bool = False
    step_by_step: bool = False
    generic: bool = False
    template_based: bool = False
    only_recall: bool = False
    pairwise: bool = False
    reason_only_on_text: bool = False
    attribute_based_step_by_step: bool = False
    two_step: bool = False

class Constants:
    """Constants used throughout the evaluation."""
    MAX_REASONING_WORDS = 16
    IGNORE_ATTRIBUTES = ['state', 'state consideration', 'distinctive marks', 'shape']
    OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


class PathManager:
    """Manages file paths for the evaluation pipeline."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
    
    def get_database_dir(self) -> str:
        """Get database directory path."""
        return os.path.join(get_db_path(self.config), self.config.category_name)
    
    def get_database_file(self) -> str:
        """Get database file path."""
        db_dir = self.get_database_dir()
        if self.config.template_based:
            return f'{db_dir}/database_user_defined_cat_template.json'
        return f'{db_dir}/database_user_defined_cat_no_template.json'
    
    def get_eval_file(self) -> str:
        """Get evaluation file path."""
        return (f'eval_files/{self.config.data_name}_seed_{self.config.seed}/'
                f'{self.config.category_name}/{self.config.concept_name}.json')
    
    def get_output_dir(self) -> str:
        """Get output directory path."""
        model_prefix = 'Mini_CPM' if self.config.model_name == 'mini_cpm' else 'QWEN'
        return (f"outputs/{model_prefix}_{self.config.data_name}_seed_{self.config.seed}/"
                f"{self.config.category_name}/{self.config.concept_name}")


class QuestionFormatter:
    """Handles question formatting and option generation."""
    
    @staticmethod
    def format_extra_info(extra_info: Dict, input_type: str = None) -> str:
        """Format extra information for model input."""
        extra_info_str = ""
        options = Constants.OPTIONS
        
        for i, (name, info) in enumerate(extra_info.items()):
            if input_type == 'text':
                option = options[i] if i < len(options) else f"Option {i+1}"
                extra_info_str += f"{option}.<image>\n Name: {name}, Info: {info}\n"
            elif input_type == 'both':
                extra_info_str += f"Image {i+2}.<image>\n Name: {name}, Info: {info}\n"
            else:
                extra_info_str += f"{i+1}.<image>\n Name: {name}, Info: {info}\n"
        
        return extra_info_str
    
    @staticmethod
    def get_category_string(config: EvalConfig) -> str:
        """Get category string based on data name."""
        if config.data_name == 'MyVLM':
            return myvlm_reverse_category_dict[config.concept_name]
        elif config.data_name == 'YoLLaVA':
            return yollava_reverse_category_dict[config.concept_name]
        elif config.data_name == 'PerVA':
            return osc_reverse_category_dict[config.category_name]
        return ""


class AnswerParser:
    """Handles parsing and validation of model answers."""
    
    @staticmethod
    def parse_option_answer(answer: str, option2num: Dict[str, int]) -> int:
        """Parse model answer and return option index."""
        import re
        import random
        
        answer = answer.lower().strip()
        
        if answer in option2num:
            return option2num[answer]
        
        # Try to extract answer from "therefore the answer is X" pattern
        try:
            match = re.search(r'therefore the answer is (\w)', answer)
            if match:
                extracted_answer = match.group(1)
                if extracted_answer in option2num:
                    return option2num[extracted_answer]
        except:
            pass
        
        # Fallback to random choice
        return random.randint(0, len(option2num) - 1)



def count_tokens(msgs: Union[str, List[str]], model_interface) -> Dict[str, int]:
    """
    Counts the number of text and image tokens in the input messages.
    
    Args:
        msgs (Union[str, List[str]]): The input messages to be tokenized.
        model_interface (ModelInterface): An instance of ModelInterface containing tokenizer and processor.
    
    Returns:
        Dict[str, int]: A dictionary with counts of text and image tokens.
    """
    if isinstance(msgs, str):
        msgs = [msgs]
    copy_msgs = deepcopy(msgs)
    images = []
    image_sizes = []
    for i, msg in enumerate(copy_msgs):
        role = msg["role"]
        content = msg["content"]
        assert role in ["system", "user", "assistant"]
        if i == 0:
            assert role in ["user", "system"], "The role of first msg should be user"
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                image_sizes.append(torch.tensor(c.size))
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
        
        msg["content"] = "".join(cur_msgs)
    prompts_lists = []
    prompts_lists.append(
            model_interface.processor.tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=None,
            )
        )
    image_sizes = [image_sizes]
    input_images_list = []
    input_images_list.append(images)
    image_tag = "(<image>./</image>)"
    image_pattern = "\(<image>./</image>\)"
    audio_tag = "(<audio>./</audio>)"
    audio_pattern = "\(<audio>./</audio>\)"
    split_pattern = f"({image_pattern}|{audio_pattern})"
    
    # for index, msg in enumerate(prompts_lists):
    text_chunks = re.split(split_pattern, prompts_lists[0])
    image_tags = re.findall(image_pattern, prompts_lists[0])
    image_token_count = 0
    modified_text_chunks = []
    image_id = 0
    for i, chunk in enumerate(text_chunks):
        if chunk == image_tag:
            image_placeholder = model_interface.image_processor.get_slice_image_placeholder(image_sizes[0][image_id], image_id, None, True)
            image_token_count += len(model_interface.tokenizer.encode(image_placeholder))
            modified_text_chunks.append(image_placeholder)
        else:
            modified_text_chunks.append(chunk)
    final_text = "".join(modified_text_chunks)
    total_tokens = len(model_interface.tokenizer.encode(final_text))
    text_token_count = total_tokens - image_token_count
    return {"text_tokens": text_token_count, "image_tokens": image_token_count, "total_tokens":total_tokens}

def load_json_file(filepath: str) -> Dict:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def extract_answer_term(text: str, term: str) -> str:
    """
    Extracts the value for a given term from the text.
    It first tries to match a quoted value, then an unquoted word.
    """
    patterns = {
        'Answer': r'"Answer":\s*(?:"([^"]+)"|([\w-]+))',
        'Confidence': r'"Confidence":\s*(?:"([^"]+)"|([\w.]+))',
        'Choice': r'"Choice":\s*(?:"([^"]+)"|([\w-]+))',
        'Caption': r'"Caption":\s*(?:"([^"]+)"|([\w-]+))',
        'A': r'"A":\s*(?:"([^"]+)"|([\w-]+))',
        'B': r'"B":\s*(?:"([^"]+)"|([\w-]+))',
        'C': r'"C":\s*(?:"([^"]+)"|([\w-]+))',
        'D': r'"D":\s*(?:"([^"]+)"|([\w-]+))',
        'E': r'"E":\s*(?:"([^"]+)"|([\w-]+))',
        'F': r'"F":\s*(?:"([^"]+)"|([\w-]+))',
        # 'Caption': r'"Caption":\s*(?:"([^"]+)"|([\w-]+))'
    }
    pattern = patterns.get(term)
    if not pattern:
        return None
    match = re.search(pattern, text)
    if match:
        return (match.group(1) or match.group(2)).strip()
    else:
        # Fallback if regex doesn't match.
        parts = text.split(term)
        if parts:
            return re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
        return None
    
def get_db_path(args) -> str:
    # if args.num_train > 1:
    #     return f"example_database/{args.data_name}_seed_{args.seed}_num_train_{args.num_train}"
    return f"example_database/{args.data_name}_seed_{args.seed}"