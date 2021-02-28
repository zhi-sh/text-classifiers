# -*- coding: utf-8 -*-
# @DateTime :2021/2/28 下午1:36
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from typing import Dict, List, Optional, Tuple, Union
import transformers
from torch import nn
from text_classifiers.models import AbstractModel
from text_classifiers.tools import tools


class Transformer(nn.Module, AbstractModel):
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, model_args: Dict = {}, cache_dir: Optional[str] = None, tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        config = transformers.AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.auto_model = transformers.AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, *tokenizer_args)

    def forward(self, features):
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token of last layers
        features.update({'text_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # 有些摩西只返回last_hidden_states和all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def tokenize(self, texts: Union[List[str], List[Tuple[str, str]]]):
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_seq_length))
        return output

    @property
    def dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        saved_config_path = os.path.join(output_path, 'config.json')
        tools.json_save(self.get_config_dict(), saved_config_path)

    @staticmethod
    def load(input_path: str):
        saved_config_path = os.path.join(input_path, 'config.json')
        config = tools.json_load(saved_config_path)
        return Transformer(model_name_or_path=input_path, **config)
