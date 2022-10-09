import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # "audio": audio_wave,
    # "spectrogram": audio_spec,
    # "duration": audio_wave.shape(1) / self.config_parser["preprocessing"]["sr"],
    # "text": data_dict["text"],
    # "text_encoded": self.text_encoder.encode(data_dict["text"]),
    # "audio_path": audio_path,
    # print(00000)
    # print(dataset_items)
    lengths_spec = []
    lengths_text = []
    texts = []
    audio_paths = []
    audio = []
    dim_1_spec = dataset_items[0]['spectrogram'].shape[1]
    for data in dataset_items:
        lengths_spec.append(data['spectrogram'].shape[2])
        lengths_text.append(data['text_encoded'].shape[1])
        texts.append(data['text'])
        audio_paths.append(data['audio_path'])
        audio.append(data['audio'])
    batch_spec = torch.zeros(len(dataset_items), dim_1_spec, max(lengths_spec))
    batch_texts = torch.zeros(len(dataset_items), max(lengths_text))
    for i, data in enumerate(dataset_items):
        batch_spec[i, :, :lengths_spec[i]] = data['spectrogram']
        batch_texts[i, :lengths_text[i]] = data['text_encoded']

    lengths_text = torch.tensor(lengths_text).long()
    lengths_spec = torch.tensor(lengths_spec).long()
    result_batch = {"spectrogram": batch_spec,
                    "text_encoded": batch_texts,
                    "text_encoded_length": lengths_text,
                    "text": texts,
                    'spectrogram_length':  lengths_spec,
                    'audio_path': audio_paths,
                    'audio': audio
                    }
    return result_batch
