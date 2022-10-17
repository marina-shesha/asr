# ASR project 

## Installation guide

Clone repository 
```shell
git clone https://github.com/marina-shesha/asr.git
```
Download requirements
```shell
pip install -r asr/requirements.txt
```

## Training
Run train.py with train_config.json

```shell
%run -i asr/train.py --config asr/hw_asr/configs/train_config.json
```
## Test 
Run test.py file to evaluate metrics and get preductions
```shell
%run -i asr/test.py \
--resume /content/asr/models/model_best.pth\
--config /content/asr/hw_asr/configs/test_config.json\
--test-data-folder asr/test_data \
-b 5\
-o test_data_out.json
```
