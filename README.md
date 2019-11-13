# coqa-bert-baselines
BERT baselines for extractive question answering on coqa (https://stanfordnlp.github.io/coqa/). The original paper for the coqa dataset can be found [here](https://arxiv.org/abs/1808.07042). We provide the following models - 

- [x] [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- [ ] [RoBERTa](https://arxiv.org/abs/1907.11692)
- [x] [DistilBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation)
- [x] [SpanBERT](https://arxiv.org/abs/1907.10529)

Except `SpanBERT` all pretrained models are provided by [huggingface](https://github.com/huggingface/transformers). The `SpanBERT` model is provided by [facebookresearch](https://github.com/facebookresearch/SpanBERT). 

This repo builds upon the original code provided with the paper which can be found [here](https://github.com/stanfordnlp/coqa-baselines).

## Dataset

The dataset can be downloaded from [here](https://stanfordnlp.github.io/coqa/). The dataset needs to be preprocessed to obtain 2 files - `coqa.train.json` and `coqa.dev.json`. You can either follow the steps provided in the [original repo](https://github.com/stanfordnlp/coqa-baselines) for preprocessing or download the preprocessed files directly from [here](https://drive.google.com/drive/folders/1XxKDaJegoj_gNv6pkXnFvya9TzborzTQ?usp=sharing). 

## Requirements

`torch` : can be installed from [here](https://pytorch.org/). This code was tested with torch 0.3.0 and cuda 9.2.

`transformers`: can be installed from  [here](https://github.com/huggingface/transformers).

## Usage
To run the models use the following command - 

``` python main.py --arguments```

The ```arguments``` are as follows : 

| Arguments | Description |
| ----------|-------------|
| trainset | Path to the training file.|
| devset | Path to the dev file. |
| model_name | Name of the pretrained model to train (`BERT`,`RoBERTa`,`DistilBERT`,`SpanBERT`) |
| model_path| If the model has been downloaded already, you can specify the path here. If left none, the code will automatically download the pretrained models and run. |
| save_state_dir | The state of the program is regularly stored in this folder. This is useful incase training stops abruptly in the middle, it will automatically restart training from where it stopped |
| pretrained_dir | The path from which to restore the entire state of the program.  This path should be the name of the same folder which you would have specified in `save_state_dir`. |
| cuda | whether to train on gpu |
| debug | whether to print during training. |
| n_history | history size to use. For more info read the [paper](https://arxiv.org/abs/1808.07042). |
| batch_size | Batch size to be used for training and validation. |
| shuffle | Whether to shuffle the dataset before each epoch
| max_epochs | Number of epochs to train. |
| lr | Learning rate to use. |
| grad_clip | Maximum norm for gradients |
| verbose | Print updates every `verbose` epochs. |
| gradient_accumulation_steps | Number of update steps to accumulate before performing a backward/update pass. |
| adam_epsilon | Epsilon for Adam optimizer. |

For the given experiments the following values were used:

```
n_history = 2
batch_size = 4 (Couldn't fit a batch size larger than this on the GPU)
lr = 5e-5
verbose = 200
gradient_accumulation_steps = 12
```

The below experiments were conducted on [google colab](https://colab.research.google.com). For all the `BERT` models, the `base` versions were used (For eg: `bert-base-uncased`)


## Results
All the results are based on `n_history = 2`:

|Model Name| Dev F1 | Dev EM |
|----------|--------|--------|
| `SpanBERT` | 63.74 | 53.42 |
| `BERT` | 63.08 | 53.03 |
| `DistilBERT` | 61.5 | 52.35 |

## Contact

For any issues/questions, you can open a GitHub issue or contact [me](http://aniketdidolkar.in/) directly. 
