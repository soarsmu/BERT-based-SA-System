# BERT-based-SA-System

We highly recommond users to explore project with docker. You can configure the environment with the following command.

```
docker pull pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
```

Then you need to download the BERT model fine-tuned on IMDb movie review dataset. We can download the model data via [Google Drive](www.google.com). After unziping the file, you can get a folder called `uncased_L-12_H-768_A-12`. You can `mkdir models` in your root folder, and put `uncased_L-12_H-768_A-12` under that newly created `models` folder.

You can use the following code to use the SA system.

```
from sentiment_analysis import SentimentAnalysis

model_checkpoint='./models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
bert_config_file='./models/uncased_L-12_H-768_A-12/bert_config.json'
vocab_file='./models/uncased_L-12_H-768_A-12/vocab.txt'

### initialize an SA system
sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                            bert_config_file=bert_config_file,
                            vocab_file=vocab_file)

text = "I am Jack and I'm very happy."

sentiment = sa_system.predict(text)
```

The method `sa_system.predict(text: str) -> bool` takes a piece of text as input and return a boolean value indicating the predicted sentiment (1 for positive and 0 for negative).



