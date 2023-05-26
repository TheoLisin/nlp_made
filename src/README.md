# Lab assignment #2

Изначально я думал сделать baseline на базе torch, после улучшить его. Потому написал всю эту инфраструктуру для обучения с использованием pytorch-lightning.
Однако после обучения бейзлайн модели и нескольких экспериментов понял, что лучше просто выбрать seq2seq модель с transformers.

[Лучшая обученная модель](https://github.com/TheoLisin/nlp_made/blob/23s_made/src/bertmodel.ipynb) это Bert2Bert. В качестве английской Bert была взята стандартная uncased модель "bert-base-uncased", для перевода на русский - "ai-forever/ruBert-base". Получилось достичь результата blue 34 (nltk).
