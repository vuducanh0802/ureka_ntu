Using huggingface to implement BERT variations


- Dataset: train.tsv (Stormfront) (https://github.com/Vicomtech/hate-speech-dataset --> all_files)
  - distilbert-base-uncased-finetuned-sst-2-english:
    - F1: 0.23 (binary), 0.40 (macro)
    - Accuracy: 0.44
  - cardiffnlp/twitter-roberta-base-sentiment:
    - F1: 0.372 (binary), 0.596(macro)
    - Accuracy: 0.72
  - bert-base-uncased:
    - F1: 0.12 (binary), 0.33 (macro)
    - Accuracy: 0.40


- Dataset: 27k gabhatecorpus ( https://osf.io/edua3/ --> 27k lines )
  - distilbert-base-uncased-finetuned-sst-2-english:
    - F1: 0.20 (binary), 0.33 (macro)
    - Accuracy: 0.36
  - cardiffnlp/twitter-roberta-base-sentiment:
    - F1: 0.17 (binary), 0.55(macro)
    - Accuracy: 0.88



