# NLP Non Deep
## Authors
- Luu Hoang Long Vo (luu-hoang-long.vo@epita.fr)
- Yassin Bouhassoun (yassin.bouhassoun@epita.fr)
- Phu Hien Le (phu-hien.le@epita.fr)
- Youssef Bouarfa Dinia (youssef.bouarfa-dinia@epita.fr)

## Architecture

```bash
.
├── README.md
├── lab_02
│   └── nlp02.ipynb
├── lab_03
│   ├── imdb_train.txt
│   ├── imdb_valid.txt
│   ├── incorrect_format_1_test.txt
│   ├── incorrect_format_2_test.txt
│   ├── lab03.docx
│   ├── lab03.pdf
│   ├── nlp03.ipynb
│   ├── test.txt
│   └── train.txt
├── lab_04
│   └── nlp04.ipynb
├── poetry.lock
├── pyproject.toml
└── theoretical_questions.pdf
```

## Lab 03

### Short description of the project

- The project is a continuation of we started on the second lab. We will try beating the results we obtained with naive Bayes using the FastText library.

### Description of the file/module architecture

- Training a FastText classifier takes a text file as input.
- For question 2 train.txt is used to train a FastText classifier with default parameters on the training data and test.txt is used for evaluation on the data using accuracy. 
- Then, for question 3 imdb_train.txt is used to train the model. imdb_valid.txt is used as a a validation file with the -autotune-validation argument.
- Finally for question 5 incorrect_format_1_test.txt and incorrect_format_2_test.txt are used to take 2 wrongly classified examples from the test set