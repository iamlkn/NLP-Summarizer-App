from datasets import load_dataset
import nltk
from nltk.corpus import stopwords


def create_txt_files(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            text = item['article'].replace('\n', ' ')
            summary = item['highlights'].replace('\n', ' ')
            f.write(f'{text} <sep> {summary}\n')


# split dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

create_txt_files(train_data, 'model/data/train.txt')
create_txt_files(val_data, 'model/data/val.txt')
create_txt_files(test_data, 'model/data/test.txt')

# stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('model/data/stop_words.txt', 'w', encoding='utf-8') as f:
    for word in stop_words:
        f.write(word + '\n')