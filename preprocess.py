import pickle
import  string
import re
import unicodedata
from collections import Counter
import os


class LoadData(object):
    def __init__(self, filename, filename_2=None, pair=False):
        self.filename = filename
        if filename_2:
            self.filename_2 = filename_2
        self.pair = pair
        if pair:
            self.text = None
        else:
            self.source_text = None
            self.target_text = None
        # store the source sentences and the target sentences
        self.source_sentences = []
        self.target_sentences = []
        # counters for  source sentences and target sentences
        self._src_counter = Counter()
        self._target_counter = Counter()
        self._special_tokens = ['<PAD>', '<GO>', '<EOS>']

    # load the data
    def load_text(self, encode='utf-8'):
        f = open(self.filename, 'r', encoding=encode)
        self.text = f.read()
        if self.pair:  # if source and target are within the same file
            f.close()
            return self.text
        self.source_text = self.text
        f_2 = open(self.filename_2, 'r')
        self.target_text = f_2.read()
        f_2.close()
        return self.source_text, self.target_text

    # split the data
    def _split_data(self):
        if self.pair:
            lines = self.text.split('\n')
            pair_sentences = [line.split('\t') for line in lines if line]
            self.source_sentences = [line[0] for line in pair_sentences]
            self.target_sentences = [line[1] for line in pair_sentences]
            return self.source_sentences, self.target_sentences
        else:
            source_lines = self.source_text.split('\n')
            self.source_sentences = [line for line in source_lines]
            target_lines = self.target_text.split('\n')
            self.target_sentences = [line for line in target_lines]
            return self.source_sentences, self.target_sentences

    # clean the data
    def clean_data(self):
        source_data, target_data = self._split_data()
        # surround  punctuations by whitespace
        punc_suround = re.compile('[?.!,¿]')
        source_data = [re.sub(r"([?.!,¿])", r" \1 ", phrase) for phrase in source_data]
        target_data = [re.sub(r"([?.!,¿])", r" \1 ", phrase) for phrase in target_data]
        source_data = [re.sub(r'[" "]+', " ", phrase) for phrase in source_data]
        target_data = [re.sub(r'[" "]+', ' ', phrase) for phrase in target_data]

        # transform all letters to latin ascii
        source_data = [unicodedata.normalize('NFD', phrase).encode('ascii', 'ignore').decode('UTF-8') for phrase in
                       source_data]
        target_data = [unicodedata.normalize('NFD', phrase).encode('ascii', 'ignore').decode('UTF-8') for phrase in
                       target_data]
        #         source_data = [phrase.decode('UTF-8') for phrase in source_data]
        #         target_data = [phrase.decode('UTF-8') for phrase in target_data]
        #  convert all sentences to lowercase
        source_data = [phrase.lower() for phrase in source_data]
        target_data = [phrase.lower() for phrase in target_data]
        # delete everything except a-z ,.!?
        source_data = [re.sub('[^a-z?,!.¿]', ' ', phrase) for phrase in source_data]
        target_data = [re.sub('[^a-z?,!.¿]', ' ', phrase) for phrase in target_data]
        # add special end token for target sentences #TODO ADDED NEW !!
        end_token = '{} '.format(self._special_tokens[-1])
        target_data = [phrase + end_token for phrase in target_data]

        return source_data, target_data

    def save_txt(self, list_phrases, save_path):

        split_path = save_path.split("/")
        if os.path.exists("/".join(split_path[:- 1])) is False and len(split_path) >= 2:
            os.makedirs("/".join(split_path[:- 1]))
        with open(save_path, 'w') as saved_file:
            for phrase in list_phrases:
                print(phrase, file=saved_file)
        print('all sentences are saved !')

    def save_binary(self, list_phrases_src, list_phrases_target, save_path):

        split_path = save_path.split("/")
        print(len(split_path))
        if os.path.exists("/".join(split_path[:- 1])) is False and len(split_path) >= 2:
            os.makedirs("/".join(split_path[:- 1]))
        with open(save_path, 'wb') as file_:
            pickle.dump([list_phrases_src, list_phrases_target], file_)

        print('file saved')

    def load_binary(self, file_path):
        with open(file_path, 'rb') as read_file:
            get_data = pickle.load(read_file)
        return get_data

    # count the unique tokens
    def count_tokens(self, sentences_list, src=True):
        for sentence in sentences_list:
            if src:
                self._src_counter.update(sentence.split())
            else:
                words = sentence.split()
                # print(words)
                # print('-' * 20)
                words.pop()
                # print(words)
                # print(' ')
                # input()
                self._target_counter.update(words)  # TODO Except <EOS> token !!!
        print('tokens are counted !')

    # save tokens in a file
    def save_tokens(self, file_name, common_words=None, src_counter=True):
        if src_counter:
            counter = self._src_counter
            special_tokens = [self._special_tokens[0]]
        else:
            counter = self._target_counter
            special_tokens = self._special_tokens
        split_path = file_name.split("/")
        if os.path.exists("/".join(split_path[:- 1])) is False and len(split_path) >= 2:
            os.makedirs("/".join(split_path[:- 1]))
        with open(file_name, 'w') as file_:
            for special in special_tokens:
                print(special, file=file_)

            if not common_words:
                for word in counter.keys():
                    print(word, file=file_)
                print('all {} tokens are saved'.format(len(counter) + len(special_tokens)))

            else:
                for word, _ in counter.most_common(common_words):
                    print(word, file=file_)
                print('the top {} tokens are saved'.format(common_words + len(special_tokens)))


path = 'fin.txt'
load_data = LoadData(path, pair=True)
# load text
load_data.load_text()  # TODO always load text !!
eng_data, finnish_sentences = load_data.clean_data()
# save binary
binary_path = 'words/eng_finn.p'
load_data.save_binary(eng_data, finnish_sentences, binary_path)
# save in text file
load_data.save_txt(eng_data, 'words/eng_sentences.txt')
load_data.save_txt(finnish_sentences, 'words/finn_sentences.txt')
# count source tokens and target tokens
load_data.count_tokens(eng_data)  # source tokens
load_data.count_tokens(finnish_sentences, src=False)  # target tokens
# save words
load_data.save_tokens('words/english_words.txt')  # source words
load_data.save_tokens('words/finnish_words.txt', src_counter=False)  # target words

#print(load_data._target_counter)
#print(len(load_data._target_counter))