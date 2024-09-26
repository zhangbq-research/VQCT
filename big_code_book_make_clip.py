import os
import nltk
from nltk.corpus import wordnet
import webcolors
from glove import GloVe
import pickle
from nltk.stem import WordNetLemmatizer
import clip
import torch
import json

colors = webcolors.CSS3_NAMES_TO_HEX

def read_txt(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    return data

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data['annotations']

if __name__ == '__main__':
    model, _ = clip.load("ViT-B/32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # read cub
    path = './text'
    class_names = os.listdir(path)

    data = []
    for class_name in class_names:
        for file_name in os.listdir(os.path.join(path, class_name)):
            lines = read_txt(os.path.join(path, class_name, file_name))
            data.extend(lines)

    # read flowers
    path = './flower/text_c10'
    class_names = os.listdir(path)

    for class_name in class_names:
        if '.' in class_name:
            continue
        for file_name in os.listdir(os.path.join(path, class_name)):
            if '.txt' in file_name:
                lines = read_txt(os.path.join(path, class_name, file_name))
                data.extend(lines)

    # read coco
    path = './coco'
    class_names = os.listdir(path)

    for class_name in class_names:
        dict_file = read_json(os.path.join(path, class_name))
        for lines in dict_file:
            data.append(lines['caption'])

    adj_code_book = []
    noun_code_book = []
    adj_noun_edges = []

    adj_code_book_num = {}
    noun_code_book_num = {}

    wnl = WordNetLemmatizer()

    for line in data:
        tokens = nltk.word_tokenize(line)
        tokens = [token.lower() for token in tokens]
        words_type = nltk.pos_tag(tokens)

        for i, word in enumerate(words_type):
            if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS':
                nun = wnl.lemmatize(word[0], wordnet.NOUN)
                if nun not in noun_code_book:
                    noun_code_book.append(nun)
                    noun_code_book_num[nun] = 1
                else:
                    noun_code_book_num[nun] = noun_code_book_num[nun] + 1
            elif word[1] == 'JJ':
                # if word[0] in colors:
                if word[0] not in adj_code_book:
                    adj_code_book.append(word[0])
                    adj_code_book_num[word[0]] = 1
                else:
                    adj_code_book_num[word[0]] = adj_code_book_num[word[0]] + 1
                # else:
                #     if word[0] not in adj_code_book.keys():
                #         adj_code_book[word[0]] = len(adj_code_book)
            if i < len(words_type)-1:
                if words_type[i][1] == 'JJ' and (words_type[i+1][1] == 'NN' or words_type[i+1][1] == 'NNS' or words_type[i+1][1] == 'NNP' or words_type[i+1][1] == 'NNPS'):
                    adj_noun_edges.append((words_type[i][0], wnl.lemmatize(words_type[i+1][0], wordnet.NOUN)))

    print(len(adj_code_book))
    print(len(noun_code_book))


    noun_num = 0
    for k, v in adj_code_book_num.items():
        if v > 10:
            noun_num = noun_num +1
    print(noun_num)

    noun_num = 0
    for k, v in noun_code_book_num.items():
        if v > 10:
            noun_num = noun_num +1
    print(noun_num)

    adj_code_book = [bi for bi in adj_code_book if adj_code_book_num[bi]>10]
    noun_code_book = [bi for bi in noun_code_book if noun_code_book_num[bi] > 10]
    print(len(adj_code_book))
    print(len(noun_code_book))

    # print(len(np.adj_code_book))
    # print(len(noun_code_book))

    adj_code_book_vec = {}
    noun_code_book_vec = {}
    model, _ = clip.load("ViT-B/32", device=device)

    for wnid in adj_code_book:
        # print(getnode(wnid).lemma_names())
        text = clip.tokenize(["a photo of {} object".format(wnid), ]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        adj_code_book_vec[wnid] = text_features[0].cpu()

    for wnid in noun_code_book:
        # print(getnode(wnid).lemma_names())
        text = clip.tokenize(["a photo of {}".format(wnid), ]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        noun_code_book_vec[wnid] = text_features[0].cpu()


    obj = {'adj_code_book': adj_code_book,
           'noun_code_book': noun_code_book,
           'noun_code_book_vec': noun_code_book_vec,
           'adj_code_book_vec': adj_code_book_vec,
           'adj_code_book_num': adj_code_book_num,
           'noun_code_book_num': noun_code_book_num,
           'adj_noun_edges': adj_noun_edges}

    with open('big_nlp_word_knowledge_clip.pkl', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


