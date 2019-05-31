import collections

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.contrib import text
from sklearn import preprocessing
from tqdm import tqdm


def load_embed(word_embedding_path):
    
    with open(word_embedding_path) as f:
        lines = [line.strip().split(' ')[0] for line in tqdm(f, desc='load embedding')]
        word_counter = collections.Counter({line: 1 for line in lines[1:]})
        
    vocab = text.vocab.Vocabulary(word_counter, reserved_tokens=['<pad>'])
    embed = text.embedding.CustomEmbedding(word_embedding_path, vocabulary=vocab)
    
    return embed


def load_label(label_path):

    label = []

    with open(label_path) as f:
        for line in tqdm(f, desc='load label'):
            line = line.strip().split('\t')
            label.append(line[0])

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(label)

    label_num = len(label)
    
    return label_encoder, label_num
    
    
def load_label_embed(label_path, embed, label_encoder, label_num, ctx=mx.cpu()):

    label_embed = np.zeros(shape=(label_num, embed.vec_len))

    with open(label_path) as f:
        for line in tqdm(f, desc='embedding label'):
            line = line.strip().split('\t')
            
            idx = label_encoder.transform([line[0]])
            _embed = embed.get_vecs_by_tokens(line[3].split(',')).mean(axis=0)
            label_embed[idx] = _embed.asnumpy()
            
    label_embed = nd.array(label_embed, ctx=ctx)

    return label_embed


def load_X(X_path, embed, X_num, feature_num):

    feature_pad = [embed.to_indices('<pad>')]
    qid_X, X = [], []

    with open(X_path) as f:
        count = 0
        for line in tqdm(f, desc='load X'):
            line = line.strip().split('\t')
            
            if(len(line) > 2):
                qid_X.append(line[0])
                feature = embed.to_indices(line[2].split(',')[:feature_num])
                feature.extend(feature_pad*(feature_num - len(feature)))
                X.append(feature)
            
            count += 1
            if(count == X_num):
                break

    X_num = len(X)

    return X, qid_X, X_num


def load_y(y_path, label_encoder, label_num, X_num, qid_X):

    y = np.zeros(shape=(X_num, label_num), dtype='float32')

    with open(y_path) as f:
        idx = 0
        for line in tqdm(f, desc='load y'):
            line = line.strip().split('\t')
            qid_y = line[0]
            
            if(qid_y == qid_X[idx]):
                _label = label_encoder.transform(line[1].split(','))
                for l in _label:
                    y[idx, l] = 1
                idx += 1

            if(idx >= X_num):
                break
    
    return y


def load_data(X_num=-1, feature_num=50, test_num=int(2e5), batch_size=64, 
    data_base_path='./', if_label_embed=False, if_log=True, ctx=mx.cpu()):
    
    X_path = data_base_path + 'question_train_set.txt'
    y_path = data_base_path + 'question_topic_train_set.txt'
    label_path = data_base_path + 'topic_info.txt'
    word_embedding_path = data_base_path + 'word_embedding.txt'
    
    embed = load_embed(word_embedding_path)
    label_encoder, label_num = load_label(label_path)

    if(if_label_embed):
        label_embed = load_label_embed(label_path, embed, label_encoder, label_num, ctx=ctx)

    X, qid_X, X_num = load_X(X_path, embed, X_num, feature_num)
    y = load_y(y_path, label_encoder, label_num, X_num, qid_X)

    X_train, X_test = X[:-test_num], X[-test_num:]
    y_train, y_test = y[:-test_num], y[-test_num:]

    train_dataset = gluon.data.dataset.ArrayDataset(X_train, y_train)
    train_data_loader = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = gluon.data.dataset.ArrayDataset(X_test, y_test)
    test_data_loader = gluon.data.DataLoader(test_dataset, batch_size=batch_size)

    if(if_log):
        print('\n' + '='*50)
        for X_train_batch, y_train_batch in train_data_loader:
            print('X_train shape', X_train_batch.shape, 'y_train shape', y_train_batch.shape)
            break
        print('train_batch_num', len(train_data_loader))
        for X_test_batch, y_test_batch in test_data_loader:
            print('X_test shape', X_test_batch.shape, 'y_test shape', y_test_batch.shape)
            break
        print('test_batch_num', len(test_data_loader))
        print('train/test split', len(X_train)/len(X))
    
    if(if_label_embed):
        return train_data_loader, test_data_loader, embed, label_embed
    else:
        return train_data_loader, test_data_loader, embed, label_num
