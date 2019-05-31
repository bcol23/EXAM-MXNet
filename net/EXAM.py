import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, rnn


class EXAM(nn.Block):
    
    def __init__(self, feature_num, embedding, label_num, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.label_num = label_num
        
        self.embed = nn.Embedding(len(embedding), embedding.vec_len)
        self.gru = rnn.GRU(hidden_size=self.hidden_size, num_layers=2, layout='NTC')
        self.dense_1 = nn.Dense(units=60, flatten=False, activation='relu')
        self.dense_2 = nn.Dense(units=1, flatten=False)
        self.label_embed = self.params.get('label_embed', shape=(self.hidden_size, label_num))
        

    def forward(self, x):
        embed = self.embed(x)
        encode = self.gru(embed)
        
        interaction = nd.dot(encode, self.label_embed.data(ctx=encode.context))
        interaction = nd.transpose(interaction, axes=(0,2,1))
        _batch_size = interaction.shape[0]
        interaction = interaction.reshape((_batch_size*self.label_num, -1))
        
        out = self.dense_2(self.dense_1(interaction))
        out = out.reshape((_batch_size, self.label_num))
        return out


class EXAM_alter(nn.Block):
    
    def __init__(self, feature_num, embedding, label_embed, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.label_num = label_embed.shape[0]
        
        self.embed = nn.Embedding(len(embedding), embedding.vec_len)
        self.gru = rnn.GRU(hidden_size=self.hidden_size, num_layers=2, layout='NTC')
        self.label_embed_trans = nn.Dense(units=self.hidden_size)
        self.dense_1 = nn.Dense(units=60, flatten=False, activation='relu')
        self.dense_2 = nn.Dense(units=1, flatten=False)
        self.label_embed = self.params.get('label_embed', shape=(label_embed.shape))
    

    def forward(self, x):
        embed = self.embed(x)
        encode = self.gru(embed)
        
        _label_embed = self.label_embed_trans(self.label_embed.data(ctx=encode.context))
        interaction = nd.dot(encode, _label_embed.T)
        interaction = nd.transpose(interaction, axes=(0,2,1))
        _batch_size = interaction.shape[0]
        interaction = interaction.reshape((_batch_size*self.label_num, -1))
        
        out = self.dense_2(self.dense_1(interaction))
        out = out.reshape((_batch_size, self.label_num))
        return out
