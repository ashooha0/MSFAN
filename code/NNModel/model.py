import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from attention_module import MultiHeadedAttention, SelfAttention

class MSFAN(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, args):
        super(MSFAN, self).__init__()
        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False
        
        self.dropout = torch.nn.Dropout(0.5)
        
        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], args.hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], args.hidden_dim, 2, padding=1)
        self.conv3 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], args.hidden_dim, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], args.hidden_dim, 4, padding=2)
        self.conv7 = torch.nn.Conv1d(4*args.hidden_dim, 3*args.hidden_dim, 5, padding=2)
        self.conv8 = torch.nn.Conv1d(3*args.hidden_dim, 2*args.hidden_dim, 3, padding=1)

        self.span = torch.nn.Linear(2*args.hidden_dim, args.span)
        self.unispan = torch.nn.Linear(2*args.hidden_dim, args.hidden_dim)
        self.bispan = torch.nn.Linear(2*args.hidden_dim, args.hidden_dim)

        self.bilstm = torch.nn.LSTM(2*args.hidden_dim, args.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attention_layer = SelfAttention(args)

        self.cls_linear = torch.nn.Linear(2*args.hidden_dim, args.class_num)

    def _get_embedding(self, sentence_tokens, mask):
        gen_embed = self.gen_embedding(sentence_tokens)
        domain_embed = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([gen_embed, domain_embed], dim=2)
        embedding = self.dropout(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def _local_feature(self, embedding):
        word_emd = embedding.transpose(1, 2)
        word1_emd = self.conv1(word_emd)[:, :, :self.args.max_sequence_len]
        word2_emd = self.conv2(word_emd)[:, :, :self.args.max_sequence_len]
        word3_emd = self.conv3(word_emd)[:, :, :self.args.max_sequence_len]
        word4_emd = self.conv4(word_emd)[:, :, :self.args.max_sequence_len]
        # word5_emd = self.conv5(word_emd)[:, :, :self.args.max_sequence_len]
        x_emb = torch.cat((word1_emd, word2_emd), dim=1)
        x_emb = torch.cat((x_emb, word3_emd), dim=1)
        x_emb = torch.cat((x_emb, word4_emd), dim=1)
        # x_emb = torch.cat((x_emb, word5_emd), dim=1)
        x_conv = self.dropout(torch.nn.functional.relu(x_emb))

        # x_conv = torch.nn.functional.relu(self.conv6(x_conv))
        # x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv7(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv8(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        # x_conv = x_conv[:, :lengths[0], :]
        # print(x_conv.size())
        return x_conv

    def forward(self, sentence_tokens, lengths, masks):
        embedding = self._get_embedding(sentence_tokens, masks)
        local_feature = self._local_feature(embedding)
        contextual_feature = self._lstm_feature(local_feature, lengths)
        # print(contextual_feature.size())

        unispan = contextual_feature
        unispan = self.unispan(unispan)
        temp = torch.cat((unispan[:, 1:, :], torch.zeros([unispan.size(0), 1, unispan.size(2)]).to(self.args.device)), dim=1)
        bispan = torch.cat((unispan, temp), dim=-1)
        bispan = self.bispan(bispan)

        span_class = self.span(local_feature[:, :lengths[0], :])
        span = torch.argmax(span_class, dim=-1)
        span = span.unsqueeze(-1).repeat(1, 1, unispan.size(-1))
        span_features = torch.where(span>0, bispan[:, :lengths[0], :], unispan[:, :lengths[0], :])
        # print(span_features.size())

        # contextual representation
        feature = span_features
        feature_attention = self.attention_layer(feature, feature, masks[:, :lengths[0]])
        feature = feature + feature_attention
        # print(feature.size())

        feature = feature.unsqueeze(2).expand([-1, -1, lengths[0], -1])
        feature_T = feature.transpose(1, 2)
        features = torch.cat([feature, feature_T], dim=3)
        #print(features.size())

        logits = self.cls_linear(features)
        return logits

