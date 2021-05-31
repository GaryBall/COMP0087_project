import torch
import torch.nn as nn
from transformers import BartModel, BartTokenizer, BartConfig, BartForCausalLM

from transformers import BartForConditionalGeneration
from torch.nn import GRU
from torch.nn.utils.rnn import pad_sequence


# bart1+NLP. BART for causalLM is used
# and a 2-layer MLP was used to capture the coherence feature
class bart_MLP(nn.Module):
    def __init__(self, tokenizer, token_size=1024):
        super(bartForBinaryClassifier, self).__init__()

        # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.bart = BartForCausalLM.from_pretrained("facebook/bart-base")
        self.bart.resize_token_embeddings(len(tokenizer))
        # self.lm_head = nn.Linear(self.bart.config.d_model, self.bart.config.vocab_size, bias=False)

        # add an module for consistency checking
        self.consist_head = nn.Linear(self.bart.config.d_model, 1, bias=False)
        self.consist_logit = nn.Linear(token_size-1, 1)
        self.consist_activation = nn.Sigmoid()
        # add an module for checking the reasonability of ingredients unit
        # self.units_logit = nn.Linear(token_size-1, 1)
        # self.units_activation = nn.Sigmoid()
        self.cuda()

    def forward(self, batch, ins_start_id, ing_start_id = None, labels=None):
        print(batch.shape)
        outputs = self.bart(batch,output_hidden_states=True)
        last_hidden = outputs["hidden_states"][-1]
        print(last_hidden.shape)
        consist_head = self.consist_head(last_hidden)
        consist_head = consist_head.squeeze(dim=2)
        ins_subvector = torch.empty(0, consist_head.size(1)).to(args.device)
        # ing_subvector = torch.empty(0, consist_head.size(1)).to(args.device)
        print("consist_head:",consist_head.shape)
        for i in range(len(consist_head)):
            sample = consist_head[i]

            # the start index of ingredients
            # ing_index = (batch[i] == ing_start_id).nonzero()[0]
            # the start index of instruction
            ins_index = (batch[i] == ins_start_id).nonzero()[0]
            # move the non-zero elements of ingredients vector to left
            # ing_z = torch.zeros(consist_head.size(1)).to(args.device)
            # ing_z[0:(ins_index-ing_index)] = sample[ing_index:ins_index]
            # ing_subvector = torch.cat((ing_subvector, ing_z.unsqueeze(0)))

            sample[0:ins_index] = 0
            nz = sample.nonzero().squeeze()
            # move the non-zero elements of instructions vector to left
            ins_z = torch.zeros(sample.numel() - nz.numel()).to(args.device)
            ins_z = torch.cat((sample[nz], ins_z)).unsqueeze(0)
            ins_subvector = torch.cat((ins_subvector, ins_z))
            # print(ins_subvector.shape)

        print(ins_subvector.shape)
        consist_output = self.consist_logit(ins_subvector.half().to(args.device))
        consist_output = self.consist_activation(consist_output)
        print(consist_output.shape)

        # units_output = self.units_logit(ing_subvector.half().to(args.device))
        # units_output = self.units_activation(units_output)
        # print(units_output.shape)

        # the usual prediction output
        logits = outputs["logits"]
        # output_token = logits.argmax(dim=2)



        # output = self.out(output_token.float())

        # we apply dropout to the sequence output, tensor has shape (batch_size, sequence_length, 768)
        # sequence_output = self.dropout(sequence_output)

        # next, we apply the linear layer. The linear layer (which applies a linear transformation)
        # takes as input the hidden states of all tokens (so seq_len times a vector of size 768, each corresponding to
        # a single token in the input sequence) and outputs 2 numbers (scores, or logits) for every token
        # so the logits are of shape (batch_size, sequence_length, 2)
        # logits = self.dropout(sequence_output)

        return logits, consist_output



# BART1 with GRU, casual languguage model setting
class bartWithGRU(nn.Module):

    def __init__(self, tokenizer, token_size=512):
        super(bartWithGRU, self).__init__()

        # initialize a BART for CLM (without encoder)
        self.bart = BartForCausalLM.from_pretrained("facebook/bart-base")
        self.bart.resize_token_embeddings(len(tokenizer))

        # add an module for consistency checking
        print(self.bart.config.d_model)
        # self.emb_size = 64
        # self.sentence_emb = nn.Linear(self.bart.config.d_model, self.emb_size)
        # GRU is used to process the coherence feature
        self.gru = nn.GRU(self.bart.config.d_model, 32, 1, batch_first=False)
        self.gru_head = nn.Linear(32, 1)
        self.linear_activation = nn.Sigmoid()
        self.cuda()

    def forward(self, batch, inst_pos, labels=None):
        # inst_pos: the position of token <NEXT_INSTR>, as a index list.
        outputs = self.bart(batch, output_hidden_states=True)
        last_hidden = outputs["hidden_states"][-1]
        tokens_list = []

        # get the hidden state of token <NEXT_INSTR>
        for i in range(batch.size()[0]):
            batch_hidden = last_hidden[inst_pos[i][:, 0], inst_pos[i][:, 1], :]
            tokens_list.append(batch_hidden)

        # match the batch size for GRU input
        gru_input = pad_sequence(tokens_list)
        # GRU input
        gru_output, _ = self.gru(gru_input)

        # one-layer feed-forward neural network for binary classification
        consist_output = self.gru_head(gru_output[-1])
        consist_output = self.linear_activation(consist_output)

        # the next-token prediction output
        logits = outputs["logits"]

        return logits, consist_output



# bart2 + GRU. BART for conditional generation as backbone model
# and a GRU is used for CDM
class bartWithGRU_CG(nn.Module):
    def __init__(self, tokenizer, token_size=512):
        super(bartWithGRU, self).__init__()

        # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.bart.resize_token_embeddings(len(tokenizer))
        # self.lm_head = nn.Linear(self.bart.config.d_model, self.bart.config.vocab_size, bias=False)

        # add an module for consistency checking
        self.gru = torch.nn.GRU(self.bart.config.d_model, 32, 1, batch_first=False)
        self.gru_head = nn.Linear(32, 1)
        self.linear_activation = nn.Sigmoid()
        self.cuda()

    def forward(self, encoder_batch, batch, inst_pos):
        # inst_pos: the position of token <NEXT_INSTR>, as a index list.
        # print(batch.shape)

        outputs = self.bart(input_ids=encoder_batch,
                            decoder_input_ids=batch,
                            output_hidden_states=True)
        last_hidden = outputs["decoder_hidden_states"][-1]
        # gru_input = torch.zeros(batch.size()[0],max_length,self.bart.config.d_model)
        tokens_list = []
        for i in range(batch.size()[0]):
            batch_hidden = last_hidden[inst_pos[i][:, 0], inst_pos[i][:, 1], :]
            tokens_list.append(batch_hidden)
        gru_input = pad_sequence(tokens_list)
        gru_output, _ = self.gru(gru_input)

        # print(ins_subvector.shape)
        # consist_output = self.consist_logit(ins_subvector.half().to(args.device))
        consist_output = self.gru_head(gru_output[-1])
        # print(consist_output.shape)
        consist_output = self.linear_activation(consist_output)

        logits = outputs["logits"]

        return logits, consist_output