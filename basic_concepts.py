from transformers import BertForPreTraining, AutoModel, AutoConfig
#
#
# def compute_hidden_state(input_at_the_moment, last_hidden_state):
#     processed_input = U * input_at_the_moment
#     weighted_last_hidden_state = W * last_hidden_state
#
#
# model = BertForPreTraining.from_pretrain('')

from transformers import BertTokenizer, RobertaTokenizer, RobertaForMaskedLM
# sample = """you're the last one on earth that I want to lose."""

# print(f'{sample=}')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

