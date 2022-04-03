from transformers import BertForPreTraining, AutoModel, AutoConfig
#
#
# def compute_hidden_state(input_at_the_moment, last_hidden_state):
#     processed_input = U * input_at_the_moment
#     weighted_last_hidden_state = W * last_hidden_state
#
#
# model = BertForPreTraining.from_pretrain('')

from transformers import BertTokenizer, RobertaTokenizer, RobertaForMaskedLM, BartTokenizer, BartForConditionalGeneration
sample = [
    """you're the last one on earth that I want to lose.""",
    """do you love sarah, yes?""",
    "United States"
    ]

# print(f'{sample=}')

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

input = tokenizer(sample, return_tensors='pt', padding=True, truncation=True)
input_ids = input['input_ids']
output = model(**input)
# print(output)

print(input_ids.shape)
print(input)
