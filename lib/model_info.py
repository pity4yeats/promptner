from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification

plms = ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased', 'dslim/bert-large-NER',
        'roberta-base', 'roberta-large',
        'facebook/bart-base', 'facebook/bart-large',
        'microsoft/deberta-base', 'microsoft/deberta-large', 'microsoft/deberta-xlarge']


def print_model_summary(models):
    """
    bert-base-cased 108310272
    bert-base-uncased 109482240
    bert-large-cased 333579264
    bert-large-uncased 335141888
    dslim/bert-large-NER 333579264
    roberta-base 124645632
    roberta-large 355359744
    facebook/bart-base 139420416
    facebook/bart-large 406291456
    microsoft/deberta-base 138601728
    microsoft/deberta-large 405163008
    microsoft/deberta-xlarge 757804032
    """
    plms_summary = {}
    for plm in models:
        model = AutoModel.from_pretrained(plm)
        plms_summary[plm] = model.num_parameters()

    for k, v in plms_summary.items():
        print(k, v)


# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

print_model_summary(plms)


# CoNLL2003
def build_conll03_train():
    return


def build_conll03_devel():
    return


def down_sample_conll03():
    return


def build_conll04_devel():
    return


def down_sample_conll04():
    return


# CoNLL2005
def build_conll05_train():
    return


def build_conll05_devel():
    return


def down_sample_conll05():
    return


# CoNLL2012
def build_conll12_train():
    return


def build_conll12_devel():
    return


def down_sample_conll12():
    return

