import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
from autocorrect import Speller
from textblob import TextBlob
from gramformer import Gramformer
import language_tool_python


def process_string(s):
    # Remove spaces at the beginning of the string
    s = s.lstrip()
    # Pattern to match 1 or 2 spaces between letters
    s = re.sub(r'(?<=\S) {1,2}(?=\S)', '', s)
    # Pattern to match more than two spaces between letters and replace with a single space
    s = re.sub(r'(?<=\S) {3,}(?=\S)', ' ', s)
    return s

def replace_words(text, replacements):
    # Replace words based on the dictionary provided using regex for flexibility with spaces
    for key, value in replacements.items():
        # Create a regex pattern that allows for any number and kind of whitespace
        pattern = r'\b' + r'\s*'.join(key) + r'\b'  # \b is a word boundary
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    return text
def remove_repeated_letters(text):
    # Use regex to match and replace occurrences where a letter is repeated more than twice consecutively
    # It matches and captures a letter and checks if it is followed by itself more than once
    # The replacement pattern retains only two instances of the matched letter
    return re.sub(r'(.)\1+', r'\1', text)

def grammatical_correction(text, gramformer_instance):
    try:
        corrections = gramformer_instance.correct(text, max_candidates=1)
        # Typically, corrections would be a list of tuples where each tuple is (corrected_text, score)
        if corrections and isinstance(corrections, list) and len(corrections[0]) > 0:
            corrected_text = corrections[0][0]  # Access the first correction
            return corrected_text
        else:
            return text
    except Exception as e:
        print(f"Error during grammatical correction: {e}")
        return text


def spell_correct(text):
    speller = Speller()
    corrected_text = speller(text)
    return corrected_text


def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())


def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer






class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()

# Define all the classes here (CNNLayerNorm, ResidualCNN, etc.)
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, utterance) in data:  # Adjusted to match the data structure
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def GreedyDecoder(output, labels=None, label_lengths=None, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []

    # Only process labels if they are provided and not empty
    if labels is not None and len(labels) > 0 and label_lengths is not None:
        for i, label in enumerate(labels):
            target = text_transform.int_to_text(label[:label_lengths[i]].tolist())
            targets.append(target)

    for args in arg_maxes:
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))

    return decodes, targets

def custom_dataloader(csv_file, audio_dir, batch_size, transform, data_type="train"):
    df = pd.read_csv(csv_file)
    dataset = []
    for index, row in df.iterrows():
        waveform, _ = torchaudio.load(os.path.join(audio_dir, row['filename']))
        dataset.append((waveform, row['text']))

    def collate_fn(batch):
        return data_processing(batch, data_type)

    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val








#end of copied classes and defs










def load_model(model_path, device):
    model = SpeechRecognitionModel(
        n_cnn_layers=3,
        n_rnn_layers=5,
        rnn_dim=512,
        n_class=29,
        n_feats=64,
        stride=2,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move the model to the specified device
    model.eval()
    return model

def predict(audio_path, model, device):
    waveform, sample_rate = torchaudio.load(audio_path)
    text_transform = TextTransform()

    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64
    )
    spectrogram = spec_transform(waveform).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(spectrogram)
        output = torch.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        decoded_preds, _ = GreedyDecoder(output, blank_label=28, collapse_repeated=True)

    return decoded_preds




def main():
    # Initialize Gramformer
    gf = Gramformer(models=1, use_gpu=torch.cuda.is_available())

    audio_path = "/home/team1/ble-uart-peripheral/audio_files/recent_16k.mp3"
    model_path = "/home/team1/STT_code/speech_recognition_model.pth"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #print(f"Using device: {device}")

    model = load_model(model_path, device)
    raw_transcription = predict(audio_path, model, device)

    # Join the words to form a single text string and convert to lower case
    transcription_text = ''.join(raw_transcription).lower()

    # Remove excessive repeats from the final corrected text
    corrected_text = remove_repeated_letters(transcription_text)

    replacements_plural = {
        " degres": " degrees",
        " eighte ": " eighteen ",
        " ineten ": " nineteen ",
        " fiv ": " five ",
        " ninetythre ": " ninety three ",
        " fet": " feet ",
        "retr ce": "retrace",
        "sto": "stop"
    }


    #catch degree non-plural issue caused by gramformer
    plural_fix_text = replace_words(corrected_text, replacements_plural)


    # Apply grammatical correction using Gramformer
    grammatically_corrected_text = grammatical_correction(plural_fix_text, gf)

    # Apply spelling correction using autocorrect
    spell_corrected_text = spell_correct(grammatically_corrected_text)

    # Replace failed words with their corrections as the final step (fixes: double hallucinations)
    replacements = {
        "twenty twenty": "twenty",
        "thirty thirty": "thirty",
        "forty forty": "forty",
        "fifty fifty": "fifty",
        "sixty sixty": "sixty",
        "seventy seventy": "seventy",
        "eighty eighty": "eighty",
        "ninety ninety": "ninety",
        "traces": "retrace",
        "trace": "retrace",
        "retract": "retrace",
        " te ": " two ",
        " the ": " three ",
        " on ": " one ",
        " thre ": " three "
    }
    final_corrected_text = replace_words(spell_corrected_text, replacements)

    #print("Raw Transcription:", transcription_text)
    #print("Filtered Transcription:", final_corrected_text)

    print(final_corrected_text)

if __name__ == "__main__":
    main()
