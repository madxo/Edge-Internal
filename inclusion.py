import csv
import nltk

from nltk.tokenize import word_tokenize
from nltk.tree.tree import Tree
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from flask import Flask, request
from typing import List

app = Flask('test')


csv_file_name: str = '/app/biased_words_and_suggestions.csv'
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
threshold = 0.9


def read_biased_words_and_suggestions() -> List[dict]:
    biased_words_data: list[dict] = []
    with open(csv_file_name, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_index, row in enumerate(csv_reader):
            if row_index == 0:
                type_index: int = row.index('type')
                row_length: int = len(row)
            else:
                biased_words: list[str] = []
                suggestions: list[str] = []
                for biased_word_index in range(type_index):
                    biased_word: str = row[biased_word_index]
                    if not biased_word or not biased_word.strip():
                        break
                    biased_words.append(biased_word.strip())
                for suggestion_index in range(type_index + 1, row_length):
                    suggestion: str = row[suggestion_index]
                    if not suggestion or not suggestion.strip():
                        break
                    suggestions.append(suggestion.strip())
                biased_words_data.append({
                    'biased_words': biased_words,
                    'type': row[type_index].strip() if row[type_index] and row[type_index].strip() else 'other',
                    'suggestions': suggestions
                })
    return biased_words_data


def get_phrases_from_sentence(sentence: str) -> List[str]:
    grammar: str = 'NP: {<DT>?<JJ>*<NN|NNS>+<IN>*<NN|NNS>*}'
    chunk_parser: nltk.RegexpParser = nltk.RegexpParser(grammar)
    tokenized_words: list[str] = word_tokenize(sentence)
    pos_tags: list[(str, str)] = nltk.pos_tag(tokenized_words)
    tree: Tree = chunk_parser.parse(pos_tags)
    # tree.draw()
    return [
        ' '.join(leaf[0] for leaf in subtree.leaves())
        for subtree in tree.subtrees(lambda subtree: subtree.label() == 'NP')
    ]


def populate_biased_word_embeddings(biased_words_data: List[dict]):
    for biased_word_data in biased_words_data:
        biased_word_data['biased_words_embeddings'] = model.encode(biased_word_data['biased_words'])


def get_dei_suggestions(phrases: List[str], biased_words_data: List[dict]) -> List[dict]:
    phrase_embeddings = model.encode(phrases)
    dei_suggestions: list[dict] = []
    phrase_indices_matched: list[int] = []
    for biased_word_data in biased_words_data:
        biased_words_embeddings = biased_word_data['biased_words_embeddings']
        for biased_word_embedding in biased_words_embeddings:
            for phrase_index, phrase_embedding in enumerate(phrase_embeddings):
                if phrase_index in phrase_indices_matched:
                    continue
                similarity = 1 - distance.cosine(biased_word_embedding, phrase_embedding)
                if similarity >= threshold:
                    dei_suggestions.append({
                        'phrase': phrases[phrase_index],
                        'bias_type': biased_word_data['type'],
                        'suggestions': biased_word_data['suggestions'],
                        'suggestion_embeddings': model.encode(biased_word_data['suggestions'])
                    })
                    phrase_indices_matched.append(phrase_index)
    return dei_suggestions


def replace_phrase_in_text(text: str, phrase: str, suggestion: str) -> str:
    if phrase not in text:
        return text
    if text == phrase:
        return suggestion.capitalize() if 'A' <= phrase[0] <= 'Z' else suggestion

    phrase_length: int = len(phrase)
    replace_phrase_in_sentence = False
    phrase_start_index = text.index(phrase)
    if phrase_start_index == 0:
        char_after_phrase = text[phrase_length]
        if (char_after_phrase < 'a' or char_after_phrase > 'z') and (char_after_phrase < 'A' or char_after_phrase > 'Z'):
            replace_phrase_in_sentence = True
    elif phrase_start_index + phrase_length == len(text):
        char_before_phrase = text[phrase_start_index - 1]
        if (char_before_phrase < 'a' or char_before_phrase > 'z') and (char_before_phrase < 'A' or char_before_phrase > 'Z'):
            replace_phrase_in_sentence = True
    else:
        char_before_phrase = text[phrase_start_index - 1]
        char_after_phrase = text[phrase_start_index + phrase_length]
        if (char_before_phrase < 'a' or char_before_phrase > 'z') and (char_before_phrase < 'A' or char_before_phrase > 'Z') \
                and (char_after_phrase < 'a' or char_after_phrase > 'z') and (char_after_phrase < 'A' or char_after_phrase > 'Z'):
            replace_phrase_in_sentence = True

    if replace_phrase_in_sentence:
        suggestion_capitalize = False
        if 'A' <= phrase[0] <= 'Z':
            suggestion_capitalize = True
        text = text.replace(phrase, suggestion.capitalize() if suggestion_capitalize else suggestion)
    return text


def get_suggested_sentences(sentence: str, dei_suggestions: List[dict]) -> dict:
    bias_types: list[str] = []
    for dei_suggestion in dei_suggestions:
        bias_type: str = dei_suggestion['bias_type']
        if bias_type not in bias_types:
            bias_types.append(bias_type)
        closest_suggestion_index = -1
        least_embedding_difference = 2
        phrase_embedding = model.encode(dei_suggestion['phrase'])
        # Among dei_suggestion['suggestions'], use the suggestion that's closest in meaning to dei_suggestion['phrase'] - using embeddings
        for suggestion_index, suggestion_embedding in enumerate(dei_suggestion['suggestion_embeddings']):
            embedding_difference = distance.cosine(phrase_embedding, suggestion_embedding)
            if embedding_difference < least_embedding_difference:
                least_embedding_difference = embedding_difference
                closest_suggestion_index = suggestion_index
        # print(f'Replacing {dei_suggestion["phrase"]} by {dei_suggestion["suggestions"][closest_suggestion_index]}')
        sentence = replace_phrase_in_text(
            sentence, dei_suggestion['phrase'], dei_suggestion['suggestions'][closest_suggestion_index])
        # print(sentence)
    return {
        'biasTypes': bias_types,
        'newSentence': sentence
    }

@app.route('/')
def run():
    input_string = request.args.get('data')
    biased_words_data: list[dict] = read_biased_words_and_suggestions()
    populate_biased_word_embeddings(biased_words_data)
    phrases: list[str] = get_phrases_from_sentence(input_string)
    # print(phrases)
    # print()
    dei_suggestions: list[dict] = get_dei_suggestions(phrases, biased_words_data)
    # print(dei_suggestions)
    suggested_sentence: dict = get_suggested_sentences(input_string, dei_suggestions)
    print(f'Bias types in this sentence are: {suggested_sentence["biasTypes"]}')
    print()
    return suggested_sentence['newSentence']


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
