from sibling_decoder import SiblingDecoder
from sibling_encoder import SiblingEncoder

example_text = """
test2 test examples
test2 test examples
"""


def get_final_dictionary(dictionary):
    result = {}
    reversed_dictionary = {}

    for k, v in dictionary.items():
        if isinstance(k, tuple):
            reversed_dictionary.update({v: k})
            continue

        reversed_dictionary.update({k: v})

    def get_value(key):
        new_key = reversed_dictionary.get(key, '')

        if isinstance(new_key, tuple):
            try:
                return ''.join([get_value(k) for k in new_key])
            except KeyError:
                return "KeyError"

        return new_key

    for key, value in reversed_dictionary.items():
        if isinstance(value, tuple):
            result[key] = get_value(key)
        else:
            result[key] = value

    return result

if __name__ == "__main__":
    encoder = SiblingEncoder()
    encoded_text = encoder.translate(example_text)
    # print(encoded_text)

    dictionary = encoder.get_dictionary()
    print(get_final_dictionary(dictionary))

    decoder = SiblingDecoder(encoder.get_dictionary())
    decoded_text = decoder.translate(encoded_text)
    # print(decoded_text)
