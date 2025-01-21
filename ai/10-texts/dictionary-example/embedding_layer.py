from sibling_decoder import SiblingDecoder
from sibling_encoder import SiblingEncoder

example_text = """
test test examples
test test examples
"""


def get_final_dictionary(diccionario):
    result = {}

    def obtener_valor(key):
        if isinstance(key, tuple):
            try:
                return ''.join([obtener_valor(k) for k in key])
            except KeyError:
                return "KeyError"
        return diccionario.get(key, '')

    for key, value in diccionario.items():
        if isinstance(key, tuple):
            result[key] = obtener_valor(key)
        else:
            result[key] = value

    return result

if __name__ == "__main__":
    encoder = SiblingEncoder()
    encoded_text = encoder.translate(example_text)
    # print(encoded_text)

    dictionary = encoder.get_dictionary()
    print(dictionary)
    print(get_final_dictionary(dictionary))

    decoder = SiblingDecoder(encoder.get_dictionary())
    decoded_text = decoder.translate(encoded_text)
    # print(decoded_text)
