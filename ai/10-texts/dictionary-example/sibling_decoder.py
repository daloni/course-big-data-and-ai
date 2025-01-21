example_text = """
test2 test1
"""

class SiblingDecoder:
    def __init__(self, dictionary = None):
        self.dictionary = dictionary

        if dictionary is None:
            self.dictionary = { c: chr(c) for c in range(256) }

    def translate(self, encoded_text):
        reversed_dictionary = {v: k for k, v in self.dictionary.items()}

        while True:
            encoded_text, has_repeated = self.find_and_replace(encoded_text, reversed_dictionary)
            if not has_repeated:
                break

        return "".join([chr(c) for c in encoded_text])

    def find_and_replace(self, encoded_text, dictionary):
        has_repeated = False
        decoded_text = []

        for i in range(len(encoded_text)):
            if encoded_text[i] in dictionary:
                value = dictionary[encoded_text[i]]
                decoded_text.append(value[0])
                decoded_text.append(value[1])
                has_repeated = True
            else:
                decoded_text.append(encoded_text[i])

        return decoded_text, has_repeated
