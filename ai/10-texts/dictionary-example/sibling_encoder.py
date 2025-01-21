class SiblingEncoder:
    def __init__(self):
        self.new_encoding = 256
        self.dictionary = { c: chr(c) for c in range(self.new_encoding) }

    def translate(self, text_to_encode):
        text = text_to_encode.lower().replace(" ", "").replace("\n", "")
        encoded_text = [ord(c) for c in text]

        while True:
            encoded_text, has_repeated = self.find_and_replace(encoded_text)
            if not has_repeated:
                break

        return encoded_text

    def find_and_replace(self, encoded_text):
        has_repeated = False

        for i in range(len(encoded_text) - 1):
            pair = (encoded_text[i], encoded_text[i + 1])
            has_repeated_pair = self.count_pairs(encoded_text, pair, i + 1)
            has_repeated_pair = has_repeated_pair or len(encoded_text) == 2

            if pair not in self.dictionary and has_repeated_pair:
                self.dictionary[pair] = self.new_encoding
                self.new_encoding += 1
                has_repeated = True

        if has_repeated:
            encoded_text = self.replace_in_dictionary(encoded_text)

        return encoded_text, has_repeated

    def replace_in_dictionary(self, encoded_text):
        new_encoded_text = []
        skip_next = False

        for i in range(len(encoded_text) - 1):
            if skip_next:
                skip_next = False
                continue

            pair = (encoded_text[i], encoded_text[i + 1])
            if pair in self.dictionary:
                new_encoded_text.append(self.dictionary[pair])
                skip_next = True
            else:
                new_encoded_text.append(encoded_text[i])

        if not skip_next:
            new_encoded_text.append(encoded_text[-1])

        return new_encoded_text

    def count_pairs(self, encoded_text, pair, index):
        has_repeated = False

        for i in range(index, len(encoded_text) - 1):
            if (encoded_text[i], encoded_text[i + 1]) == pair:
                has_repeated = True

        return has_repeated

    def get_dictionary(self):
        return self.dictionary
