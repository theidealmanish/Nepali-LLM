def is_utf8_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            file.read()
        return True
    except UnicodeDecodeError:
        return False

# Example usage
filename = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/merged_nepali_corpus.txt"
if is_utf8_file(filename):
    print("The file is UTF-8 encoded")
else:
    print("The file is not UTF-8 encoded")
