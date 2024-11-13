from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

paths = [str(x) for x in Path("nepali-text").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=False, add_prefix_space=False)

vocab_size = 50000
min_frequency = 3
special_tokens=["<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
                ]
# Customize training
tokenizer.train(files=paths, 
                vocab_size=vocab_size, 
                min_frequency=3,
                special_tokens=special_tokens,
                show_progress=True,
                )
#Save the Tokenizer to disk
tokenizer.save_model("Robert")
tokenizer.save("Robert/config.json")