from ai.rag import RetrieveAugment

rag = RetrieveAugment()
rag.load_phi3()

messages = [
    # {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    # {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": True,
}

output = rag.llm(messages, **generation_args)
print(output[0]['generated_text'])