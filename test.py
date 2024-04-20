from mlx_lm import load, generate

def main():
    
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-8bit")

    messages = [
        {"role": "system", "content": "you are helpful and skillfull programmer."},
        {"role": "user", "content": "write game of life in python."}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(
        model, 
        tokenizer,
        prompt=inputs,
        verbose=True,
        temp=0.6,
        top_p=0.9,
        max_tokens=4096,
    )

if __name__ == "__main__":
    main()