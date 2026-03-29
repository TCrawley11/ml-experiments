import pandas as pd
from sklearn.model_selection import train_test_split

# need to create a 51760 x 1 numpy arr of the dataset with the formatted instruction and output

def get_format_split():
    """
    Returns three numpy arrays for train, test, and eval split datasets.
    """
    df = pd.read_json("hf://datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json")

    # format each column respectively
    input_instruction_text = (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        "Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
        "\n\n### Input:\n{input}\n\n### Response:\n{output}"
    )
    no_input_instruction_text = (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        "Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n{output}"
    )

    def format_row(row):
        if row['input'].strip():
            return input_instruction_text.format(
            instruction = row['instruction'],
            input = row['input'],
            output = row['output']
            )
        else:
            return no_input_instruction_text.format(
            instruction = row['instruction'],
            output = row['output']
            )

    df['text'] = df.apply(format_row,axis=1)

    # make it a numpy arr along one column
    text_arr = df['text'].to_numpy()

    # split 85% train 10% test 5% eval
    train_arr, temp_arr = train_test_split(text_arr, test_size=0.15, random_state=42)
    test_arr, eval_arr = train_test_split(temp_arr, test_size=1/3, random_state=42)

    print(f"Succesfully downloaded and split instruction dataset of length: {len(text_arr)}")
    print(f"Train: {len(train_arr)}, Test: {len(test_arr)}, Eval: {len(eval_arr)}")

    return train_arr, test_arr, eval_arr
