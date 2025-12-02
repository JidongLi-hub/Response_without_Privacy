from utils import *


def extract_last_hidden_state_for_PS_token(
        model,
        tokenizer, 
        dataset,
        output_file="data/hidden_states_PS_O.npy"
    ):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    batch_size=1

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    PS_hidden_states = []
    O_hidden_states = []

    model.eval()
    for batch in tqdm(data_loader, desc="Extracting hidden states"):
        inputs = tokenizer(batch["source_text"], return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        new_labels = batch["new_labels"]
        PS_index = [i for i, label in enumerate(new_labels) if label[0] == "[PS]"]
        O_index = []
        for _ in range(len(PS_index)):
            while True:
                i = random.randint(0,len(new_labels)-1)
                if new_labels[i][0] == "O":
                    O_index.append(i)
                    break

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        last_hidden_states = outputs.hidden_states[-1][0]  # last hidden state shape: (batch_size, seq_len, hidden_size4096)
        for PS_i, O_i in zip(PS_index, O_index):
            PS_hidden_states.append(last_hidden_states[PS_i, :].cpu().numpy())
            O_hidden_states.append(last_hidden_states[O_i, :].cpu().numpy())
        del outputs
        torch.cuda.empty_cache()
    hidden_states = PS_hidden_states + O_hidden_states
    labels = [1]*len(PS_hidden_states) + [0]*len(O_hidden_states)
    # save the hidden states and labels to train classifier model
    np.save(output_file, {'hidden_states': hidden_states, 'labels': labels})
    print(f"Saved hidden states and labels to {output_file}")

        
if __name__ == "__main__":
    
    model_path="/model/fangly/mllm/ljd/models/Meta-Llama-3-8B-Instruct"
    dataset_file="data/relabeled_data.jsonl"

    dataset = load_dataset("json", data_files=dataset_file, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

    extract_last_hidden_state_for_PS_token(model, tokenizer, dataset, output_file="data/hidden_states_PS_O-Instruct.npy")
