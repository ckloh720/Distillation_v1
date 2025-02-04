import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F

from huggingface_hub import login

# Provide your Hugging Face token

# Load LLaMA 2 as the teacher model
#model_name="meta-llama/Llama-2-7b"
#model_name="meta-llama/Llama-3.1-8B"
#teacher_tokenizer = AutoTokenizer.from_pretrained(model_name)
#teacher_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

teacher_modelname = "gpt2"
teacher_tokenizer = GPT2Tokenizer.from_pretrained(teacher_modelname)
teacher_model = GPT2LMHeadModel.from_pretrained(teacher_modelname)

student_modelname = "EleutherAI/gpt-neo-125M"
student_tokenizer = GPT2Tokenizer.from_pretrained(student_modelname)
student_model = AutoModelForCausalLM.from_pretrained(student_modelname)

teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_prompt = "In law"
print ("======= Comparison test to distillation and training using prompt: ", teacher_prompt)

# test prior to distillation
# Function to generate text from a given model and prompt
def generate_text(model, tokenizer, prompt, max_length=50):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = inputs['attention_mask']

    outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=1.0, pad_token_id=model.config.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Test the teacher model
teacher_model.to('cuda' if torch.cuda.is_available() else 'cpu')
teacher_generated_text = generate_text(teacher_model, teacher_tokenizer, teacher_prompt)
print(f"Teacher Model ({teacher_modelname}) Output:\n", teacher_generated_text)

# Test the student model
student_model.to('cuda' if torch.cuda.is_available() else 'cpu')
student_prompt = teacher_prompt
student_generated_text = generate_text(student_model, student_tokenizer, student_prompt)
print(f"Student Model ({student_modelname}) Output:\n", student_generated_text)

# Prepare dataset (e.g., the Wikipedia dataset)

wiki_name = "20231101.en"
dataset = load_dataset("wikimedia/wikipedia", wiki_name, split="train[:1%]")

# Take an even smaller fraction manually
dataset = dataset.shuffle(seed=42).select(range(16))  # First 10000 samples
print ("Dataset :", dataset)

# Tokenize the dataset for the teacher and student
def tokenize_function(examples):
    return teacher_tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataloader = DataLoader(tokenized_dataset, batch_size=8)

# Distillation Loss (KL divergence)
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    # Soft labels (from teacher)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL Divergence Loss
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return loss

# Setup optimizer
optimizer = AdamW(student_model.parameters(), lr=5e-5)

# Training loop
import torch
import torch.nn.functional as F

def train(student_model, teacher_model, train_dataloader, optimizer, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)

    teacher_model.eval()  # Teacher model in eval mode (no gradients)
    student_model.train()  # Student model in train mode

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch in train_dataloader:
            # Use torch.stack() to correctly convert a list of lists into a tensor
            input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']]).to(device)
            attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in batch['attention_mask']]).to(device)

            # Forward pass through teacher model (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                teacher_prob = F.softmax(teacher_outputs.logits, dim=-1)

            #print("after forward pass thru teacher model, teacher logits: ",teacher_outputs.logits)
            print("after forward pass thru teacher model, teacher prob: ",teacher_prob)


            # Forward pass through student model
            student_outputs = student_model(input_ids, attention_mask=attention_mask)
            print("after forward pass thru student model")

            # Compute loss (KL Divergence Loss for distillation)
            loss = F.kl_div(
                F.log_softmax(student_outputs.logits, dim=-1),
                teacher_prob,
                reduction='batchmean'
            )
            print("after compute loss")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss after epoch {epoch + 1}: {loss.item()}")

    print("Training completed!")


# **Testing the Student Model**
def test_model(student_model, teacher_model, prompt="Once upon a time,"):
    student_model.eval()
    teacher_model.eval()

    print("\n==== Model Comparison Test ====")
    print(f"Prompt: {prompt}\n")

    # Generate response from the teacher model
    teacher_inputs = teacher_tokenizer(prompt, return_tensors="pt").to(student_model.device)
    with torch.no_grad():
        teacher_output = teacher_model.generate(**teacher_inputs, max_length=50)
    teacher_text = teacher_tokenizer.decode(teacher_output[0], skip_special_tokens=True)

    # Generate response from the student model
    student_inputs = student_tokenizer(prompt, return_tensors="pt").to(student_model.device)
    with torch.no_grad():
        student_output = student_model.generate(**student_inputs, max_length=50)
    student_text = student_tokenizer.decode(student_output[0], skip_special_tokens=True)

    # Print outputs
    print("**Teacher Model Output:**")
    print(teacher_text)
    print("\n**Student Model Output:**")
    print(student_text)

# Start training
train(student_model, teacher_model, train_dataloader, optimizer)

# Test the trained student model
test_model(student_model, teacher_model, prompt=teacher_prompt)
