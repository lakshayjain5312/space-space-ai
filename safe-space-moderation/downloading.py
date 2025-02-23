from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify a local directory where you want to store the model permanently.
local_model_dir = snapshot_download("Hate-speech-CNERG/hindi-abusive-MuRIL", local_dir="./my_local_model")

# Load the model and tokenizer from that directory.
tokenizer_of_hindi = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True)
model_of_hindi = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
model_of_hindi.eval()