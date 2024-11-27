from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

class Brain:
    def __init__(self, model_name='distilgpt2', logger=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.logger = logger

    def _generate(self, prompt):
        """Generate a response from the language model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def decide_hands(self, prompt):
        """Uses the LLM to decide the number of hands to bid."""
        response = self._generate(prompt)
        match = re.search(r'\b\d+\b', response)
        hands = int(match.group()) if match else 0  # Default to 0 if no valid response
        if self.logger:
            self.logger.info(f"LLM decided hands: {hands}")
        return hands

    def decide_card(self, prompt):
        """Uses the LLM to decide which card to play."""
        response = self._generate(prompt)
        match = re.search(r'\b([SHDC][AKQJ2-9]0?)\b', response)  # Matches valid card formats
        card = match.group(1) if match else None
        if self.logger:
            self.logger.info(f"LLM decided card: {card}")
        return card




