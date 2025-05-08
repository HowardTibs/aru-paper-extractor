import os
import torch
import json
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class ModelLoader:
    @staticmethod
    def load_tokenizer(model_path):
        """Load tokenizer directly from the model path"""
        print(f"Loading tokenizer from {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise e
    
    @staticmethod
    def load_model(model_path, device="cpu"):
        """Load the actual model directly from the model path"""
        print(f"Loading full model from {model_path}")
        
        # Check if model path exists and list its contents
        if os.path.exists(model_path):
            print(f"Model directory exists. Contents: {os.listdir(model_path)}")
        else:
            print(f"Warning: Model directory {model_path} does not exist!")
            return None
        
        # Check for the critical configuration file
        if not os.path.exists(os.path.join(model_path, 'configuration_phi3.py')):
            print(f"Warning: configuration_phi3.py not found in {model_path}")
            print("This file is required for model loading.")
        
        try:
            # First load the config to check what we're dealing with
            config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            print(f"Model config: {config.model_type}, vocab size: {config.vocab_size}")
            
            # Now load the actual model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to the appropriate device
            model = model.to(device)
            model.eval()
            
            print(f"Model loaded successfully on {device}")
            return model
            
        except Exception as e:
            print(f"Error during model loading: {e}")
            raise e