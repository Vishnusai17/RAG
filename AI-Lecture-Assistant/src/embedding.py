"""Image embedding generation using CLIP."""

import torch
import open_clip
import numpy as np
from PIL import Image
from typing import List, Tuple
from pathlib import Path

from src.config import CLIP_MODEL


class ImageEmbedder:
    """Generate CLIP embeddings for video frames."""
    
    def __init__(self, model_name: str = CLIP_MODEL):
        """
        Initialize CLIP model.
        
        Args:
            model_name: CLIP model architecture (default: ViT-L-14)
        """
        print(f"Loading CLIP model '{model_name}'...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained='openai'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer for text queries
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        print(f"CLIP model loaded successfully on {self.device}")
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding as numpy array
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    
    def embed_images_batch(self, image_paths: List[str], 
                          batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            
        Returns:
            Array of normalized embeddings (shape: [num_images, embedding_dim])
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess batch
            images = []
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                images.append(self.preprocess(image))
            
            image_tensors = torch.stack(images).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(image_tensors)
                # Normalize
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text query.
        Used for image retrieval based on text queries.
        
        Args:
            text: Text query
            
        Returns:
            Normalized embedding as numpy array
        """
        # Tokenize text
        text_tokens = self.tokenizer([text]).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_text(text_tokens)
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
