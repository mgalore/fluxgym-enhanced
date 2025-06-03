import os
import json
import logging
from typing import Dict, Any, Optional

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class ConfigLoader:
    """Secure configuration loader for API keys and application settings."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file with fallback to environment variables."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.warning(f"Configuration file {self.config_file} not found. Using environment variables.")
                self.config = self._create_default_config()
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration structure."""
        return {
            "api_keys": {},
            "settings": {
                "max_images": 150,
                "default_resolution": 512,
                "default_epochs": 16,
                "default_learning_rate": "8e-4",
                "default_network_dim": 4,
                "default_vram": "20G"
            },
            "paths": {
                "models_dir": "models",
                "outputs_dir": "outputs",
                "datasets_dir": "datasets",
                "captions_dir": "generated_captions"
            }
        }
    
    def get_api_key(self, service: str) -> str:
        """
        Get API key for a service with fallback to environment variables.
        
        Args:
            service: The service name (e.g., 'google_api_key')
        
        Returns:
            The API key string
        
        Raises:
            ConfigurationError: If API key is not found
        """
        # First try to get from config file
        api_key = self.config.get("api_keys", {}).get(service)
        
        # Fallback to environment variable
        if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
            env_var_name = service.upper()
            api_key = os.getenv(env_var_name)
        
        if not api_key:
            available_env_vars = [
                "GOOGLE_API_KEY",
                "GOOGLE_GENERATIVE_AI_API_KEY",
                "GEMINI_API_KEY"
            ]
            error_msg = (
                f"API key for '{service}' not found. "
                f"Please set it in {self.config_file} or as an environment variable. "
                f"Available environment variables: {', '.join(available_env_vars)}"
            )
            raise ConfigurationError(error_msg)
        
        return api_key
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value with optional default."""
        return self.config.get("settings", {}).get(key, default)
    
    def get_path(self, key: str, default: str = "") -> str:
        """Get a path setting."""
        return self.config.get("paths", {}).get(key, default)
    
    def validate_config(self) -> bool:
        """Validate that required configuration is present."""
        try:
            # Check if Google API key is available
            self.get_api_key("google_api_key")
            return True
        except ConfigurationError:
            return False
    
    def create_sample_config(self) -> None:
        """Create a sample configuration file."""
        sample_config = {
            "api_keys": {
                "google_api_key": "YOUR_GOOGLE_API_KEY_HERE"
            },
            "settings": {
                "max_images": 150,
                "default_resolution": 512,
                "default_epochs": 16,
                "default_learning_rate": "8e-4",
                "default_network_dim": 4,
                "default_vram": "20G"
            },
            "paths": {
                "models_dir": "models",
                "outputs_dir": "outputs",
                "datasets_dir": "datasets",
                "captions_dir": "generated_captions"
            }
        }
        
        sample_file = "config.sample.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"Sample configuration created at {sample_file}")
        print("Please copy it to config.json and update with your actual API keys.")

# Global configuration instance
config = ConfigLoader() 