"""
Gemini API integration for extracting lab values from PDF lab reports.

This module uses Google's Gemini Vision API to extract the following values from lab reports:
- Age
- Gender
- Rheumatoid Factor (RF) [IU/mL]
- Anti-CCP [U/mL]
- C-Reactive Protein (CRP) [mg/dL]
- Erythrocyte Sedimentation Rate (ESR) [mm/h]

Setup:
1. Get your Gemini API key from: https://makersuite.google.com/app/apikeys
2. Set environment variable: export GOOGLE_API_KEY="your-api-key-here"
   OR
3. Paste your API key in config.py in the GEMINI_API_KEY variable
4. The module will use either method automatically
"""

import os
import json
import base64
import re
from typing import Dict, Tuple, Optional
import google.generativeai as genai


class GeminiPDFExtractor:
    """Extract lab values from PDF reports using Gemini Vision API."""
    
    # Required lab values for RA prediction
    REQUIRED_VALUES = {
        'age': 'Age (in years)',
        'gender': 'Gender (Male/Female)',
        'rf': 'Rheumatoid Factor (RF) [IU/mL]',
        'anti_ccp': 'Anti-CCP [U/mL]',
        'crp': 'C-Reactive Protein (CRP) [mg/dL]',
        'esr': 'Erythrocyte Sedimentation Rate (ESR) [mm/h]'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini extractor with API key.
        
        Args:
            api_key: Gemini API key. If None, will try to load from:
                    1. Environment variable GOOGLE_API_KEY
                    2. config.py GEMINI_API_KEY
                    3. config.py GEMINI_API_KEY_ALTERNATE (fallback)
        """
        if api_key is None:
            # Try environment variable first
            api_key = os.getenv('GOOGLE_API_KEY')
            
            # Try primary config key
            if not api_key:
                try:
                    from config import GEMINI_API_KEY
                    if GEMINI_API_KEY:
                        api_key = GEMINI_API_KEY
                except (ImportError, AttributeError):
                    pass
            
            # Try alternate config key as fallback
            if not api_key:
                try:
                    from config import GEMINI_API_KEY_ALTERNATE
                    if GEMINI_API_KEY_ALTERNATE:
                        api_key = GEMINI_API_KEY_ALTERNATE
                except (ImportError, AttributeError):
                    pass
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please:\n"
                "1. Get key from: https://makersuite.google.com/app/apikeys\n"
                "2. Set GOOGLE_API_KEY environment variable, OR\n"
                "3. Add GEMINI_API_KEY to config.py, OR\n"
                "4. Add GEMINI_API_KEY_ALTERNATE to config.py"
            )
        
        self.api_key = api_key
        try:
            genai.configure(api_key=api_key)
            # Try gemini-2.0-flash first (latest model), fall back to gemini-1.5-pro
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                self.model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {str(e)}")
    
    def extract_from_pdf(self, pdf_path: str) -> Tuple[bool, Dict[str, any], str]:
        """
        Extract lab values from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success: bool, data: dict, message: str)
            - success: True if extraction successful
            - data: Dictionary with extracted values (empty if failed)
            - message: Status message or error description
        """
        try:
            # Read PDF file
            if not os.path.exists(pdf_path):
                return False, {}, f"File not found: {pdf_path}"
            
            if not pdf_path.lower().endswith('.pdf'):
                return False, {}, "File must be a PDF"
            
            # Convert PDF to base64
            with open(pdf_path, 'rb') as f:
                pdf_data = base64.standard_b64encode(f.read()).decode('utf-8')
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt()
            
            # Send to Gemini with PDF
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": "application/pdf",
                    "data": pdf_data
                }
            ])
            
            # Parse response
            extracted_text = response.text
            values = self._parse_response(extracted_text)
            
            if not values:
                return False, {}, "Could not extract lab values from PDF. Ensure it contains:\n" + \
                       "\n".join([f"- {v}" for v in self.REQUIRED_VALUES.values()])
            
            return True, values, f"Successfully extracted {len(values)} lab values"
            
        except ValueError as e:
            return False, {}, f"Value error: {str(e)}"
        except Exception as e:
            return False, {}, f"Error processing PDF: {str(e)}"
    
    def extract_from_image_bytes(self, image_bytes: bytes, filename: str = "report") -> Tuple[bool, Dict[str, any], str]:
        """
        Extract lab values from image bytes (from file upload).
        
        Args:
            image_bytes: Image file bytes (PNG, JPG, GIF, WebP)
            filename: Original filename for context
            
        Returns:
            Tuple of (success: bool, data: dict, message: str)
        """
        try:
            # Encode to base64
            image_data = base64.standard_b64encode(image_bytes).decode('utf-8')
            
            # Determine MIME type
            mime_type = self._get_mime_type(filename)
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt()
            
            # Send to Gemini with image
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": mime_type,
                    "data": image_data
                }
            ])
            
            # Parse response
            extracted_text = response.text
            values = self._parse_response(extracted_text)
            
            if not values:
                return False, {}, "Could not extract lab values from image."
            
            return True, values, f"Successfully extracted {len(values)} lab values"
            
        except Exception as e:
            return False, {}, f"Error processing image: {str(e)}"
    
    def _create_extraction_prompt(self) -> str:
        """Create the prompt for Gemini to extract lab values."""
        return """Please extract the following lab test values from this medical lab report:

1. Age (in years) - look for "Age", "DOB", "Date of Birth"
2. Gender (Male/Female) - look for "Gender", "Sex", "M/F"
3. Rheumatoid Factor (RF) [IU/mL] - look for "RF", "Rheumatoid Factor"
4. Anti-CCP [U/mL] - look for "Anti-CCP", "Anti-Cyclic Citrullinated Peptide"
5. C-Reactive Protein (CRP) [mg/dL] - look for "CRP", "C-Reactive Protein"
6. Erythrocyte Sedimentation Rate (ESR) [mm/h] - look for "ESR", "Sed Rate"

Return ONLY a valid JSON object with these exact keys (use null for missing values):
{
    "age": <number or null>,
    "gender": "<Male|Female or null>",
    "rf": <number or null>,
    "anti_ccp": <number or null>,
    "crp": <number or null>,
    "esr": <number or null>,
    "confidence": "<high|medium|low>",
    "notes": "<any relevant notes>"
}

IMPORTANT: Return ONLY valid JSON, no other text."""
    
    def _parse_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse Gemini's JSON response.
        
        Args:
            response_text: Raw text response from Gemini
            
        Returns:
            Dictionary with extracted values, or empty dict if parsing fails
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return {}
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate and clean data
            cleaned = {}
            
            # Age: must be number 0-150
            if data.get('age') is not None:
                age = float(data['age']) if data['age'] else None
                if age and 0 <= age <= 150:
                    cleaned['age'] = int(age)
            
            # Gender: must be Male/Female
            if data.get('gender'):
                gender_str = str(data['gender']).strip().lower()
                if 'female' in gender_str or 'f' == gender_str:
                    cleaned['gender'] = 'Female'
                elif 'male' in gender_str or 'm' == gender_str:
                    cleaned['gender'] = 'Male'
            
            # Lab values: must be positive numbers
            for key in ['rf', 'anti_ccp', 'crp', 'esr']:
                if data.get(key) is not None:
                    val = float(data[key]) if data[key] else None
                    if val is not None and val >= 0:
                        cleaned[key] = float(val)
            
            return cleaned
            
        except (json.JSONDecodeError, ValueError, AttributeError):
            return {}
    
    @staticmethod
    def _get_mime_type(filename: str) -> str:
        """Determine MIME type from filename."""
        ext = filename.lower().split('.')[-1]
        mime_types = {
            'pdf': 'application/pdf',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        return mime_types.get(ext, 'application/octet-stream')


def extract_lab_values_from_report(file_path: str, api_key: Optional[str] = None) -> Tuple[bool, Dict[str, any], str]:
    """
    Convenience function to extract lab values from a report file.
    
    Args:
        file_path: Path to report file (PDF or image)
        api_key: Optional Gemini API key
        
    Returns:
        Tuple of (success, values_dict, message)
    """
    try:
        extractor = GeminiPDFExtractor(api_key=api_key)
        
        if file_path.lower().endswith('.pdf'):
            return extractor.extract_from_pdf(file_path)
        else:
            with open(file_path, 'rb') as f:
                return extractor.extract_from_image_bytes(f.read(), file_path)
                
    except Exception as e:
        return False, {}, f"Extraction failed: {str(e)}"


# Example usage:
if __name__ == "__main__":
    # Get API key from environment or config
    extractor = GeminiPDFExtractor()
    
    # Extract from PDF
    success, values, message = extractor.extract_from_pdf("sample_report.pdf")
    print(f"Success: {success}")
    print(f"Values: {values}")
    print(f"Message: {message}")
