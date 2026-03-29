"""
Structured Output Handler for LLM Responses

Robustly extracts and validates JSON from any LLM output format.
Handles markdown code blocks, preambles, malformed JSON, and validation errors.

Usage:
    from utils.structured_output import StructuredOutputHandler
    from pydantic import BaseModel
    
    class UserInfo(BaseModel):
        name: str
        age: int
    
    result = StructuredOutputHandler.parse(llm_response, UserInfo)
    print(result.name)

Author: Jeffrey Dean (based on 57 articles of LLM best practices)
"""

from typing import Type, TypeVar, Optional, List
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass
import json
import re

T = TypeVar('T', bound=BaseModel)

# Check json_repair availability at import time, not inside exception handlers
try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


@dataclass
class ParseResult:
    """Result of parsing attempt with debugging context"""
    success: bool
    data: Optional[BaseModel]
    raw_output: str
    extracted_json: Optional[str]
    error: Optional[str]
    repair_attempted: bool


class StructuredOutputError(Exception):
    """
    Raised when structured output parsing fails.
    
    Includes full context for debugging:
    - Original LLM output
    - Extracted JSON (if found)
    - Expected schema
    - Detailed error message
    """
    
    def __init__(
        self,
        message: str,
        raw_output: str,
        extracted_json: Optional[str],
        model_class: Type[BaseModel]
    ):
        self.message = message
        self.raw_output = raw_output
        self.extracted_json = extracted_json
        self.model_class = model_class
        
        # Create detailed error message for debugging
        details = [
            f"Structured Output Parsing Failed: {message}",
            "",
            f"Expected Model: {model_class.__name__}",
            ""
        ]
        
        if len(raw_output) > 500:
            details.extend([
                "Raw Output (first 500 chars):",
                raw_output[:500] + "...",
                ""
            ])
        else:
            details.extend([
                "Raw Output:",
                raw_output,
                ""
            ])
        
        if extracted_json:
            if len(extracted_json) > 500:
                details.extend([
                    "Extracted JSON (first 500 chars):",
                    extracted_json[:500] + "...",
                    ""
                ])
            else:
                details.extend([
                    "Extracted JSON:",
                    extracted_json,
                    ""
                ])
        
        details.extend([
            "Expected Schema:",
            json.dumps(model_class.model_json_schema(), indent=2)
        ])
        
        super().__init__("\n".join(details))


class StructuredOutputHandler:
    """
    Robust structured output parsing for LLM responses.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Preambles and postambles
    - Malformed JSON (missing quotes, trailing commas)
    - Multiple JSON objects
    - Validation errors with context
    
    Based on lessons from:
    - Article #40 (XGrammar): Structured output prevents 90% of failures
    - Article #55 (GRPO): JSON repair critical for malformed output
    - Article #22, #23 (Prompting): Clear schema improves compliance
    """
    
    @staticmethod
    def parse(
        raw_output: str,
        model_class: Type[T],
        strict: bool = False,
        allow_repair: bool = True
    ) -> T:
        """
        Parse LLM output into Pydantic model.
        
        Args:
            raw_output: Raw text from LLM (can include markdown, preambles, etc.)
            model_class: Target Pydantic model class
            strict: If True, raise on first error. If False, try repair
            allow_repair: If True, attempt JSON repair on parse failures
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            StructuredOutputError: If parsing/validation fails with full context
            
        Example:
            >>> class User(BaseModel):
            ...     name: str
            ...     age: int
            >>> 
            >>> response = '''
            ... Here's the user info:
            ... ```json
            ... {"name": "Alice", "age": 30}
            ... ```
            ... '''
            >>> user = StructuredOutputHandler.parse(response, User)
            >>> print(user.name)  # "Alice"
        """
        result = StructuredOutputHandler.parse_with_details(
            raw_output,
            model_class,
            strict,
            allow_repair
        )
        
        if not result.success:
            raise StructuredOutputError(
                message=result.error,
                raw_output=raw_output,
                extracted_json=result.extracted_json,
                model_class=model_class
            )
        
        return result.data
    
    @staticmethod
    def parse_with_details(
        raw_output: str,
        model_class: Type[T],
        strict: bool = False,
        allow_repair: bool = True
    ) -> ParseResult:
        """
        Parse with full debugging details.
        
        Returns ParseResult with success status and metadata.
        Useful for logging/debugging failed parses without raising.
        
        Example:
            >>> result = StructuredOutputHandler.parse_with_details(
            ...     llm_output, UserModel
            ... )
            >>> if result.success:
            ...     print(f"Parsed: {result.data}")
            ... else:
            ...     logger.error(f"Parse failed: {result.error}")
        """
        
        # Step 1: Extract JSON from raw output
        extracted = StructuredOutputHandler._extract_json(raw_output)
        
        if not extracted:
            return ParseResult(
                success=False,
                data=None,
                raw_output=raw_output,
                extracted_json=None,
                error="No JSON found in response",
                repair_attempted=False
            )
        
        # Step 2: Try strict parsing first
        try:
            data = model_class.model_validate_json(extracted)
            return ParseResult(
                success=True,
                data=data,
                raw_output=raw_output,
                extracted_json=extracted,
                error=None,
                repair_attempted=False
            )
        except (json.JSONDecodeError, ValidationError) as e:
            if strict:
                return ParseResult(
                    success=False,
                    data=None,
                    raw_output=raw_output,
                    extracted_json=extracted,
                    error=f"Strict mode: {str(e)}",
                    repair_attempted=False
                )
            
            # Step 3: Try JSON repair if allowed
            if allow_repair:
                try:
                    if not _HAS_JSON_REPAIR:
                        raise ImportError("json_repair not installed")
                    repaired = json_repair.loads(extracted)
                    data = model_class.model_validate(repaired)
                    
                    return ParseResult(
                        success=True,
                        data=data,
                        raw_output=raw_output,
                        extracted_json=extracted,
                        error=None,
                        repair_attempted=True
                    )
                except Exception as repair_error:
                    return ParseResult(
                        success=False,
                        data=None,
                        raw_output=raw_output,
                        extracted_json=extracted,
                        error=f"Parse failed: {e}\nRepair failed: {repair_error}",
                        repair_attempted=True
                    )
            else:
                return ParseResult(
                    success=False,
                    data=None,
                    raw_output=raw_output,
                    extracted_json=extracted,
                    error=str(e),
                    repair_attempted=False
                )
    
    @staticmethod
    def _extract_json(raw_output: str) -> Optional[str]:
        """
        Extract JSON from various formats.
        
        Handles:
        - ```json ... ``` (markdown with language)
        - ``` ... ``` (markdown without language)
        - Plain JSON with preamble/postamble
        - Multiple potential JSON objects (returns first valid)
        
        Uses json.JSONDecoder.raw_decode() instead of brace tracking
        to correctly handle braces inside JSON string values.
        """
        
        if not raw_output or not raw_output.strip():
            return None
        
        # Strategy 1: Look for markdown code blocks
        # Match ```json ... ``` or ``` ... ```
        json_block = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```',
            raw_output,
            re.DOTALL | re.IGNORECASE
        )
        
        if json_block:
            return json_block.group(1).strip()
        
        # Strategy 2: Use JSONDecoder.raw_decode to find first valid JSON
        # This correctly handles braces/brackets inside string values
        decoder = json.JSONDecoder()
        text = raw_output.strip()
        
        for i, char in enumerate(text):
            if char in '{[':
                try:
                    obj, end_idx = decoder.raw_decode(text, i)
                    return text[i:end_idx]
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Try the whole thing as-is
        try:
            json.loads(text)
            return text
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    @staticmethod
    def create_schema_prompt(
        model_class: Type[BaseModel],
        include_examples: bool = True
    ) -> str:
        """
        Generate prompt instructions for structured output.
        
        Creates clear, LLM-friendly schema documentation that improves
        compliance with expected output format.
        
        Args:
            model_class: Pydantic model to document
            include_examples: Include example JSON in prompt
            
        Returns:
            Formatted prompt string to append to user query
            
        Example:
            >>> class Analysis(BaseModel):
            ...     summary: str
            ...     score: int
            >>> 
            >>> prompt = "Analyze this text"
            >>> schema = StructuredOutputHandler.create_schema_prompt(Analysis)
            >>> full_prompt = f"{prompt}\n\n{schema}"
            >>> response = llm.generate(full_prompt)
        """
        
        schema = model_class.model_json_schema()
        
        prompt_parts = [
            "Please format your response as valid JSON matching this schema:",
            "",
            "```json",
            json.dumps(schema, indent=2),
            "```",
            "",
            "IMPORTANT:",
            "- Return ONLY the JSON object, no other text",
            "- Use double quotes for strings",
            "- Do not include comments in the JSON",
            "- Ensure all required fields are present",
            "- Follow the exact field names shown in the schema"
        ]
        
        if include_examples and 'properties' in schema:
            # Generate a simple example
            example = StructuredOutputHandler._generate_example(model_class)
            if example:
                prompt_parts.extend([
                    "",
                    "Example format:",
                    "```json",
                    json.dumps(example, indent=2),
                    "```"
                ])
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def _generate_example(model_class: Type[BaseModel]) -> Optional[dict]:
        """Generate example data for schema"""
        
        try:
            schema = model_class.model_json_schema()
            example = {}
            
            for field_name, field_info in schema.get('properties', {}).items():
                field_type = field_info.get('type', 'string')
                
                if field_type == 'string':
                    example[field_name] = f"example_{field_name}"
                elif field_type == 'integer':
                    example[field_name] = 42
                elif field_type == 'number':
                    example[field_name] = 3.14
                elif field_type == 'boolean':
                    example[field_name] = True
                elif field_type == 'array':
                    example[field_name] = ["item1", "item2"]
                elif field_type == 'object':
                    example[field_name] = {"key": "value"}
            
            return example
        except Exception:
            return None
    
    @staticmethod
    def parse_multiple(
        raw_output: str,
        model_class: Type[T],
        allow_repair: bool = True
    ) -> List[T]:
        """
        Parse multiple JSON objects from single response.
        
        Useful when LLM returns array or multiple objects.
        
        Args:
            raw_output: LLM response containing array of objects
            model_class: Pydantic model for each object
            allow_repair: Whether to repair malformed objects
            
        Returns:
            List of validated Pydantic model instances
            
        Example:
            >>> response = '''
            ... [
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25}
            ... ]
            ... '''
            >>> users = StructuredOutputHandler.parse_multiple(response, User)
            >>> print(len(users))  # 2
        """
        
        extracted = StructuredOutputHandler._extract_json(raw_output)
        if not extracted:
            return []
        
        try:
            # Try as array first
            data = json.loads(extracted)
            if isinstance(data, list):
                results = []
                for item in data:
                    try:
                        validated = model_class.model_validate(item)
                        results.append(validated)
                    except ValidationError:
                        if allow_repair:
                            if not _HAS_JSON_REPAIR:
                                raise ImportError("json_repair not installed")
                            repaired = json_repair.loads(json.dumps(item))
                            validated = model_class.model_validate(repaired)
                            results.append(validated)
                        else:
                            raise
                return results
            else:
                # Single object, return as list of one
                validated = model_class.model_validate(data)
                return [validated]
        except Exception as e:
            raise StructuredOutputError(
                message=f"Failed to parse multiple objects: {e}",
                raw_output=raw_output,
                extracted_json=extracted,
                model_class=model_class
            )
