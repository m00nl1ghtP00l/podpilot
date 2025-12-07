#!/usr/bin/env python3
"""
Generate language lessons from transcriptions
Supports multiple languages via adapter system
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import concurrent.futures
import multiprocessing
from llm_providers import get_provider, LLMProvider, OllamaProvider, OpenAIProvider, AnthropicProvider
from llm_config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL


def expand_env_vars(value: str) -> str:
    """Expand environment variables in a string (supports $VAR and ${VAR} syntax)"""
    if not isinstance(value, str):
        return value
    
    def replace_var(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))  # Return original if not found
    
    return re.sub(r'\$\{(\w+)\}|\$(\w+)', replace_var, value)

# Import language adapter system
try:
    from adapters import get_language_adapter, LanguageAdapter
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False
    # Fallback: create a minimal adapter for backward compatibility
    class FallbackAdapter:
        def get_lesson_system_prompt(self):
            return """# Role
You are an expert language teacher.

# Task
Analyze text and create structured lessons with vocabulary and grammar explanations.

# Output Format
Always respond in **valid JSON format only** (no markdown code blocks, no explanatory text). Use the following structure:

```json
{
  "vocabulary": [
    {
      "word": "word in target language",
      "reading": "pronunciation/reading (if applicable)",
      "meaning": "English meaning",
      "proficiency_level": "proficiency level",
      "example_sentence": "Example sentence using the word",
      "example_translation": "English translation of example"
    }
  ],
  "grammar_points": [
    {
      "pattern": "Grammar pattern name",
      "explanation": "Explanation of how to use this grammar",
      "proficiency_level": "proficiency level",
      "example_sentence": "Example sentence",
      "example_translation": "English translation"
    }
  ],
  "key_phrases": [
    {
      "phrase": "phrase in target language",
      "translation": "English translation",
      "context": "When/where this phrase is used"
    }
  ],
  "summary": "Brief summary of the lesson content"
}
```

# Instructions
- Extract important vocabulary with meanings and proficiency levels
- Identify grammar patterns with clear explanations
- Include key phrases with context
- Provide a brief summary
- Ensure all JSON is valid and properly formatted"""
        def get_lesson_user_prompt_template(self):
            return """# Analysis Request

{episode_title_section}

## Text to Analyze

{transcription_text}

## Task
Create a comprehensive lesson by extracting:

1. **Vocabulary**: Important words with meanings and proficiency levels
2. **Grammar**: Patterns and structures with explanations
3. **Key Phrases**: Useful phrases with context
4. **Summary**: Brief overview of the content"""

# Import config loading functions
try:
    from find_podcasts import load_config, find_podcast_by_name
except ImportError:
    # Fallback if find_podcasts is not available
    def load_config(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_podcast_by_name(config, name):
        for podcast in config["youtube_channels"]:
            if podcast["channel_name_short"] == name:
                return podcast
        raise ValueError(f"Podcast '{name}' not found")


def load_transcription(transcription_path: Path) -> str:
    """Load transcription text from file"""
    try:
        # Try to load from .txt file first (clean transcript)
        txt_path = transcription_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove timestamps if present (lines starting with [)
                lines = []
                for line in content.split('\n'):
                    if not line.strip().startswith('[') and not line.strip().startswith('http'):
                        lines.append(line)
                return '\n'.join(lines).strip()
        
        # Fall back to JSON if available
        json_path = transcription_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'text' in data:
                    return data['text']
                elif isinstance(data, dict) and 'segments' in data:
                    # Extract text from segments
                    return ' '.join([seg.get('text', '') for seg in data.get('segments', [])])
        
        raise FileNotFoundError(f"Transcription file not found: {transcription_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading transcription: {e}")


def generate_lesson(provider: LLMProvider, transcription_text: str, 
                   episode_title: Optional[str] = None,
                   language_adapter: Optional[LanguageAdapter] = None,
                   prompt_variant: Optional[str] = None,
                   prompt_files: Optional[Dict] = None) -> Dict:
    """Generate a lesson from transcription text
    
    Args:
        provider: LLM provider instance
        transcription_text: Text to analyze
        episode_title: Optional episode title for context
        language_adapter: Optional language adapter (defaults to Japanese if adapters available)
    """
    # Get language adapter
    if language_adapter is None:
        if ADAPTERS_AVAILABLE:
            # Default to Japanese for backward compatibility
            language_adapter = get_language_adapter("ja")
            if language_adapter is None:
                # Fallback if Japanese adapter not registered
                language_adapter = FallbackAdapter()
        else:
            language_adapter = FallbackAdapter()
    
    # Get prompts from adapter (with optional variant and config files)
    system_prompt = language_adapter.get_lesson_system_prompt(variant=prompt_variant, prompt_files=prompt_files)
    user_prompt_template = language_adapter.get_lesson_user_prompt_template(variant=prompt_variant, prompt_files=prompt_files)
    
    # Format user prompt
    episode_title_section = f"Episode Title: {episode_title}\n\n" if episode_title else ""
    prompt = user_prompt_template.format(
        episode_title_section=episode_title_section,
        transcription_text=transcription_text
    )
    
    try:
        # Determine format based on provider type
        # For OpenAI/Anthropic: enforce JSON format via API (they support it well)
        # For Ollama: rely on prompt instructions only (more flexible, matches UI behavior)
        use_json_format = isinstance(provider, (OpenAIProvider, AnthropicProvider))
        # Don't enforce JSON format for Ollama - let it generate naturally per prompt
        # This matches behavior when pasting directly into Ollama UI
        
        response = provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent, educational output
            format=None  # No format enforcement - let LLM generate markdown naturally
        )
        
        # LLM returns markdown directly - just clean it up and return
        markdown_content = response.strip()
        
        # Remove any markdown code block wrappers if present
        if markdown_content.startswith('```'):
            # Extract content from markdown code block
            lines = markdown_content.split('\n')
            content_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    content_lines.append(line)
            markdown_content = '\n'.join(content_lines).strip()
        
        # Return as a simple dict with markdown content
        return {"markdown": markdown_content}
    except Exception as e:
        # Don't wrap RuntimeError from provider - it already has good error messages
        if isinstance(e, RuntimeError) and ("Ollama" in str(e) or "OpenAI" in str(e) or "Anthropic" in str(e)):
            raise e
        raise RuntimeError(f"Error generating lesson: {e}")


def save_lesson(lesson_data: Dict, output_path: Path, format: str = "markdown", language_adapter=None):
    """Save lesson to file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lesson_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Lesson saved to {output_path}")
    elif format == "markdown":
        # LLM already returns markdown, just save it directly
        if "markdown" in lesson_data:
            md_content = lesson_data["markdown"]
        else:
            # Fallback: convert old JSON format to markdown (backward compatibility)
            md_content = format_lesson_markdown(lesson_data, language_adapter)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Lesson saved to {output_path}")
    else:
        raise ValueError(f"Unknown format: {format}")


def format_lesson_markdown(lesson_data: Dict, language_adapter=None) -> str:
    """Format lesson data as markdown"""
    # Determine language name for title
    if language_adapter:
        language_name = language_adapter.language_name
        proficiency_key = "jlpt_level" if language_adapter.language_code == "ja" else "cefr_level"
        proficiency_label = "JLPT Level" if language_adapter.language_code == "ja" else "CEFR Level"
    else:
        language_name = "Language"
        proficiency_key = "jlpt_level"  # Default to JLPT for backward compatibility
        proficiency_label = "JLPT Level"
    
    lines = [f"# {language_name} Lesson\n"]
    
    if "summary" in lesson_data and lesson_data["summary"]:
        lines.append(f"## Summary\n\n{lesson_data['summary']}\n\n")
    
    if "vocabulary" in lesson_data and lesson_data["vocabulary"]:
        lines.append("## Vocabulary\n\n")
        for word in lesson_data["vocabulary"]:
            # Handle different vocabulary structures
            word_text = word.get('word', 'N/A')
            reading = word.get('reading') or word.get('pronunciation', '')
            
            if reading:
                lines.append(f"### {word_text} ({reading})")
            else:
                lines.append(f"### {word_text}")
            
            lines.append(f"**Meaning:** {word.get('meaning', 'N/A')}")
            
            # Handle different proficiency level keys
            level = word.get('jlpt_level') or word.get('cefr_level') or word.get('proficiency_level', 'N/A')
            if level != 'N/A':
                lines.append(f"**{proficiency_label}:** {level}")
            
            if word.get('example_sentence'):
                lines.append(f"\n**Example:**\n- {word['example_sentence']}")
                if word.get('example_translation'):
                    lines.append(f"  → {word['example_translation']}")
            lines.append("")
    
    if "grammar_points" in lesson_data and lesson_data["grammar_points"]:
        lines.append("## Grammar Points\n\n")
        for grammar in lesson_data["grammar_points"]:
            lines.append(f"### {grammar.get('pattern', 'N/A')}")
            
            # Handle different proficiency level keys
            level = grammar.get('jlpt_level') or grammar.get('cefr_level') or grammar.get('proficiency_level', 'N/A')
            if level != 'N/A':
                lines.append(f"**{proficiency_label}:** {level}")
            
            lines.append(f"\n**Explanation:**\n{grammar.get('explanation', 'N/A')}")
            
            if grammar.get('example_sentence'):
                lines.append(f"\n**Example:**\n- {grammar['example_sentence']}")
                if grammar.get('example_translation'):
                    lines.append(f"  → {grammar['example_translation']}")
            lines.append("")
    
    if "key_phrases" in lesson_data and lesson_data["key_phrases"]:
        lines.append("## Key Phrases\n\n")
        for phrase in lesson_data["key_phrases"]:
            lines.append(f"- **{phrase.get('phrase', 'N/A')}**")
            if phrase.get('translation'):
                lines.append(f"  - Translation: {phrase['translation']}")
            if phrase.get('context'):
                lines.append(f"  - Context: {phrase['context']}")
            lines.append("")
    
    return "\n".join(lines)


def get_file_date_from_name(filename: str):
    """Extract date from filename (format: YYYY-MM-DD_...)"""
    try:
        date_str = filename.split('_')[0]
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, IndexError):
        return None


def find_transcription_files(audio_dir: Path, from_date=None, to_date=None):
    """Find transcription files in a directory
    
    Prefers clean transcript files (_transcript.txt) over formatted ones (.txt).
    If both exist, only processes the clean version.
    """
    transcription_files = []
    seen_basenames = set()
    
    # First pass: collect all .txt files
    all_txt_files = []
    for txt_file in audio_dir.glob("*.txt"):
        # Skip lesson files
        if "_lesson" in txt_file.stem:
            continue
        
        # Check date if filters provided
        if from_date or to_date:
            file_date = get_file_date_from_name(txt_file.stem)
            if file_date:
                if from_date and file_date < from_date:
                    continue
                if to_date and file_date > to_date:
                    continue
        
        all_txt_files.append(txt_file)
    
    # Second pass: prefer _transcript.txt files over regular .txt files
    # (local_whisper_transcribe.py creates both: formatted .txt and clean _transcript.txt)
    for txt_file in sorted(all_txt_files):
        base_name = txt_file.stem
        
        # If this is a _transcript.txt file, use it
        if base_name.endswith("_transcript"):
            clean_base = base_name.replace("_transcript", "")
            seen_basenames.add(clean_base)
            transcription_files.append(txt_file)
        # If this is a regular .txt file, check if _transcript.txt exists
        elif base_name not in seen_basenames:
            # Check if corresponding _transcript.txt exists
            transcript_version = txt_file.parent / f"{base_name}_transcript.txt"
            if not transcript_version.exists():
                # No clean version exists, use the formatted one
                transcription_files.append(txt_file)
                seen_basenames.add(base_name)
            # If _transcript.txt exists, skip this formatted version (will be picked up in first pass)
    
    return sorted(transcription_files)


def calculate_worker_count(requested_jobs: int, max_utilization: float) -> int:
    """Calculate optimal worker count based on CPU cores and utilization target"""
    cpu_count = multiprocessing.cpu_count()
    
    if requested_jobs == 0:
        # Auto-detect: use max_utilization of available cores
        workers = max(1, int(cpu_count * max_utilization))
    else:
        workers = requested_jobs
    
    # Cap at reasonable maximum (don't exceed CPU count)
    workers = min(workers, cpu_count)
    
    return max(1, workers)


def process_single_file_for_parallel(txt_file: Path, provider_type: str, provider_kwargs: dict,
                                     output_format: str, language_adapter=None, prompt_variant=None, prompt_files=None):
    """Process a single transcription file (for parallel execution)"""
    try:
        # Create a new provider instance for this thread (important for Ollama)
        provider = get_provider(provider_type, **provider_kwargs)
        
        # Load transcription
        transcription_text = load_transcription(txt_file)
        if not transcription_text:
            return (txt_file, False, "Empty transcription")
        
        # Get episode title from metadata if available
        episode_title = None
        # Try to load title from metadata JSON (would need config_data passed in, but keeping simple for now)
        
        # Generate lesson
        ext = "md" if output_format == "markdown" else output_format
        output_path = txt_file.parent / f"{txt_file.stem}_lesson.{ext}"
        lesson_data = generate_lesson(provider, transcription_text, 
                                     episode_title=episode_title,
                                     language_adapter=language_adapter,
                                     prompt_variant=prompt_variant,
                                     prompt_files=prompt_files)
        
        # Save lesson
        save_lesson(lesson_data, output_path, output_format, language_adapter)
        
        return (txt_file, True, None)
    except Exception as e:
        return (txt_file, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Generate language lessons from transcriptions (supports multiple languages via adapters)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('transcription_file', type=Path, nargs='?',
                       help='Path to transcription file (.txt or .json). Optional if --name is provided.')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output file path (default: same as transcription with _lesson.md suffix)')
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown',
                       help='Output format: markdown (default, user-friendly) or json (only if explicitly requested)')
    parser.add_argument('--provider', choices=['auto', 'ollama', 'openai', 'anthropic'], 
                       default='auto',
                       help='LLM provider to use (default: auto - tries local first)')
    parser.add_argument('--model', 
                       help='Model name (e.g., llama3.1, gpt-4o-mini, claude-3-5-sonnet-20241022)')
    parser.add_argument('--api-key',
                       help='API key (for OpenAI/Anthropic, or set via environment variables)')
    parser.add_argument('--title',
                       help='Episode title (optional, for context)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (use with --name)')
    parser.add_argument('--name',
                       help='Short name of the podcast from config file (loads LLM settings from config)')
    parser.add_argument('--language',
                       help='Language code (e.g., ja, en, es). Overrides config language setting.')
    parser.add_argument('--prompt-variant',
                       help='Prompt variant name (e.g., detailed, simple). Looks for system_prompt_{variant}.md files.')
    parser.add_argument('--from-date',
                       help='Start date filter (YYYY-MM-DD). Only process files from this date onwards.')
    parser.add_argument('--to-date',
                       help='End date filter (YYYY-MM-DD). Only process files up to this date.')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that already have lesson files')
    parser.add_argument('--simulate', action='store_true',
                       help='Show what would be processed without generating lessons')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1, use 0 for auto-detect based on CPU/GPU). Only works with Ollama provider.')
    parser.add_argument('--max-utilization', type=float, default=0.7,
                       help='Maximum CPU/GPU utilization target when using auto-detect (0.0-1.0, default: 0.7 = 70%%)')
    
    # If run with no arguments, show usage and help
    if len(sys.argv) == 1:
        parser.print_usage()
        print()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Always try to load defaults from config file
    config_provider = None
    config_model = None
    language_adapter = None
    config_data = None
    podcast = None
    
    try:
        config_data = load_config(args.config)
        analysis_cfg = config_data.get('analysis', {})
        config_provider = analysis_cfg.get('provider')
        config_model = analysis_cfg.get('model')
        config_ollama_url_raw = analysis_cfg.get('base_url') or analysis_cfg.get('ollama_url')
        if config_ollama_url_raw:
            config_ollama_url = expand_env_vars(config_ollama_url_raw)
        else:
            config_ollama_url = None
        
        # Load language adapter from config or command line
        prompt_variant = None
        if ADAPTERS_AVAILABLE:
            language_code = args.language or config_data.get('language', 'ja')  # Default to Japanese for backward compatibility
            language_adapter = get_language_adapter(language_code)
            if language_adapter:
                print(f"Using language adapter: {language_adapter.language_name} ({language_adapter.language_code})")
            else:
                print(f"Warning: Language adapter for '{language_code}' not found, using default")
            
            # Get prompt variant from CLI or config
            prompt_variant = args.prompt_variant or analysis_cfg.get('prompt_variant')
            if prompt_variant:
                print(f"Using prompt variant: {prompt_variant}")
            
            # Get prompt_files from config (takes precedence over variant)
            prompt_files = analysis_cfg.get('prompt_files')
            if prompt_files:
                print(f"Using prompt files from config:")
                if "system" in prompt_files:
                    print(f"  System: {prompt_files['system']}")
                if "user" in prompt_files:
                    print(f"  User: {prompt_files['user']}")
            else:
                prompt_files = None
        else:
            prompt_files = None
        
        # If --name was provided, validate it exists
        if args.name:
            podcast = find_podcast_by_name(config_data, args.name)
            print(f"Loaded analysis settings from config: provider={config_provider or DEFAULT_LLM_PROVIDER}, model={config_model or DEFAULT_LLM_MODEL}")
    except FileNotFoundError:
        # Config file doesn't exist, that's okay - use code defaults
        pass
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using command-line arguments or code defaults")
    
    # Determine transcription file(s) to process
    transcription_files = []
    
    if args.transcription_file:
        # Single file provided
        transcription_files = [args.transcription_file]
    elif args.name and config_data:
        # No file provided, but --name was given - find all transcriptions for this podcast
        data_root = Path(config_data.get("data_root", "."))
        channel_name = podcast["channel_name_short"]
        audio_dir = data_root / channel_name
        
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            print("Make sure you've downloaded and transcribed audio files first.")
            sys.exit(1)
        
        # Parse date filters
        from_date = None
        to_date = None
        if args.from_date:
            try:
                from_date = datetime.strptime(args.from_date, '%Y-%m-%d')
            except ValueError:
                print(f"Error: Invalid date format for --from-date. Use YYYY-MM-DD")
                sys.exit(1)
        if args.to_date:
            try:
                to_date = datetime.strptime(args.to_date, '%Y-%m-%d')
            except ValueError:
                print(f"Error: Invalid date format for --to-date. Use YYYY-MM-DD")
                sys.exit(1)
        
        transcription_files = find_transcription_files(audio_dir, from_date, to_date)
        
        if not transcription_files:
            print(f"Error: No transcription files found in {audio_dir}")
            if from_date or to_date:
                print(f"  (with date filter: from={from_date.date() if from_date else 'any'}, to={to_date.date() if to_date else 'any'})")
            print("Make sure you've transcribed audio files first.")
            sys.exit(1)
        
        print(f"Found {len(transcription_files)} transcription file(s) for '{args.name}'")
        if from_date or to_date:
            print(f"  Date filter: from={from_date.date() if from_date else 'any'}, to={to_date.date() if to_date else 'any'}")
    else:
        # Neither file nor --name provided
        print("Error: Either provide a transcription_file or use --name to process all transcriptions for a podcast")
        parser.print_usage()
        sys.exit(1)
    
    # Apply skip-existing filter if requested
    if args.skip_existing:
        filtered = []
        ext = "md" if args.format == "markdown" else args.format
        for txt_file in transcription_files:
            lesson_file = txt_file.parent / f"{txt_file.stem}_lesson.{ext}"
            if not lesson_file.exists():
                filtered.append(txt_file)
        transcription_files = filtered
        if len(filtered) < len(transcription_files) + len(filtered) - len(transcription_files):
            print(f"After filtering existing lessons: {len(transcription_files)} files to process")
    
    # Simulation mode
    if args.simulate:
        print("\n" + "="*80)
        print("SIMULATION MODE - No lessons will be generated")
        print("="*80)
        print(f"\nWould process {len(transcription_files)} file(s):")
        for f in transcription_files:
            ext = "md" if args.format == "markdown" else args.format
            output_path = f.parent / f"{f.stem}_lesson.{ext}"
            exists = "✓ exists" if output_path.exists() else "✗ missing"
            print(f"  {f.name} → {output_path.name} ({exists})")
        return
    
    if not transcription_files:
        print("No files to process")
        return
    
    # Get LLM provider settings
    provider_type = args.provider if args.provider != 'auto' else (config_provider or DEFAULT_LLM_PROVIDER)
    model = args.model or config_model or DEFAULT_LLM_MODEL
    
    # Determine if we should use parallel processing
    use_parallel = False
    worker_count = 1
    
    if len(transcription_files) > 1 and provider_type == 'ollama':
        # Only enable parallel processing for Ollama with multiple files
        worker_count = calculate_worker_count(args.jobs, args.max_utilization)
        use_parallel = (worker_count > 1)
        
        if use_parallel:
            print(f"\nUsing {worker_count} parallel workers (targeting {args.max_utilization*100:.0f}% utilization)")
        else:
            print("\nProcessing sequentially")
    elif len(transcription_files) > 1 and provider_type in ('openai', 'anthropic'):
        print("\nNote: Parallel processing disabled for cloud providers (using sequential processing)")
    
    # Prepare provider kwargs
    provider_kwargs = {}
    if model:
        provider_kwargs['model'] = model
    if args.api_key:
        provider_kwargs['api_key'] = args.api_key
    if provider_type == 'ollama':
        # Ollama URL must be configured in config file (no default fallback, like Whisper model_path)
        if not config_ollama_url:
            print("Error: Ollama base_url is not configured.")
            print("\nPlease add 'base_url' to the 'analysis' block in your config file:")
            print("  config/podcasts.json")
            print("\nExample:")
            print('  "analysis": {')
            print('    "provider": "ollama",')
            print('    "model": "qwen2.5:14b",')
            print('    "base_url": "${OLLAMA_BASE_URL}"  // or "http://localhost:11434"')
            print('  }')
            print("\nYou can use environment variables: ${OLLAMA_BASE_URL} or $OLLAMA_BASE_URL")
            sys.exit(1)
        provider_kwargs['base_url'] = config_ollama_url
    
    # Initialize provider and test connection (for sequential processing, we'll reuse this instance)
    provider = None
    if not use_parallel:
        # For sequential processing, initialize once and reuse
        try:
            print(f"Initializing {provider_type} provider with model {model}...")
            provider = get_provider(provider_type, **provider_kwargs)
            
            # Test connection and availability before processing
            if isinstance(provider, OllamaProvider):
                print("Testing Ollama connection...")
                if not provider.is_available():
                    print(f"\n{'='*80}")
                    print("ERROR: Cannot connect to Ollama or model is not available")
                    print(f"{'='*80}")
                    print(f"Base URL: {provider.base_url}")
                    print(f"Model: {provider.model}")
                    print("\nPlease check:")
                    print("  1. Ollama is running: ollama serve")
                    print("  2. The model is installed: ollama pull " + model)
                    print("  3. The base_url in config is correct")
                    sys.exit(1)
                print(f"✓ Connected to Ollama at {provider.base_url}")
                print(f"✓ Model '{provider.model}' is available")
            else:
                # For other providers, just check if they're available
                if not provider.is_available():
                    print(f"\n{'='*80}")
                    print(f"ERROR: {provider_type} provider is not available")
                    print(f"{'='*80}")
                    sys.exit(1)
                print(f"✓ {provider_type} provider is ready")
            
            print(f"✓ Provider initialized successfully\n")
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Failed to initialize provider")
            print(f"{'='*80}")
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # For parallel processing, test that we can initialize and connect
        try:
            print(f"Testing {provider_type} provider connection...")
            test_provider = get_provider(provider_type, **provider_kwargs)
            
            # Test connection for Ollama
            if isinstance(test_provider, OllamaProvider):
                if not test_provider.is_available():
                    print(f"\n{'='*80}")
                    print("ERROR: Cannot connect to Ollama or model is not available")
                    print(f"{'='*80}")
                    print(f"Base URL: {test_provider.base_url}")
                    print(f"Model: {test_provider.model}")
                    print("\nPlease check:")
                    print("  1. Ollama is running: ollama serve")
                    print("  2. The model is installed: ollama pull " + model)
                    print("  3. The base_url in config is correct")
                    sys.exit(1)
                print(f"✓ Connected to Ollama at {test_provider.base_url}")
                print(f"✓ Model '{test_provider.model}' is available")
            else:
                if not test_provider.is_available():
                    print(f"\n{'='*80}")
                    print(f"ERROR: {provider_type} provider is not available")
                    print(f"{'='*80}")
                    sys.exit(1)
                print(f"✓ {provider_type} provider is ready")
            
            del test_provider  # Clean up test instance
            print(f"✓ Connection test passed\n")
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Failed to initialize provider")
            print(f"{'='*80}")
            print(f"Error: {e}")
            sys.exit(1)
    
    # Process files (parallel or sequential)
    success = 0
    failed = 0
    
    if use_parallel:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_single_file_for_parallel,
                    txt_file,
                    provider_type,
                    provider_kwargs,
                    args.format,
                    language_adapter,
                    prompt_variant,
                    prompt_files
                ): txt_file for txt_file in transcription_files
            }
            
            for future in concurrent.futures.as_completed(futures):
                txt_file = futures[future]
                try:
                    result_file, success_flag, error_msg = future.result()
                    if success_flag:
                        print(f"✓ {result_file.name}")
                        success += 1
                    else:
                        # Check for timeout errors - exit immediately
                        if error_msg and ("timeout" in error_msg.lower() or "timed out" in error_msg.lower()):
                            print(f"\n{'='*80}")
                            print("FATAL ERROR: Request timed out")
                            print(f"{'='*80}")
                            print(f"File: {result_file.name}")
                            print(f"Error: {error_msg}")
                            print("\nThis usually means:")
                            print("  - The LLM request is taking too long (>5 minutes)")
                            print("  - The model is overloaded or stuck")
                            print("  - Network connectivity issues")
                            print("\nExiting to prevent further timeouts...")
                            sys.exit(1)
                        print(f"✗ {result_file.name}: {error_msg}")
                        failed += 1
                except Exception as e:
                    error_msg = str(e)
                    # Check for timeout errors - exit immediately
                    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        print(f"\n{'='*80}")
                        print("FATAL ERROR: Request timed out")
                        print(f"{'='*80}")
                        print(f"File: {txt_file.name}")
                        print(f"Error: {error_msg}")
                        print("\nThis usually means:")
                        print("  - The LLM request is taking too long (>5 minutes)")
                        print("  - The model is overloaded or stuck")
                        print("  - Network connectivity issues")
                        print("\nExiting to prevent further timeouts...")
                        sys.exit(1)
                    print(f"✗ {txt_file.name}: {e}")
                    failed += 1
    else:
        # Sequential processing
        for transcription_file in transcription_files:
            print(f"\n{'='*80}")
            print(f"Processing: {transcription_file.name}")
            print(f"{'='*80}")
            
            # Determine output path
            if args.output and len(transcription_files) == 1:
                # Single file with explicit output
                output_path = args.output
            else:
                # Default: same directory as transcription with _lesson suffix
                # Use .md extension for markdown format
                ext = "md" if args.format == "markdown" else args.format
                output_path = transcription_file.parent / f"{transcription_file.stem}_lesson.{ext}"
            
            # Skip if lesson already exists (unless output is explicitly specified)
            if output_path.exists() and not args.output:
                print(f"⏭️  Skipping (lesson already exists: {output_path})")
                continue
            
            # Load transcription
            print(f"Loading transcription from {transcription_file}...")
            try:
                transcription_text = load_transcription(transcription_file)
                if not transcription_text:
                    print(f"Warning: Transcription file is empty, skipping")
                    failed += 1
                    continue
                print(f"Loaded {len(transcription_text)} characters of text")
            except Exception as e:
                print(f"Error loading transcription: {e}")
                failed += 1
                continue
            
            # Provider is already initialized (reused for all files in sequential mode)
            
            # Get episode title from metadata if available
            episode_title = args.title
            if not episode_title and args.name and config_data:
                # Try to load title from metadata JSON
                try:
                    data_root = Path(config_data.get("data_root", "."))
                    channel_name = podcast["channel_name_short"]
                    metadata_file = data_root / f"{channel_name}.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        # Find matching entry by filename
                        for entry in metadata:
                            if entry.get('filename', '').replace('.mp3', '') in transcription_file.stem:
                                episode_title = entry.get('title', '')
                                break
                except Exception:
                    pass  # Ignore errors loading metadata
            
            # Generate lesson
            print("Generating lesson...")
            print("  (This may take several minutes for complex transcriptions)")
            try:
                lesson_data = generate_lesson(provider, transcription_text, 
                                             episode_title=episode_title,
                                             language_adapter=language_adapter,
                                             prompt_variant=prompt_variant,
                                             prompt_files=prompt_files)
            except Exception as e:
                error_msg = str(e)
                # Check for timeout errors - exit immediately
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    print(f"\n{'='*80}")
                    print("FATAL ERROR: Request timed out")
                    print(f"{'='*80}")
                    print(f"Error: {error_msg}")
                    print("\nThis usually means:")
                    print("  - The LLM request is taking too long (>10 minutes)")
                    print("  - The model is overloaded or stuck")
                    print("  - The transcription text may be too long")
                    print("  - Try processing a shorter transcription or splitting the text")
                    print("\nExiting to prevent further timeouts...")
                    sys.exit(1)
                print(f"Error generating lesson: {e}")
                failed += 1
                continue
            
            # Save lesson
            try:
                save_lesson(lesson_data, output_path, args.format, language_adapter)
                success += 1
            except Exception as e:
                print(f"Error saving lesson: {e}")
                failed += 1
                continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"  Success: {success}")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

