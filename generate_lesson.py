#!/usr/bin/env python3
"""
Generate JLPT-style lessons from Japanese transcriptions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional
from llm_providers import get_provider, LLMProvider, OllamaProvider, OpenAIProvider, AnthropicProvider
from llm_config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL

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


LESSON_SYSTEM_PROMPT = """You are an expert Japanese language teacher specializing in JLPT (Japanese Language Proficiency Test) preparation. 
Your task is to analyze Japanese text and create structured lessons with vocabulary and grammar explanations.

Always respond in valid JSON format with the following structure:
{
  "vocabulary": [
    {
      "word": "word in Kanji, Hiragana, or Katakana",
      "reading": "hiragana/katakana reading",
      "meaning": "English meaning",
      "jlpt_level": "N5|N4|N3|N2|N1",
      "example_sentence": "Example sentence using the word",
      "example_translation": "English translation of example"
    }
  ],
  "grammar_points": [
    {
      "pattern": "Grammar pattern name",
      "explanation": "Explanation of how to use this grammar",
      "jlpt_level": "N5|N4|N3|N2|N1",
      "example_sentence": "Example sentence",
      "example_translation": "English translation"
    }
  ],
  "key_phrases": [
    {
      "phrase": "Japanese phrase",
      "translation": "English translation",
      "context": "When/where this phrase is used"
    }
  ],
  "summary": "Brief summary of the lesson content"
}"""


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
                   episode_title: Optional[str] = None) -> Dict:
    """Generate a lesson from transcription text"""
    
    prompt = f"""Analyze the following Japanese text and create a comprehensive JLPT-style lesson.

{f"Episode Title: {episode_title}" if episode_title else ""}

Japanese Text:
{transcription_text}

Please extract:
1. Important vocabulary words with their readings, meanings, and JLPT levels
2. Grammar patterns and structures with explanations
3. Key phrases that are useful for learners
4. A brief summary of the content

Focus on words and grammar that would be useful for JLPT learners (N5-N1 levels)."""
    
    try:
        # Determine format based on provider type
        use_json_format = isinstance(provider, (OpenAIProvider, AnthropicProvider))
        if isinstance(provider, OllamaProvider):
            # Ollama supports JSON format directly
            use_json_format = True
        
        response = provider.generate(
            prompt=prompt,
            system_prompt=LESSON_SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for more consistent, educational output
            format="json_object" if use_json_format else None
        )
        
        # Try to parse JSON response
        # Sometimes LLMs wrap JSON in markdown code blocks
        response = response.strip()
        if response.startswith('```'):
            # Extract JSON from markdown code block
            lines = response.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            response = '\n'.join(json_lines)
        elif response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        
        lesson_data = json.loads(response)
        return lesson_data
    except json.JSONDecodeError as e:
        print(f"Error: LLM response was not valid JSON")
        print(f"Response: {response[:500]}...")
        raise RuntimeError(f"Failed to parse lesson JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Error generating lesson: {e}")


def save_lesson(lesson_data: Dict, output_path: Path, format: str = "json"):
    """Save lesson to file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lesson_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Lesson saved to {output_path}")
    elif format == "markdown":
        # Convert to markdown format
        md_content = format_lesson_markdown(lesson_data)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Lesson saved to {output_path}")
    else:
        raise ValueError(f"Unknown format: {format}")


def format_lesson_markdown(lesson_data: Dict) -> str:
    """Format lesson data as markdown"""
    lines = ["# Japanese Lesson\n"]
    
    if "summary" in lesson_data:
        lines.append(f"## Summary\n\n{lesson_data['summary']}\n\n")
    
    if "vocabulary" in lesson_data and lesson_data["vocabulary"]:
        lines.append("## Vocabulary\n\n")
        for word in lesson_data["vocabulary"]:
            lines.append(f"### {word.get('word', 'N/A')} ({word.get('reading', 'N/A')})")
            lines.append(f"**Meaning:** {word.get('meaning', 'N/A')}")
            lines.append(f"**JLPT Level:** {word.get('jlpt_level', 'N/A')}")
            if word.get('example_sentence'):
                lines.append(f"**Example:** {word['example_sentence']}")
                if word.get('example_translation'):
                    lines.append(f"  → {word['example_translation']}")
            lines.append("")
    
    if "grammar_points" in lesson_data and lesson_data["grammar_points"]:
        lines.append("## Grammar Points\n\n")
        for grammar in lesson_data["grammar_points"]:
            lines.append(f"### {grammar.get('pattern', 'N/A')}")
            lines.append(f"**JLPT Level:** {grammar.get('jlpt_level', 'N/A')}")
            lines.append(f"**Explanation:** {grammar.get('explanation', 'N/A')}")
            if grammar.get('example_sentence'):
                lines.append(f"**Example:** {grammar['example_sentence']}")
                if grammar.get('example_translation'):
                    lines.append(f"  → {grammar['example_translation']}")
            lines.append("")
    
    if "key_phrases" in lesson_data and lesson_data["key_phrases"]:
        lines.append("## Key Phrases\n\n")
        for phrase in lesson_data["key_phrases"]:
            lines.append(f"- **{phrase.get('phrase', 'N/A')}**")
            lines.append(f"  - Translation: {phrase.get('translation', 'N/A')}")
            if phrase.get('context'):
                lines.append(f"  - Context: {phrase['context']}")
            lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate JLPT-style lessons from Japanese transcriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('transcription_file', type=Path,
                       help='Path to transcription file (.txt or .json)')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output file path (default: same as transcription with _lesson.json suffix)')
    parser.add_argument('-f', '--format', choices=['json', 'markdown'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--provider', choices=['auto', 'ollama', 'openai', 'anthropic'], 
                       default='auto',
                       help='LLM provider to use (default: auto - tries local first)')
    parser.add_argument('--model', 
                       help='Model name (e.g., llama3.1, gpt-4o-mini, claude-3-5-sonnet-20241022)')
    parser.add_argument('--api-key',
                       help='API key (for OpenAI/Anthropic, or set via environment variables)')
    parser.add_argument('--ollama-url', default=DEFAULT_OLLAMA_URL,
                       help=f'Ollama base URL (default: {DEFAULT_OLLAMA_URL})')
    parser.add_argument('--title',
                       help='Episode title (optional, for context)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (use with --name)')
    parser.add_argument('--name',
                       help='Short name of the podcast from config file (loads LLM settings from config)')
    
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
    try:
        config_data = load_config(args.config)
        analysis_cfg = config_data.get('analysis', {})
        config_provider = analysis_cfg.get('provider')
        config_model = analysis_cfg.get('model')
        
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
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.transcription_file.parent / f"{args.transcription_file.stem}_lesson.{args.format}"
    
    # Load transcription
    print(f"Loading transcription from {args.transcription_file}...")
    try:
        transcription_text = load_transcription(args.transcription_file)
        if not transcription_text:
            print("Error: Transcription file is empty")
            sys.exit(1)
        print(f"Loaded {len(transcription_text)} characters of text")
    except Exception as e:
        print(f"Error loading transcription: {e}")
        sys.exit(1)
    
    # Get LLM provider (priority: command-line > config > code default)
    provider_type = args.provider if args.provider != 'auto' else (config_provider or DEFAULT_LLM_PROVIDER)
    model = args.model or config_model or DEFAULT_LLM_MODEL
    
    print(f"Initializing {provider_type} provider with model {model}...")
    provider_kwargs = {}
    if model:
        provider_kwargs['model'] = model
    if args.api_key:
        provider_kwargs['api_key'] = args.api_key
    if provider_type == 'ollama':
        provider_kwargs['base_url'] = args.ollama_url
    
    try:
        provider = get_provider(provider_type, **provider_kwargs)
    except Exception as e:
        print(f"Error initializing provider: {e}")
        sys.exit(1)
    
    # Generate lesson
    print("Generating lesson...")
    try:
        lesson_data = generate_lesson(provider, transcription_text, args.title)
    except Exception as e:
        print(f"Error generating lesson: {e}")
        sys.exit(1)
    
    # Save lesson
    try:
        save_lesson(lesson_data, output_path, args.format)
    except Exception as e:
        print(f"Error saving lesson: {e}")
        sys.exit(1)
    
    print("\n✓ Lesson generation complete!")


if __name__ == '__main__':
    main()

