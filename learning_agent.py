#!/usr/bin/env python3
"""
Learning Agent - Orchestrates the podpilot pipeline to create learning packs
Uses LLM to understand user queries and execute the appropriate tools
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

from lesson_search import (
    get_all_lessons, get_lessons_for_channel,
    search_lessons_by_keywords, search_lessons_by_jlpt_level,
    LessonContent, parse_lesson_markdown, find_lesson_files
)
from llm_providers import get_provider, LLMProvider
from llm_config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
from channel_fetcher import load_config, find_podcast_by_name, fetch_rss_feed, parse_rss_feed
from download_audio import load_json, download_file, process_existing_file
from local_whisper_transcribe import transcribe_audio, find_whisper_executable, check_existing_transcription
from generate_lesson import (
    generate_lesson, load_transcription, find_transcription_files,
    get_file_date_from_name, expand_env_vars, save_lesson
)
from adapters import get_language_adapter


class ToolExecutor:
    """Executes podpilot tools programmatically"""
    
    def __init__(self, config_data: Dict, provider: LLMProvider):
        self.config_data = config_data
        self.provider = provider
        self.data_root = Path(config_data.get("data_root", "."))
        self.analysis_cfg = config_data.get('analysis', {})
        self.transcription_cfg = config_data.get('transcription', {})
        
    def find_episodes(self, channel_name: str, from_date: Optional[datetime] = None,
                     to_date: Optional[datetime] = None, keywords: Optional[List[str]] = None) -> List[Dict]:
        """Find episodes matching criteria"""
        try:
            podcast = find_podcast_by_name(self.config_data, channel_name)
            channel_id = podcast["channel_id"]
            
            # Fetch RSS feed
            xml_content = fetch_rss_feed(channel_id)
            
            # Parse RSS feed
            from_date_obj = from_date.date() if from_date else None
            to_date_obj = to_date.date() if to_date else None
            
            # Debug output
            if from_date_obj or to_date_obj:
                print(f"Debug find_episodes: Filtering by date range: {from_date_obj} to {to_date_obj}")
            
            channel_info = parse_rss_feed(
                xml_content,
                from_date=from_date_obj,
                to_date=to_date_obj,
                include_description=True
            )
            
            episodes = channel_info.get('videos', [])
            
            # Filter by keywords if provided
            if keywords:
                keywords_lower = [kw.lower() for kw in keywords]
                filtered = []
                for episode in episodes:
                    title = episode.get('title', '').lower()
                    desc = episode.get('description', {}).get('text', '').lower() if isinstance(episode.get('description'), dict) else ''
                    if any(kw in title or kw in desc for kw in keywords_lower):
                        filtered.append(episode)
                episodes = filtered
            
            return episodes
        except Exception as e:
            print(f"Error finding episodes: {e}")
            return []
    
    def download_audio(self, channel_name: str, episodes: List[Dict],
                      force: bool = False) -> List[Path]:
        """Download audio files for episodes"""
        downloaded = []
        audio_dir = self.data_root / channel_name
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        for episode in episodes:
            url = episode.get('link')
            clean_filename = episode.get('clean_filename', '')
            video_id = episode.get('id')
            if not url or not clean_filename:
                continue
            
            output_path = audio_dir / f"{clean_filename}.mp3"
            
            # Check if file exists (exact match)
            if output_path.exists() and not force:
                print(f"⏭️  Skipping {output_path.name} (already exists)")
                downloaded.append(output_path)
                continue
            
            # Also check by video ID if available (handles filename variations)
            if video_id and not force:
                # Look for any file containing the video ID
                matching_files = list(audio_dir.glob(f"*{video_id}*.mp3"))
                if matching_files:
                    # Use the first matching file
                    existing_file = matching_files[0]
                    print(f"⏭️  Skipping (found existing file with same video ID: {existing_file.name})")
                    downloaded.append(existing_file)
                    continue
            
            try:
                print(f"Downloading: {episode.get('title', 'Unknown')}")
                success = download_file(url, output_path, simulate=False)
                if success:
                    downloaded.append(output_path)
                    print(f"✓ Downloaded: {output_path.name}")
                else:
                    print(f"✗ Failed to download: {episode.get('title', 'Unknown')}")
            except Exception as e:
                print(f"✗ Error downloading {episode.get('title', 'Unknown')}: {e}")
        
        return downloaded
    
    def transcribe_audio(self, audio_files: List[Path], force: bool = False) -> List[Path]:
        """Transcribe audio files"""
        transcriptions = []
        
        # Get transcription config
        provider_type = self.transcription_cfg.get('provider', 'whisper.cpp')
        
        if provider_type == 'whisper.cpp':
            # Use local whisper
            model_config = {
                'executable': find_whisper_executable(),
                'model_path': expand_env_vars(self.transcription_cfg.get('model_path', ''))
            }
            
            for audio_file in audio_files:
                # Check if already transcribed
                if not force and check_existing_transcription(audio_file):
                    print(f"⏭️  Skipping transcription for {audio_file.name} (already exists)")
                    transcriptions.append(audio_file.with_suffix('_transcript.txt'))
                    continue
                
                try:
                    print(f"Transcribing: {audio_file.name}")
                    success = transcribe_audio(audio_file, model_config, language="ja")
                    if success:
                        transcript_path = audio_file.parent / f"{audio_file.stem}_transcript.txt"
                        if transcript_path.exists():
                            transcriptions.append(transcript_path)
                            print(f"✓ Transcribed: {audio_file.name}")
                        else:
                            # Fallback to .txt
                            txt_path = audio_file.with_suffix('.txt')
                            if txt_path.exists():
                                transcriptions.append(txt_path)
                    else:
                        print(f"✗ Failed to transcribe: {audio_file.name}")
                except Exception as e:
                    print(f"✗ Error transcribing {audio_file.name}: {e}")
        else:
            print(f"Transcription provider '{provider_type}' not yet supported in agent mode")
            print("Please use transcribe.py or local_whisper_transcribe.py directly")
        
        return transcriptions
    
    def generate_lessons(self, transcription_files: List[Path], force: bool = False) -> List[Path]:
        """Generate lessons from transcription files"""
        lessons = []
        
        # Get LLM provider for lesson generation
        provider_type = self.analysis_cfg.get('provider', DEFAULT_LLM_PROVIDER)
        model = self.analysis_cfg.get('model', DEFAULT_LLM_MODEL)
        prompt_variant = self.analysis_cfg.get('prompt_variant')
        prompt_files = self.analysis_cfg.get('prompt_files')
        
        provider_kwargs = {'model': model}
        if provider_type == 'ollama':
            base_url_raw = self.analysis_cfg.get('base_url') or self.analysis_cfg.get('ollama_url')
            if base_url_raw:
                provider_kwargs['base_url'] = expand_env_vars(base_url_raw)
        
        try:
            from llm_providers import get_provider
            lesson_provider = get_provider(provider_type, **provider_kwargs)
            language_adapter = get_language_adapter("ja")
        except Exception as e:
            print(f"Error initializing lesson generation: {e}")
            return []
        
        for transcript_file in transcription_files:
            lesson_path = transcript_file.parent / f"{transcript_file.stem.replace('_transcript', '')}_lesson.md"
            
            # Skip if already exists and not forcing
            if lesson_path.exists() and not force:
                print(f"⏭️  Skipping lesson generation for {transcript_file.name} (already exists)")
                lessons.append(lesson_path)
                continue
            
            try:
                print(f"Generating lesson from: {transcript_file.name}")
                transcription_text = load_transcription(transcript_file)
                if not transcription_text:
                    print(f"Warning: Empty transcription, skipping")
                    continue
                
                lesson_data = generate_lesson(
                    lesson_provider,
                    transcription_text,
                    episode_title=None,
                    language_adapter=language_adapter,
                    prompt_variant=prompt_variant,
                    prompt_files=prompt_files
                )
                
                # Save lesson
                save_lesson(lesson_data, lesson_path, "markdown", language_adapter)
                lessons.append(lesson_path)
                print(f"✓ Generated lesson: {lesson_path.name}")
            except Exception as e:
                print(f"✗ Error generating lesson from {transcript_file.name}: {e}")
        
        return lessons


def create_plan_prompt(user_query: str, available_channels: List[str]) -> str:
    """Create a prompt for the LLM to plan which tools to use"""
    
    prompt = f"""You are a Japanese language learning assistant that can orchestrate a podcast learning pipeline.

The user's request is:
"{user_query}"

Available tools:
1. find_episodes(channel_name, from_date, to_date, keywords) - Find podcast episodes
2. download_audio(channel_name, episodes) - Download audio files
3. transcribe_audio(audio_files) - Transcribe audio to text
4. generate_lessons(transcription_files) - Generate lessons from transcriptions
5. create_learning_pack(query, lessons) - Create a learning pack

Available channels: {', '.join(available_channels)}

Based on the user's query, determine what actions need to be taken. Consider:
- Does the user need new episodes? (check if they mention specific topics, dates, or channels)
- Do audio files need to be downloaded?
- Do audio files need to be transcribed?
- Do transcriptions need lessons generated?
- What should the final learning pack focus on?

Respond with a JSON plan:
{{
  "steps": [
    {{
      "action": "find_episodes",
      "channel": "channel_name",
      "from_date": "YYYY-MM-DD or null",
      "to_date": "YYYY-MM-DD or null",
      "keywords": ["keyword1", "keyword2"]
    }},
    {{
      "action": "download_audio",
      "channel": "channel_name",
      "episodes": "use_episodes_from_previous_step"
    }},
    {{
      "action": "transcribe_audio",
      "audio_files": "use_from_previous_step"
    }},
    {{
      "action": "generate_lessons",
      "transcription_files": "use_from_previous_step"
    }},
    {{
      "action": "create_learning_pack",
      "query": "refined query for learning pack",
      "lessons": "use_from_previous_step"
    }}
  ],
  "reasoning": "Brief explanation of the plan"
}}

Only include steps that are necessary. If episodes/audio/transcriptions/lessons already exist, you can skip those steps."""
    
    return prompt


def create_learning_pack_prompt(user_query: str, selected_lessons: List[LessonContent]) -> str:
    """Create a prompt for generating the learning pack"""
    
    # Combine content from selected lessons
    lesson_contents = []
    for i, lesson in enumerate(selected_lessons):
        content = f"""## Lesson {i+1}: {lesson.episode_title}
{f"Date: {lesson.date.strftime('%Y-%m-%d')}" if lesson.date else ""}

### Summary
{lesson.summary if lesson.summary else "No summary available"}

### Vocabulary
"""
        for vocab in lesson.vocabulary[:10]:  # Limit to top 10 per lesson
            word = vocab.get('word', '')
            reading = vocab.get('reading', '')
            level = vocab.get('jlpt_level', '')
            content += f"- **{word}**"
            if reading:
                content += f" ({reading})"
            if level:
                content += f" - JLPT {level}"
            content += "\n"
        
        content += "\n### Grammar Points\n"
        for grammar in lesson.grammar_points[:5]:  # Limit to top 5 per lesson
            pattern = grammar.get('pattern', '')
            level = grammar.get('jlpt_level', '')
            content += f"- **{pattern}**"
            if level:
                content += f" - JLPT {level}"
            content += "\n"
        
        if lesson.key_phrases:
            content += "\n### Key Phrases\n"
            for phrase in lesson.key_phrases[:5]:
                content += f"- **{phrase.get('phrase', '')}**\n"
        
        lesson_contents.append(content)
    
    prompt = f"""You are an expert Japanese language teacher creating a personalized learning pack.

The user's learning goal/question is:
"{user_query}"

I have selected {len(selected_lessons)} relevant lessons. Here is their content:

{chr(10).join(['='*80 + lesson_content + '='*80 for lesson_content in lesson_contents])}

Create a comprehensive learning pack that addresses the user's query. The pack should include:

1. **Introduction**: Brief overview of what will be learned and why it's relevant
2. **Learning Objectives**: Clear goals for this pack
3. **Vocabulary Focus**: Curated vocabulary list organized by JLPT level or topic
4. **Grammar Focus**: Key grammar points with explanations
5. **Practice Exercises**: Create exercises based on the content (fill-in-the-blank, multiple choice, etc.)
6. **Review Section**: Summary of key points to remember
7. **Next Steps**: Suggestions for further study

Format the output as clean Markdown. Make it engaging and educational, suitable for self-study."""
    
    return prompt


def generate_learning_pack(provider: LLMProvider, user_query: str, 
                          selected_lessons: List[LessonContent]) -> str:
    """Generate a learning pack using the LLM"""
    
    system_prompt = """You are an expert Japanese language teacher specializing in creating personalized learning materials. 
You create comprehensive, well-structured learning packs that help students achieve their specific learning goals.
Always respond in Markdown format with clear sections and formatting."""
    
    user_prompt = create_learning_pack_prompt(user_query, selected_lessons)
    
    try:
        response = provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=None
        )
        return response.strip()
    except Exception as e:
        raise RuntimeError(f"Error generating learning pack: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Learning Agent - Orchestrates podpilot tools to create learning packs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('query', type=str,
                       help='Your learning question or goal (e.g., "I want to learn business Japanese from recent episodes")')
    parser.add_argument('--name', type=str,
                       help='Podcast channel name (optional, searches all channels if not specified)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to config file')
    parser.add_argument('--output', type=Path,
                       help='Output file path (default: learning_pack_YYYYMMDD_HHMMSS.md)')
    parser.add_argument('--provider', choices=['auto', 'ollama', 'openai', 'anthropic'],
                       default='auto',
                       help='LLM provider to use')
    parser.add_argument('--model',
                       help='Model name (overrides config)')
    parser.add_argument('--api-key',
                       help='API key (for OpenAI/Anthropic)')
    parser.add_argument('--from-date',
                       help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--to-date',
                       help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download/re-transcribe/re-generate even if files exist')
    parser.add_argument('--skip-tools', action='store_true',
                       help='Skip tool execution, only search existing lessons and create pack')
    
    args = parser.parse_args()
    
    # Load config
    try:
        config_data = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Get available channels
    available_channels = [ch['channel_name_short'] for ch in config_data.get('youtube_channels', [])]
    
    # Initialize LLM provider
    analysis_cfg = config_data.get('analysis', {})
    provider_type = args.provider if args.provider != 'auto' else (analysis_cfg.get('provider') or DEFAULT_LLM_PROVIDER)
    model = args.model or analysis_cfg.get('model') or DEFAULT_LLM_MODEL
    
    provider_kwargs = {}
    if model:
        provider_kwargs['model'] = model
    if args.api_key:
        provider_kwargs['api_key'] = args.api_key
    if provider_type == 'ollama':
        base_url_raw = analysis_cfg.get('base_url') or analysis_cfg.get('ollama_url')
        if base_url_raw:
            provider_kwargs['base_url'] = expand_env_vars(base_url_raw)
        else:
            print("Error: Ollama base_url is not configured in config file")
            sys.exit(1)
    
    try:
        print(f"Initializing {provider_type} provider...")
        provider = get_provider(provider_type, **provider_kwargs)
        
        if not provider.is_available():
            print(f"Error: {provider_type} provider is not available")
            sys.exit(1)
        print(f"✓ Provider ready\n")
    except Exception as e:
        print(f"Error initializing provider: {e}")
        sys.exit(1)
    
    # Create tool executor
    executor = ToolExecutor(config_data, provider)
    
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
    
    # Plan execution
    if not args.skip_tools:
        print("Planning execution steps...")
        system_prompt = """You are a Japanese language learning assistant that plans tool execution.
Respond ONLY with valid JSON. Do not include any explanatory text outside the JSON."""
        
        user_prompt = create_plan_prompt(args.query, available_channels)
        
        try:
            response = provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
                print(f"Plan: {plan.get('reasoning', 'No reasoning provided')}\n")
                
                # Execute plan
                episodes = []
                audio_files = []
                transcription_files = []
                lesson_files = []
                
                for step in plan.get('steps', []):
                    action = step.get('action')
                    
                    if action == 'find_episodes':
                        channel = step.get('channel') or args.name or available_channels[0]
                        step_from_date = datetime.strptime(step['from_date'], '%Y-%m-%d') if step.get('from_date') else from_date
                        step_to_date = datetime.strptime(step['to_date'], '%Y-%m-%d') if step.get('to_date') else to_date
                        keywords = step.get('keywords', [])
                        
                        print(f"Finding episodes in channel '{channel}'...")
                        episodes = executor.find_episodes(channel, step_from_date, step_to_date, keywords)
                        print(f"Found {len(episodes)} episodes\n")
                    
                    elif action == 'download_audio':
                        channel = step.get('channel') or args.name or available_channels[0]
                        print(f"Downloading audio files...")
                        audio_files = executor.download_audio(channel, episodes, force=args.force)
                        print(f"Downloaded {len(audio_files)} audio files\n")
                    
                    elif action == 'transcribe_audio':
                        print(f"Transcribing audio files...")
                        transcription_files = executor.transcribe_audio(audio_files, force=args.force)
                        print(f"Transcribed {len(transcription_files)} files\n")
                    
                    elif action == 'generate_lessons':
                        print(f"Generating lessons...")
                        lesson_files = executor.generate_lessons(transcription_files, force=args.force)
                        print(f"Generated {len(lesson_files)} lessons\n")
            else:
                print("Warning: Could not parse plan from LLM, using existing lessons only")
                args.skip_tools = True
        except Exception as e:
            print(f"Warning: Error planning execution: {e}")
            print("Falling back to searching existing lessons only")
            args.skip_tools = True
    
    # Load lessons (either from execution or existing files)
    print("Loading lessons...")
    if args.skip_tools or 'lesson_files' not in locals():
        # Search existing lessons
        if args.name:
            lessons = get_lessons_for_channel(executor.data_root, args.name, from_date, to_date)
        else:
            lessons = get_all_lessons(executor.data_root, from_date, to_date)
    else:
        # Parse lesson files from execution
        lessons = []
        for lesson_file in lesson_files:
            lesson = parse_lesson_markdown(lesson_file)
            if lesson:
                lessons.append(lesson)
    
    if not lessons:
        print("Error: No lessons found. Make sure you have generated lesson files first.")
        print("Or remove --skip-tools to let the agent fetch/download/transcribe/generate automatically.")
        sys.exit(1)
    
    print(f"Found {len(lessons)} lesson(s)")
    
    # Select relevant lessons using LLM
    print(f"\nSelecting relevant lessons for: {args.query}")
    # Simple keyword-based selection for now (can be enhanced with LLM)
    keywords = args.query.lower().split()
    selected_lessons = search_lessons_by_keywords(lessons, keywords)[:10]
    
    if not selected_lessons:
        selected_lessons = lessons[:5]  # Fallback to first 5
    
    print(f"Selected {len(selected_lessons)} relevant lesson(s)")
    
    # Generate learning pack
    print(f"\nGenerating learning pack...")
    try:
        learning_pack = generate_learning_pack(provider, args.query, selected_lessons)
    except Exception as e:
        print(f"Error generating learning pack: {e}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"learning_pack_{timestamp}.md")
    
    # Save learning pack
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Learning Pack\n\n")
        f.write(f"**Query:** {args.query}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Based on {len(selected_lessons)} lesson(s)**\n\n")
        f.write("---\n\n")
        f.write(learning_pack)
    
    print(f"\n✓ Learning pack saved to: {output_path}")
    print(f"\nPack includes content from {len(selected_lessons)} lesson(s)")


if __name__ == '__main__':
    main()
