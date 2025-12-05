#!/usr/bin/env python3
"""
Batch process transcriptions into lessons
Works with the existing transcription pipeline
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import concurrent.futures
import os
import multiprocessing
from generate_lesson import load_transcription, generate_lesson, save_lesson
from llm_providers import get_provider
from llm_config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL

# Import config loading functions
try:
    from find_podcasts import load_config, find_podcast_by_name
except ImportError:
    import json
    def load_config(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_podcast_by_name(config, name):
        for podcast in config["youtube_channels"]:
            if podcast["channel_name_short"] == name:
                return podcast
        raise ValueError(f"Podcast '{name}' not found")


def find_transcription_files(audio_dir: Path, from_date=None, to_date=None):
    """Find transcription files in a directory"""
    transcription_files = []
    
    # Look for .txt files (transcriptions)
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
        
        transcription_files.append(txt_file)
    
    return sorted(transcription_files)


def get_file_date_from_name(filename: str):
    """Extract date from filename (format: YYYY-MM-DD_...)"""
    try:
        date_str = filename.split('_')[0]
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, IndexError):
        return None


def calculate_worker_count(requested_jobs: int, max_utilization: float) -> int:
    """Calculate optimal worker count based on CPU cores and utilization target"""
    cpu_count = multiprocessing.cpu_count()
    
    if requested_jobs == 0:
        # Auto-detect: use 70% of available cores
        workers = max(1, int(cpu_count * max_utilization))
    else:
        workers = requested_jobs
    
    # Cap at reasonable maximum (don't exceed CPU count)
    workers = min(workers, cpu_count)
    
    return max(1, workers)


def process_single_file(txt_file: Path, provider_type: str, provider_kwargs: dict, 
                        output_format: str, lock=None):
    """Process a single transcription file (for parallel execution)"""
    try:
        # Create a new provider instance for this thread (important for Ollama)
        provider = get_provider(provider_type, **provider_kwargs)
        
        # Load transcription
        transcription_text = load_transcription(txt_file)
        if not transcription_text:
            return (txt_file, False, "Empty transcription")
        
        # Generate lesson
        output_path = txt_file.parent / f"{txt_file.stem}_lesson.{output_format}"
        lesson_data = generate_lesson(provider, transcription_text)
        
        # Save lesson
        save_lesson(lesson_data, output_path, output_format)
        
        return (txt_file, True, None)
    except Exception as e:
        return (txt_file, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Batch generate lessons from transcriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-a', '--audio-dir', type=Path,
                       help='Directory containing transcription files (required unless using --name)')
    parser.add_argument('--name',
                       help='Short name of the podcast from config file (loads LLM settings and audio-dir from config)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (use with --name)')
    parser.add_argument('-f', '--from-date',
                       help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('-t', '--to-date',
                       help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--provider', choices=['auto', 'ollama', 'openai', 'anthropic'],
                       default='auto',
                       help='LLM provider (default: auto, or from config if --name used)')
    parser.add_argument('--model',
                       help='Model name (overrides config if --name used)')
    parser.add_argument('--api-key',
                       help='API key for cloud providers')
    parser.add_argument('--ollama-url', default=DEFAULT_OLLAMA_URL,
                       help=f'Ollama base URL (default: {DEFAULT_OLLAMA_URL})')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that already have lesson files')
    parser.add_argument('--simulate', action='store_true',
                       help='Show what would be processed without generating')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1, use 0 for auto-detect based on CPU/GPU)')
    parser.add_argument('--max-utilization', type=float, default=0.7,
                       help='Maximum CPU/GPU utilization target (0.0-1.0, default: 0.7 = 70%%)')
    
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
    config_audio_dir = None
    try:
        config_data = load_config(args.config)
        analysis_cfg = config_data.get('analysis', {})
        config_provider = analysis_cfg.get('provider')
        config_model = analysis_cfg.get('model')
        
        # If --name was provided, get podcast-specific settings
        if args.name:
            podcast = find_podcast_by_name(config_data, args.name)
            # Get audio directory from config
            data_root = Path(config_data.get("data_root", "."))
            channel_name = podcast["channel_name_short"]
            config_audio_dir = data_root / channel_name
            print(f"Loaded settings from config:")
            print(f"  Provider: {config_provider or DEFAULT_LLM_PROVIDER}, Model: {config_model or DEFAULT_LLM_MODEL}")
            print(f"  Audio directory: {config_audio_dir}")
    except FileNotFoundError:
        # Config file doesn't exist, that's okay - use code defaults
        if args.name:
            print(f"Error: Config file not found: {args.config}")
            print("Cannot use --name without config file")
            sys.exit(1)
    except Exception as e:
        if args.name:
            print(f"Error loading config: {e}")
            sys.exit(1)
        # If --name not used, just warn and continue
        print(f"Warning: Could not load config: {e}")
        print("Using command-line arguments or code defaults")
    
    # Determine audio directory
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
    elif config_audio_dir:
        audio_dir = config_audio_dir
    else:
        parser.error("Either --audio-dir or --name must be provided")
    
    # Parse dates
    from_date = None
    to_date = None
    if args.from_date:
        from_date = datetime.strptime(args.from_date, '%Y-%m-%d')
    if args.to_date:
        to_date = datetime.strptime(args.to_date, '%Y-%m-%d')
    
    # Find transcription files
    if not audio_dir.exists():
        print(f"Error: Directory not found: {audio_dir}")
        sys.exit(1)
    
    print(f"Scanning {audio_dir} for transcription files...")
    transcription_files = find_transcription_files(audio_dir, from_date, to_date)
    
    if not transcription_files:
        print("No transcription files found")
        sys.exit(0)
    
    print(f"Found {len(transcription_files)} transcription files")
    
    if args.skip_existing:
        # Filter out files that already have lessons
        filtered = []
        for txt_file in transcription_files:
            lesson_file = txt_file.parent / f"{txt_file.stem}_lesson.{args.format}"
            if not lesson_file.exists():
                filtered.append(txt_file)
        transcription_files = filtered
        print(f"After filtering existing lessons: {len(transcription_files)} files to process")
    
    if args.simulate:
        print("\nWould process the following files:")
        for f in transcription_files:
            print(f"  {f.name}")
        return
    
    if not transcription_files:
        print("No files to process")
        return
    
    # Initialize provider (priority: command-line > config > code default)
    provider_type = args.provider if args.provider != 'auto' else (config_provider or DEFAULT_LLM_PROVIDER)
    model = args.model or config_model or DEFAULT_LLM_MODEL
    
    # Determine actual provider type (resolve 'auto' if needed)
    actual_provider = provider_type
    if provider_type == 'auto':
        # Try to determine from config, otherwise it will be resolved by get_provider
        actual_provider = config_provider or DEFAULT_LLM_PROVIDER
    
    print(f"\nInitializing {provider_type} provider with model {model}...")
    provider_kwargs = {}
    if model:
        provider_kwargs['model'] = model
    if args.api_key:
        provider_kwargs['api_key'] = args.api_key
    if actual_provider == 'ollama':
        provider_kwargs['base_url'] = args.ollama_url
    
    # Only enable parallelization for Ollama
    use_parallel = (actual_provider == 'ollama')
    
    # Calculate worker count (only if using Ollama)
    if use_parallel:
        worker_count = calculate_worker_count(args.jobs, args.max_utilization)
        if worker_count > 1:
            print(f"Using {worker_count} parallel workers for Ollama (targeting {args.max_utilization*100:.0f}% utilization)")
        else:
            print("Processing sequentially")
    else:
        worker_count = 1
        if actual_provider in ('openai', 'anthropic'):
            print("Note: Parallel processing disabled for cloud providers (using sequential processing)")
    
    # Test provider initialization
    try:
        test_provider = get_provider(provider_type, **provider_kwargs)
        del test_provider  # Clean up test instance
    except Exception as e:
        print(f"Error initializing provider: {e}")
        sys.exit(1)
    
    # Process files
    success = 0
    failed = 0
    
    if use_parallel and worker_count > 1:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_single_file,
                    txt_file,
                    provider_type,
                    provider_kwargs,
                    args.format
                ): txt_file for txt_file in transcription_files
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                txt_file = futures[future]
                try:
                    result_file, success_flag, error_msg = future.result()
                    if success_flag:
                        print(f"✓ {result_file.name}")
                        success += 1
                    else:
                        print(f"✗ {result_file.name}: {error_msg}")
                        failed += 1
                except Exception as e:
                    print(f"✗ {txt_file.name}: {e}")
                    failed += 1
    else:
        # Sequential processing (original behavior)
        for txt_file in transcription_files:
            print(f"\n{'='*60}")
            print(f"Processing: {txt_file.name}")
            print(f"{'='*60}")
            
            result_file, success_flag, error_msg = process_single_file(
                txt_file, provider_type, provider_kwargs, args.format
            )
            
            if success_flag:
                success += 1
            else:
                print(f"  ✗ Error: {error_msg}")
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

