# Test Coverage Improvement Guide

Current overall coverage: **64%**

## Priority Areas for Improvement

### 1. `generate_lesson.py` (23% coverage) - **HIGHEST PRIORITY**

**Missing Coverage:**
- Lines 434-901: `main()` function (CLI interface) - **~470 lines untested**
- Lines 255-322: `format_lesson_markdown()` function (deprecated but still present)
- Lines 325-383: `get_file_date_from_name()` and `find_transcription_files()` edge cases
- Lines 384-398: `calculate_worker_count()` edge cases
- Lines 400-431: `process_single_file_for_parallel()` error handling
- Error handling paths in `generate_lesson()` (lines 222-226)

**Recommended Tests:**

#### CLI Interface Tests (`main()` function)
```python
# tests/test_generate_lesson_cli.py
- Test single file processing with --name
- Test batch processing with --from-date and --to-date
- Test --skip-existing flag
- Test --simulate flag
- Test parallel processing (-j flag)
- Test --max-utilization flag
- Test timeout error handling (should exit immediately)
- Test Ollama connection test before batch processing
- Test prompt_variant handling
- Test prompt_files from config
- Test error handling for missing files
- Test output format selection (markdown vs json)
```

#### Batch Processing Tests
```python
# tests/test_generate_lesson_batch.py
- Test find_transcription_files() with date filters
- Test find_transcription_files() preferring _transcript.txt over .txt
- Test get_file_date_from_name() with various formats
- Test calculate_worker_count() with different inputs
- Test process_single_file_for_parallel() success and error cases
- Test parallel execution with ThreadPoolExecutor
```

#### Error Handling Tests
```python
# tests/test_generate_lesson_errors.py
- Test timeout detection and immediate exit
- Test provider initialization errors
- Test transcription loading errors
- Test lesson generation errors
- Test file saving errors
```

### 2. `adapters/base.py` (50% coverage)

**Missing Coverage:**
- Lines 57-59, 64, 71: Prompt file loading from config
- Lines 94-96, 101, 108: User prompt template loading
- Lines 179-183, 194-212: Schema generation
- Line 243: Environment variable expansion in `_resolve_prompt_path()`

**Recommended Tests:**
```python
# tests/test_adapters_base.py
- Test prompt file loading from config (prompt_files parameter)
- Test prompt file loading with environment variable expansion
- Test fallback to variant-specific files
- Test fallback to default files
- Test simple fallback prompt when no files found
- Test schema generation for different languages
```

### 3. `adapters/japanese.py` (51% coverage)

**Missing Coverage:**
- Lines 47-48: Adapter initialization
- Lines 97-161: Prompt loading methods
- Lines 164, 168-169: Schema methods

**Recommended Tests:**
```python
# tests/test_adapters_japanese.py
- Test JapaneseAdapter initialization
- Test get_lesson_system_prompt() with variants
- Test get_lesson_user_prompt_template() with variants
- Test prompt file loading
- Test schema injection into prompts
- Test segment_text() with various Japanese text
- Test clean_title() with Japanese characters
```

### 4. `llm_providers.py` (65% coverage)

**Missing Coverage:**
- Lines 50-62: Provider registry
- Lines 72, 75, 88, 90, 93: Error handling
- Lines 107, 121, 125: Provider availability checks
- Lines 138-139, 161, 165, 175, 178: Error paths
- Lines 183-184, 193-197, 201: OpenAI/Anthropic error handling
- Lines 207-226: Anthropic provider
- Lines 244-246, 280, 288-292: Provider registry and custom providers

**Recommended Tests:**
```python
# tests/test_llm_providers_extended.py
- Test provider registry (register_provider, get_provider)
- Test custom provider registration
- Test Ollama error handling (connection errors, model not found)
- Test OpenAI error handling (API errors, rate limits)
- Test Anthropic error handling
- Test is_available() for all providers with various error conditions
- Test timeout handling
```

### 5. `extract_duration.py` (54% coverage)

**Missing Coverage:**
- Lines 17-26: Import error handling
- Lines 112, 118: Duration update functions
- Lines 163-164, 182-199: Metadata update functions
- Lines 212-215, 218, 231, 235-240: Error handling
- Lines 257, 263-354: Main function and CLI

**Recommended Tests:**
```python
# tests/test_extract_duration_extended.py
- Test update_video_duration() with various inputs
- Test update_video_duration_from_file() error cases
- Test extract_metadata_duration() with missing files
- Test CLI interface
- Test filtering logic for short videos
```

### 6. `download_audio.py` (71% coverage)

**Missing Coverage:**
- Lines 39-40, 51, 79, 82-83, 92, 99: Error handling
- Lines 126-129, 155-157, 177-178, 186: Download paths
- Lines 203-209, 213-220, 228: YouTube download logic
- Lines 259-271, 323-341: Duration checking
- Lines 350, 356-357, 364-371, 374-389: File processing
- Lines 403-405, 419-420, 436, 443-444: Error paths
- Lines 456-458, 468-470, 499-500, 512: Edge cases
- Lines 544-548, 557, 584-590, 597-599, 608: Main function paths
- Lines 621-630, 633-638, 685-687, 725-727: CLI handling

**Recommended Tests:**
```python
# tests/test_download_audio_extended.py
- Test YouTube download with various error conditions
- Test duration checking before download
- Test min_duration filtering
- Test existing file handling
- Test metadata updates
- Test CLI interface with various arguments
- Test resume functionality (if implemented)
```

## Testing Best Practices

### 1. Use Fixtures for Common Setup
```python
@pytest.fixture
def mock_provider():
    provider = Mock(spec=LLMProvider)
    provider.generate.return_value = "Test markdown content"
    return provider

@pytest.fixture
def tmp_transcription_file(tmp_path):
    file = tmp_path / "test_transcript.txt"
    file.write_text("Test transcription content")
    return file
```

### 2. Test Error Paths
```python
def test_generate_lesson_provider_error():
    """Test error handling when provider fails"""
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.generate.side_effect = RuntimeError("Provider error")
    
    with pytest.raises(RuntimeError):
        generate_lesson(mock_provider, "test text")
```

### 3. Test Edge Cases
```python
def test_find_transcription_files_empty_directory(tmp_path):
    """Test finding files in empty directory"""
    result = find_transcription_files(tmp_path)
    assert result == []

def test_calculate_worker_count_zero_jobs():
    """Test worker count calculation with 0 jobs (auto-detect)"""
    result = calculate_worker_count(0, 0.8)
    assert result >= 1
```

### 4. Use Mocks for External Dependencies
```python
@patch('generate_lesson.get_provider')
@patch('generate_lesson.load_transcription')
def test_process_single_file_success(mock_load, mock_get_provider):
    """Test successful file processing"""
    # Setup mocks
    # Execute
    # Assert
```

### 5. Test CLI Arguments
```python
def test_main_single_file_mode(tmp_path, capsys):
    """Test main() with single file mode"""
    with patch('sys.argv', ['generate_lesson.py', '--name', 'test', str(tmp_path / 'test.txt')]):
        main()
    captured = capsys.readouterr()
    assert "Generating lesson" in captured.out
```

## Quick Wins (Easy to Test)

1. **`format_lesson_markdown()`** - Simple function, easy to test with various inputs
2. **`get_file_date_from_name()`** - Simple parsing function
3. **`calculate_worker_count()`** - Simple calculation function
4. **`expand_env_vars()`** - Simple string manipulation
5. **Error handling in `generate_lesson()`** - Add tests for exception paths

## Coverage Goals

- **Short term**: 75% overall coverage
  - Focus on `generate_lesson.py` CLI interface
  - Add error handling tests
  - Test batch processing functions

- **Medium term**: 85% overall coverage
  - Complete adapter tests
  - Complete LLM provider tests
  - Complete download_audio tests

- **Long term**: 90%+ overall coverage
  - All edge cases covered
  - All error paths tested
  - Integration tests for full workflows

## Running Coverage Reports

```bash
# Generate HTML report
pytest --cov=. --cov-report=html

# View report
open htmlcov/index.html

# Generate terminal report with missing lines
pytest --cov=. --cov-report=term-missing

# Coverage for specific module
pytest --cov=generate_lesson --cov-report=term-missing
```

## Notes

- Some functions like `format_lesson_markdown()` may be deprecated but still present in code
- CLI interfaces (`main()` functions) are harder to test but important for integration
- Error handling paths are critical for robustness
- Mock external dependencies (file I/O, network calls, subprocess) to make tests fast and reliable

