---
name: youtube-summary
description: Generate comprehensive learning summaries from Japanese podcast episodes. Use when the user wants to create study materials, vocabulary lists, or quizzes from recent podcast episodes.
allowed-tools: Bash, Read, Write, Glob, Grep
---

# YouTube Summary Skill

Generate markdown learning materials from Japanese podcast episodes using native tools (no Python scripts).

## Configuration

- **Data directory:** `$PODPILOT_DATA` (set this env variable to your data directory)
- **Whisper model:** `$WHISPER_MODEL_PATH` (ggml-base.bin)
- **Channels:** hnh, sjn, ss, yuyu

### Channel IDs
| Short | Channel ID |
|-------|------------|
| hnh | UCauyM-A8JIJ9NQcw5_jF00Q |
| sjn | UC_NROu3WWx1KZ7tNl275F7A |
| ss | UCqMY-cp1He6IAi1cIz-gX1g |
| yuyu | UC8dWfySP_cKDMFj6aFfQbFA |

## Workflow

### 1. Fetch Episodes from RSS

```bash
curl -s "https://www.youtube.com/feeds/videos.xml?channel_id=<CHANNEL_ID>"
```

Parse the XML to extract:
- `<yt:videoId>` - Video ID
- `<title>` - Episode title
- `<published>` - Publish date

### 2. Download Audio

```bash
yt-dlp -x --audio-format mp3 --audio-quality 0 \
  -o "<data_dir>/<channel>/<date>_<title>_<video_id>.%(ext)s" \
  "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

### 3. Transcribe with Whisper

Run whisper with both TXT and SRT output:

```bash
whisper-cli -m $WHISPER_MODEL_PATH -l ja \
  -f "<audio_file>.mp3" \
  --output-txt --output-srt \
  -of "<output_base>"
```

### 4. Create Linked Transcript

Convert SRT to transcript with clickable YouTube timestamps:

```bash
VIDEO_ID="<video_id>"
awk -v vid="$VIDEO_ID" '
BEGIN { RS=""; FS="\n" }
{
    split($2, times, " --> ")
    start_time = times[1]
    gsub(",", ".", start_time)
    split(start_time, parts, ":")
    seconds = int(parts[1] * 3600 + parts[2] * 60 + parts[3])
    text = ""
    for (i = 3; i <= NF; i++) {
        if (text != "") text = text "\n"
        text = text $i
    }
    printf "[%s] https://www.youtube.com/watch?v=%s&t=%d\n%s\n\n", start_time, vid, seconds, text
}
' "<srt_file>" > "<output>_linked.txt"
```

Output format:
```
[00:00:00.000] https://www.youtube.com/watch?v=VIDEO_ID&t=0
皆さんこんにちは...

[00:00:05.760] https://www.youtube.com/watch?v=VIDEO_ID&t=5
このチャンネルでは...
```

### 5. Generate Lesson

Use the `/japanese-lesson` skill format to analyze the transcript and create:
- Summary (Japanese with furigana + English)
- Vocabulary tables by JLPT level (N1→N5)
- Grammar points with examples
- Reading comprehension with context clues
- 10-question quiz

Save to: `<data_dir>/<channel>/<date>_<title>_lesson.md`

## Output Files

For each episode, create:
```
<data_dir>/<channel>/
├── <date>_<title>_<video_id>.mp3         # Audio
├── <date>_<title>_<video_id>.txt         # Plain transcript
├── <date>_<title>_<video_id>.srt         # Subtitles
├── <date>_<title>_<video_id>_linked.txt  # Transcript with YouTube links
└── <date>_<title>_<video_id>_lesson.md   # Lesson (vocab, grammar, quiz)
```

## Example Usage

User: "Process the latest sjn episode"

1. Fetch RSS for sjn (UC_NROu3WWx1KZ7tNl275F7A)
2. Show available episodes, let user pick
3. Download audio with yt-dlp
4. Transcribe with whisper-cli (creates .txt and .srt)
5. Convert SRT to linked transcript (_linked.txt)
6. Generate lesson using /japanese-lesson format
7. Report files created

## Notes

- Transcription takes ~1 minute per 10 minutes of audio
- Lesson generation is done by Claude directly (no external LLM call)
- Linked transcripts allow clicking to jump to exact moment in YouTube video
