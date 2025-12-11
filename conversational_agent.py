#!/usr/bin/env python3
"""
Conversational Learning Agent - Interactive chat interface using Ollama
Allows back-and-forth conversation to create learning packs
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone

from lesson_search import (
    get_all_lessons, get_lessons_for_channel,
    search_lessons_by_keywords, LessonContent, parse_lesson_markdown
)
from llm_providers import OllamaProvider
import requests
from channel_fetcher import load_config, find_podcast_by_name
from generate_lesson import expand_env_vars
from learning_agent import ToolExecutor, create_learning_pack_prompt, generate_learning_pack
from download_audio import parse_date_arg
from chroma_search import get_chroma_collection, search_transcriptions, store_transcription_chunks, check_chunks_exist
from transcription_chunks import find_transcription_files, chunk_transcription_file
import logging

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Conversational agent that maintains context and handles follow-up questions"""
    
    def __init__(self, config_data: Dict, provider: OllamaProvider):
        self.config_data = config_data
        self.provider = provider
        self.conversation_history: List[Dict[str, str]] = []
        self.current_plan: Optional[Dict] = None
        self.executed_lessons: List[LessonContent] = []
        self.executor = ToolExecutor(config_data, provider)
        # Track conversation context for learning pack creation
        self.current_channel: Optional[str] = None
        self.current_from_date: Optional[datetime] = None
        self.current_to_date: Optional[datetime] = None
        # Track recent lessons used for learning pack (for answering questions)
        self.recent_lessons: List[LessonContent] = []
        # Track episodes that need processing (for "yes" confirmation)
        self.pending_episodes: List[Dict] = []
        
        # Initialize Chroma collection (optional, falls back to keyword search if unavailable)
        self.chroma_collection = None
        try:
            self.chroma_collection = get_chroma_collection(self.executor.data_root, config_data)
            if self.chroma_collection:
                logger.info("Chroma collection initialized successfully")
            else:
                print("ℹ️  Chroma semantic search is disabled in configuration.")
                print("   Will use keyword-based search using lesson summaries.")
                logger.info("Chroma disabled or unavailable, will use keyword search")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Could not initialize Chroma semantic search: {error_msg}")
            print("   Will use keyword-based search using lesson summaries as fallback.")
            logger.warning(f"Could not initialize Chroma: {e}. Will use keyword search as fallback.")
            self.chroma_collection = None
        
        # Initialize with system message
        self.conversation_history.append({
            "role": "system",
            "content": """You are a helpful Japanese language learning assistant. You help users create personalized learning packs from podcast content.

You can:
1. Find podcast episodes
2. Download audio files
3. Transcribe audio
4. Generate lessons
5. Create learning packs
6. Answer questions about existing lessons
7. Suggest next steps

Be conversational, helpful, and proactive. Ask clarifying questions when needed."""
        })
    
    def add_user_message(self, message: str):
        """Add user message to conversation history"""
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
    
    def add_assistant_message(self, message: str):
        """Add assistant response to conversation history"""
        self.conversation_history.append({
            "role": "assistant",
            "content": message
        })
    
    def get_conversation_context(self, last_n: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context"""
        # Always include system message
        return [self.conversation_history[0]] + self.conversation_history[-last_n:]
    
    def understand_intent(self, user_message: str) -> Dict:
        """Use LLM to understand what the user wants to do"""
        context = self.get_conversation_context()
        context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        
        prompt = f"""Based on the conversation history and the user's latest message, determine their intent.

Conversation history:
{context_text}

User's latest message: "{user_message}"

What does the user want to do? Respond with JSON:
{{
  "intent": "create_pack" | "ask_question" | "modify_request" | "get_suggestions" | "check_status",
  "needs_tools": true | false,
  "action": "find_episodes" | "download" | "transcribe" | "generate_lessons" | "create_pack" | "answer_question" | null,
  "parameters": {{
    "channel": "channel_name or null (extract from message, e.g., 'sjn' from 'sjn 7d')",
    "keywords": ["keyword1", "keyword2"],
    "from_date": "null (DO NOT calculate dates - leave as null for relative dates like 'last 7 days' or '7d')",
    "to_date": "null (DO NOT calculate dates - leave as null for relative dates)"
  }},
  "clarification_needed": true | false,
  "clarification_question": "question to ask user or null"
}}

IMPORTANT: 
- DO NOT calculate dates. If user mentions "last 7 days", "last week", "7d", "7w", etc., set from_date and to_date to null.
- Only set dates if user provides explicit dates like "2024-01-15" or "January 15, 2024".
- Extract channel name from the message (e.g., "sjn" from "sjn 7d", "hnh" from "find episodes from hnh").
- Relative date expressions will be parsed automatically by the system."""
        
        try:
            response = self.provider.generate(
                prompt=prompt,
                system_prompt="You are an intent classifier. Respond ONLY with valid JSON.",
                temperature=0.3,
                max_tokens=300
            )
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    json_str = json_match.group(0)
                    # Remove trailing commas
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error understanding intent: {e}")
        
        # Default fallback
        return {
            "intent": "create_pack",
            "needs_tools": False,
            "action": None,
            "parameters": {},
            "clarification_needed": False,
            "clarification_question": None
        }
    
    def chat(self, user_message: str) -> str:
        """Process user message and return response"""
        self.add_user_message(user_message)
        
        # Check if user is confirming to process pending episodes
        user_lower = user_message.lower().strip()
        if self.pending_episodes and user_lower in ['yes', 'y', 'ok', 'okay', 'sure', 'go ahead', 'do it', 'proceed']:
            return self.process_pending_episodes()
        
        # Check for common lesson/pack creation requests
        if any(phrase in user_lower for phrase in ['create lesson', 'make lesson', 'generate lesson', 'create pack', 'make pack', 'generate pack']):
            # If we have recent lessons or current channel context, create a learning pack
            if self.recent_lessons or self.current_channel:
                return self.create_learning_pack_from_conversation()
            # Otherwise, try to understand intent normally
            pass
        
        # Understand intent
        intent = self.understand_intent(user_message)
        
        # Handle clarification requests
        if intent.get("clarification_needed"):
            question = intent.get("clarification_question", "Could you clarify what you'd like to do?")
            self.add_assistant_message(question)
            return question
        
        # Handle different intents
        if intent["intent"] == "ask_question":
            return self.answer_question(user_message)
        elif intent["intent"] == "check_status":
            return self.check_status()
        elif intent["intent"] == "get_suggestions":
            return self.get_suggestions()
        elif intent["needs_tools"]:
            return self.execute_tools(intent)
        else:
            # General conversation
            return self.converse(user_message)
    
    def ensure_transcription_chunks_loaded(self, channel: Optional[str] = None,
                                           from_date: Optional[datetime] = None,
                                           to_date: Optional[datetime] = None) -> None:
        """
        Ensure transcription chunks are loaded into Chroma.
        Only processes transcriptions that haven't been chunked yet.
        """
        if not self.chroma_collection:
            return
        
        try:
            # Find transcription files
            transcription_files = find_transcription_files(
                self.executor.data_root,
                channel_name=channel,
                from_date=from_date,
                to_date=to_date
            )
            
            if not transcription_files:
                logger.info("No transcription files found to chunk")
                return
            
            # Process each transcription file
            total_chunks_stored = 0
            for transcription_file in transcription_files:
                try:
                    # Chunk the transcription
                    chunks = chunk_transcription_file(transcription_file, self.executor.data_root)
                    
                    if not chunks:
                        continue
                    
                    # Check which chunks already exist
                    chunk_ids = [c['chunk_id'] for c in chunks]
                    existing_ids = check_chunks_exist(self.chroma_collection, chunk_ids)
                    
                    # Filter out existing chunks
                    new_chunks = [c for c in chunks if c['chunk_id'] not in existing_ids]
                    
                    if new_chunks:
                        # Store new chunks
                        stored_count = store_transcription_chunks(
                            new_chunks,
                            self.chroma_collection,
                            force=False
                        )
                        total_chunks_stored += stored_count
                        logger.info(f"Stored {stored_count} new chunks from {transcription_file.name}")
                
                except Exception as e:
                    logger.error(f"Error processing transcription file {transcription_file}: {e}")
                    continue
            
            if total_chunks_stored > 0:
                logger.info(f"Loaded {total_chunks_stored} new transcription chunks into Chroma")
        
        except Exception as e:
            logger.error(f"Error ensuring transcription chunks loaded: {e}")
    
    def answer_question(self, question: str) -> str:
        """Answer questions about lessons or transcriptions using semantic search"""
        # Try semantic search with Chroma first
        if self.chroma_collection:
            try:
                # Get config for search parameters
                embeddings_cfg = self.config_data.get('embeddings', {})
                top_k = embeddings_cfg.get('top_k', 5)
                similarity_threshold = embeddings_cfg.get('similarity_threshold', 0.5)
                
                # Ensure chunks are loaded (lazy loading)
                self.ensure_transcription_chunks_loaded(
                    channel=self.current_channel,
                    from_date=self.current_from_date,
                    to_date=self.current_to_date
                )
                
                # Perform semantic search
                chunks = search_transcriptions(
                    question,
                    self.chroma_collection,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                
                if chunks:
                    # Format chunks with citations
                    context_parts = []
                    for i, chunk in enumerate(chunks, 1):
                        source = chunk.get('source', 'Unknown')
                        date = chunk.get('episode_date', '')
                        content = chunk.get('content', '')
                        similarity = chunk.get('similarity', 0.0)
                        
                        citation = f"From {source}"
                        if date:
                            citation += f" ({date})"
                        citation += f" [similarity: {similarity:.2f}]"
                        
                        context_parts.append(f"{citation}:\n{content[:300]}...")  # Limit content length
                    
                    context_text = "\n\n".join(context_parts)
                    
                    prompt = f"""You are a helpful Japanese language learning assistant.

I found {len(chunks)} relevant transcription excerpt(s) from the podcast episodes:

{context_text}

User's question: {question}

Answer the question based on the transcription excerpts above. Be specific and cite the sources. If the question is about specific content, reference the exact excerpts."""
                    
                    response = self.provider.generate(
                        prompt=prompt,
                        system_prompt="You are a helpful Japanese language learning assistant. Answer questions based on the transcription excerpts provided. Be specific and cite sources.",
                        temperature=0.7
                    )
                    
                    self.add_assistant_message(response)
                    return response
                else:
                    # No chunks found above similarity threshold
                    print(f"⚠️  Semantic search found no relevant transcription chunks (similarity threshold: {similarity_threshold:.2f}).")
                    print("   Falling back to keyword-based search using lesson summaries.")
            
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️  Semantic search failed: {error_msg}")
                print("   Falling back to keyword-based search using lesson summaries.")
                logger.warning(f"Semantic search failed: {e}. Falling back to keyword search.")
        
        # Fallback to keyword-based search using lessons
        if not self.chroma_collection:
            print("ℹ️  Chroma semantic search is not available (disabled or initialization failed).")
            print("   Using keyword-based search using lesson summaries.")
        # Use recent lessons from learning pack if available, otherwise get all lessons
        lessons_to_use = self.recent_lessons if self.recent_lessons else get_all_lessons(self.executor.data_root)
        
        if not lessons_to_use:
            return "I don't have any lessons available yet. Would you like me to find and generate some episodes?"
        
        # Create detailed context about the lessons
        lesson_contexts = []
        for i, lesson in enumerate(lessons_to_use[:5], 1):  # Limit to 5 for context
            date_str = lesson.date.strftime('%Y-%m-%d') if lesson.date else "Unknown date"
            context = f"""Lesson {i}: {lesson.episode_title}
Date: {date_str}
Summary: {lesson.summary[:200] if lesson.summary else "No summary available"}"""
            
            if lesson.vocabulary:
                vocab_list = ", ".join([v.get('word', '') for v in lesson.vocabulary[:10]])
                context += f"\nKey Vocabulary: {vocab_list}"
            
            if lesson.grammar_points:
                grammar_list = ", ".join([g.get('pattern', '') for g in lesson.grammar_points[:5]])
                context += f"\nGrammar Points: {grammar_list}"
            
            lesson_contexts.append(context)
        
        lessons_text = "\n\n".join(lesson_contexts)
        if len(lessons_to_use) > 5:
            lessons_text += f"\n\n(And {len(lessons_to_use) - 5} more lessons)"
        
        prompt = f"""You are a helpful Japanese language learning assistant.

I have {len(lessons_to_use)} lesson(s) available. Here are the details:

{lessons_text}

User's question: {question}

Answer the question based on the lesson content above. Be specific and reference the actual content from the lessons. If the question is about "the lesson" or "these lessons", refer to the lessons listed above."""
        
        response = self.provider.generate(
            prompt=prompt,
            system_prompt="You are a helpful Japanese language learning assistant. Answer questions based on the lesson content provided. Be specific and reference actual content from the lessons.",
            temperature=0.7
        )
        
        self.add_assistant_message(response)
        return response
    
    def check_status(self) -> str:
        """Check status of current work"""
        status_parts = []
        
        if self.current_plan:
            status_parts.append(f"Current plan: {len(self.current_plan.get('steps', []))} steps")
        
        if self.executed_lessons:
            status_parts.append(f"Working with {len(self.executed_lessons)} lessons")
        
        lessons = get_all_lessons(self.executor.data_root)
        if lessons:
            status_parts.append(f"Total lessons available: {len(lessons)}")
        
        if not status_parts:
            status = "Ready to help! What would you like to learn?"
        else:
            status = "Status: " + ". ".join(status_parts)
        
        self.add_assistant_message(status)
        return status
    
    def get_suggestions(self) -> str:
        """Get suggestions for next steps"""
        lessons = get_all_lessons(self.executor.data_root)
        
        if not lessons:
            suggestions = [
                "Find episodes from a specific channel",
                "Download audio from recent episodes",
                "Generate lessons from existing transcriptions"
            ]
        else:
            # Analyze lessons to suggest topics
            jlpt_levels = set()
            for lesson in lessons[:10]:
                jlpt_levels.update(lesson.jlpt_levels)
            
            suggestions = [
                f"Review {', '.join(sorted(jlpt_levels))} level vocabulary" if jlpt_levels else "Review vocabulary",
                "Create a learning pack on a specific topic",
                "Find more episodes on a topic you're interested in"
            ]
        
        response = f"Here are some suggestions:\n" + "\n".join(f"- {s}" for s in suggestions)
        self.add_assistant_message(response)
        return response
    
    def parse_relative_dates(self, user_message: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Parse relative date expressions like 'last 7 days' or '7d' from user message"""
        user_lower = user_message.lower()
        
        # First, check for direct "7d", "7w", "7m" format (e.g., "sjn 7d")
        match = re.search(r'\b(\d+)([dwm])\b', user_lower)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            try:
                from_date = parse_date_arg(f"{number}{unit}")
                if from_date:
                    from_date = from_date.replace(tzinfo=None)
                to_date = datetime.now()
                return from_date, to_date
            except Exception:
                pass
        
        # Extract "7d" format from natural language
        # Check for "last 7 days" or "last 7day" -> convert to "7d"
        match = re.search(r'last\s*(\d+)\s*days?', user_lower)
        if match:
            days = match.group(1)
            try:
                # Use the existing parse_date_arg function that supports "7d" format
                from_date = parse_date_arg(f"{days}d")
                # Convert to naive datetime for compatibility (parse_date_arg returns UTC datetime)
                if from_date:
                    from_date = from_date.replace(tzinfo=None)
                to_date = datetime.now()
                return from_date, to_date
            except Exception:
                pass
        
        # Check for "last week" -> convert to "7d"
        if re.search(r'last\s*week', user_lower):
            try:
                from_date = parse_date_arg("7d")
                if from_date:
                    from_date = from_date.replace(tzinfo=None)
                to_date = datetime.now()
                return from_date, to_date
            except Exception:
                pass
        
        return None, None
    
    def execute_tools(self, intent: Dict) -> str:
        """Execute tools based on intent"""
        action = intent.get("action")
        params = intent.get("parameters", {})
        
        if action == "find_episodes":
            channel = params.get("channel") or "hnh"
            keywords = params.get("keywords", [])
            
            # Get last user message once
            last_user_msg = next((msg["content"] for msg in reversed(self.conversation_history) if msg["role"] == "user"), "")
            
            # Fallback: extract channel from user message if LLM didn't extract it
            # This handles short formats like "sjn 7d"
            if not channel or channel == "hnh":
                # Check if message contains a known channel name
                known_channels = [podcast["name"] for podcast in self.config_data.get("podcasts", [])]
                words = last_user_msg.lower().split()
                for word in words:
                    # Remove any punctuation
                    clean_word = word.strip('.,!?;:')
                    if clean_word in known_channels:
                        channel = clean_word
                        break
            
            # Always parse relative dates from user message first (more reliable than LLM)
            from_date = None
            to_date = None
            
            parsed_from, parsed_to = self.parse_relative_dates(last_user_msg)
            if parsed_from:
                from_date = parsed_from
            if parsed_to:
                to_date = parsed_to
            
            # Only use LLM-provided dates if we didn't parse from message AND they look valid (not old dates)
            if not from_date and params.get("from_date"):
                try:
                    # Check if it's in "7d" format
                    if params["from_date"].endswith('d') or params["from_date"].endswith('w') or params["from_date"].endswith('m'):
                        parsed = parse_date_arg(params["from_date"])
                        if parsed:
                            from_date = parsed.replace(tzinfo=None)
                    else:
                        # Check if it's a recent date (within last year)
                        parsed_date = datetime.strptime(params["from_date"], '%Y-%m-%d')
                        # Only use if it's recent (not from 2023 or earlier)
                        if parsed_date.year >= datetime.now().year - 1:
                            from_date = parsed_date
                except Exception:
                    pass
            
            if not to_date and params.get("to_date"):
                try:
                    if params["to_date"].endswith('d') or params["to_date"].endswith('w') or params["to_date"].endswith('m'):
                        parsed = parse_date_arg(params["to_date"])
                        if parsed:
                            to_date = parsed.replace(tzinfo=None)
                    else:
                        parsed_date = datetime.strptime(params["to_date"], '%Y-%m-%d')
                        if parsed_date.year >= datetime.now().year - 1:
                            to_date = parsed_date
                except Exception:
                    pass
            
            episodes = self.executor.find_episodes(channel, from_date=from_date, to_date=to_date, keywords=keywords)
            
            # Track context for learning pack creation
            self.current_channel = channel
            self.current_from_date = from_date
            self.current_to_date = to_date
            
            # Check which episodes already have lessons by checking for lesson files
            # Lesson files are named: {clean_filename}_lesson.md
            episodes_with_lessons = []
            episodes_needing_work = []
            
            audio_dir = self.executor.data_root / channel
            
            for ep in episodes:
                clean_filename = ep.get('clean_filename', '')
                video_id = ep.get('id')
                
                # First check by exact filename
                has_lesson = False
                if clean_filename:
                    lesson_file = audio_dir / f"{clean_filename}_lesson.md"
                    if lesson_file.exists():
                        has_lesson = True
                
                # Also check by video ID if available (handles filename variations)
                if not has_lesson and video_id:
                    matching_lessons = list(audio_dir.glob(f"*{video_id}*_lesson.md"))
                    if matching_lessons:
                        has_lesson = True
                
                # Fallback: check by title if no filename/video ID match
                if not has_lesson:
                    ep_title = ep.get('title', '')
                    existing_lessons = get_lessons_for_channel(
                        self.executor.data_root,
                        channel,
                        from_date=from_date,
                        to_date=to_date
                    )
                    has_lesson = any(
                        ep_title in lesson.episode_title or 
                        lesson.episode_title in ep_title 
                        for lesson in existing_lessons
                    )
                
                if has_lesson:
                    episodes_with_lessons.append(ep)
                else:
                    episodes_needing_work.append(ep)
            
            response = f"Found {len(episodes)} episodes matching your criteria."
            
            if episodes_with_lessons:
                response += f"\n\n✓ {len(episodes_with_lessons)} episode(s) already have lessons (will skip download/transcribe/generate):"
                for ep in episodes_with_lessons[:3]:
                    response += f"\n  - {ep.get('title', 'Unknown')}"
                if len(episodes_with_lessons) > 3:
                    response += f"\n  ... and {len(episodes_with_lessons) - 3} more"
            
            if episodes_needing_work:
                # Store episodes for processing if user says "yes"
                self.pending_episodes = episodes_needing_work
                response += f"\n\n📝 {len(episodes_needing_work)} episode(s) need processing:"
                for ep in episodes_needing_work[:3]:
                    response += f"\n  - {ep.get('title', 'Unknown')}"
                if len(episodes_needing_work) > 3:
                    response += f"\n  ... and {len(episodes_needing_work) - 3} more"
                response += f"\n\nWould you like me to download, transcribe, and generate lessons for these?"
            else:
                # Clear pending episodes if all have lessons
                self.pending_episodes = []
            
            self.add_assistant_message(response)
            return response
        
        elif action == "create_pack":
            return self.create_learning_pack_from_conversation()
        
        elif action in ["download", "transcribe", "generate_lessons"]:
            # Handle explicit download/transcribe/generate requests
            if action == "download" and self.pending_episodes:
                return self.process_pending_episodes()
            else:
                response = f"I can help with that! Let me work on it..."
                self.add_assistant_message(response)
                return response
        
        else:
            response = f"I can help with that! Let me work on it..."
            self.add_assistant_message(response)
            return response
    
    def process_pending_episodes(self) -> str:
        """Process pending episodes: download, transcribe, and generate lessons"""
        if not self.pending_episodes:
            response = "I don't have any episodes to process. Please find episodes first."
            self.add_assistant_message(response)
            return response
        
        if not self.current_channel:
            response = "I need to know which channel to use. Please find episodes first."
            self.add_assistant_message(response)
            return response
        
        try:
            channel = self.current_channel
            episodes = self.pending_episodes
            
            # Step 1: Download audio
            print(f"\n📥 Downloading {len(episodes)} episode(s)...")
            audio_files = self.executor.download_audio(channel, episodes, force=False)
            print(f"✓ Downloaded {len(audio_files)} audio file(s)\n")
            
            if not audio_files:
                response = "No audio files were downloaded. They may already exist or there was an error."
                self.add_assistant_message(response)
                return response
            
            # Step 2: Transcribe audio
            print(f"🎤 Transcribing {len(audio_files)} audio file(s)...")
            transcription_files = self.executor.transcribe_audio(audio_files, force=False)
            print(f"✓ Transcribed {len(transcription_files)} file(s)\n")
            
            if not transcription_files:
                response = "No transcriptions were created. They may already exist or there was an error."
                self.add_assistant_message(response)
                return response
            
            # Step 3: Generate lessons
            print(f"📚 Generating lessons from {len(transcription_files)} transcription(s)...")
            lesson_files = self.executor.generate_lessons(transcription_files, force=False)
            print(f"✓ Generated {len(lesson_files)} lesson(s)\n")
            
            # Clear pending episodes
            self.pending_episodes = []
            
            response = f"✓ Successfully processed {len(episodes)} episode(s)!\n\n"
            response += f"- Downloaded: {len(audio_files)} audio file(s)\n"
            response += f"- Transcribed: {len(transcription_files)} file(s)\n"
            response += f"- Generated: {len(lesson_files)} lesson(s)\n\n"
            response += "Would you like me to create a learning pack from these lessons?"
            
            self.add_assistant_message(response)
            return response
            
        except Exception as e:
            error_msg = f"Error processing episodes: {e}"
            print(f"✗ {error_msg}")
            self.add_assistant_message(error_msg)
            return error_msg
    
    def create_learning_pack_from_conversation(self) -> str:
        """Create a learning pack based on conversation"""
        # Extract the main query from conversation
        user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        main_query = user_messages[-1] if user_messages else "Japanese vocabulary"
        
        # Get lessons based on conversation context (channel and date range)
        if self.current_channel:
            # Use the channel and date range from the conversation
            lessons = get_lessons_for_channel(
                self.executor.data_root, 
                self.current_channel,
                from_date=self.current_from_date,
                to_date=self.current_to_date
            )
        else:
            # Fallback to all lessons if no channel context
            lessons = get_all_lessons(
                self.executor.data_root,
                from_date=self.current_from_date,
                to_date=self.current_to_date
            )
        
        if not lessons:
            response = "I don't have any lessons available yet. Would you like me to find and generate some episodes first?"
            self.add_assistant_message(response)
            return response
        
        # Select relevant lessons (filter out "create", "me", "learning", "pack" from keywords)
        keywords = [kw for kw in main_query.lower().split() if kw not in ["create", "me", "a", "learning", "pack", "make", "generate"]]
        if keywords:
            selected_lessons = search_lessons_by_keywords(lessons, keywords)[:10]
        else:
            selected_lessons = []
        
        if not selected_lessons:
            # Use all lessons from the context (channel + date range)
            selected_lessons = lessons[:10]
        
        # Store lessons for answering questions later
        self.recent_lessons = selected_lessons
        
        # Generate learning pack
        try:
            print(f"\n📚 Found {len(selected_lessons)} lesson(s) to use for the learning pack")
            print(f"📝 Preparing content from lessons...")
            
            # Show which lessons are being used
            for i, lesson in enumerate(selected_lessons[:5], 1):
                date_str = lesson.date.strftime('%Y-%m-%d') if lesson.date else "Unknown date"
                print(f"   {i}. {lesson.episode_title[:60]}... ({date_str})")
            if len(selected_lessons) > 5:
                print(f"   ... and {len(selected_lessons) - 5} more")
            
            print(f"\n🤖 Generating learning pack with LLM (this may take a minute)...")
            learning_pack = generate_learning_pack(self.provider, main_query, selected_lessons)
            print(f"✓ Learning pack generated successfully!")
            
            # Save pack
            print(f"💾 Saving learning pack to file...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"learning_pack_{timestamp}.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Learning Pack\n\n")
                f.write(f"**Query:** {main_query}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Based on {len(selected_lessons)} lesson(s)**\n\n")
                f.write("---\n\n")
                f.write(learning_pack)
            
            print(f"✓ Saved to: {output_path}\n")
            
            response = f"✓ Created learning pack with {len(selected_lessons)} lessons!\n"
            response += f"Saved to: {output_path}\n\n"
            response += "Would you like me to:\n"
            response += "- Create another pack on a different topic?\n"
            response += "- Find more episodes?\n"
            response += "- Answer questions about the content?"
            
            self.add_assistant_message(response)
            return response
        except Exception as e:
            error_msg = f"Error creating learning pack: {e}"
            self.add_assistant_message(error_msg)
            return error_msg
    
    def converse(self, message: str) -> str:
        """General conversation using LLM with context"""
        # Use Ollama's chat API with full conversation history
        messages = self.get_conversation_context()
        
        # Use Ollama's chat API directly with full conversation history
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
            }
        }
        
        try:
            response = requests.post(
                f"{self.provider.base_url}/api/chat",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            assistant_response = result.get("message", {}).get("content", "")
            
            self.add_assistant_message(assistant_response)
            return assistant_response
        except Exception as e:
            # Fallback to simple generation
            response = self.provider.generate(
                prompt=message,
                system_prompt="You are a helpful Japanese language learning assistant. Be conversational and friendly.",
                temperature=0.7
            )
            self.add_assistant_message(response)
            return response


def main():
    parser = argparse.ArgumentParser(
        description='Conversational Learning Agent - Interactive chat interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to config file')
    parser.add_argument('--model',
                       help='Ollama model name (overrides config)')
    
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
    
    # Initialize Ollama provider
    analysis_cfg = config_data.get('analysis', {})
    model = args.model or analysis_cfg.get('model', 'qwen2.5:14b')
    base_url_raw = analysis_cfg.get('base_url') or analysis_cfg.get('ollama_url') or 'http://localhost:11434'
    base_url = expand_env_vars(base_url_raw) if isinstance(base_url_raw, str) else base_url_raw
    
    try:
        print(f"Connecting to Ollama...")
        provider = OllamaProvider(model=model, base_url=base_url)
        
        if not provider.is_available():
            print(f"Error: Cannot connect to Ollama")
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)
        
        print(f"✓ Connected to Ollama\n")
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        sys.exit(1)
    
    # Create conversational agent
    agent = ConversationalAgent(config_data, provider)
    
    # Start conversation loop
    print("=" * 80)
    print("CONVERSATIONAL LEARNING AGENT")
    print("=" * 80)
    print("\nI'm here to help you create personalized Japanese learning packs!")
    print("You can ask me to:")
    print("  - Find episodes on specific topics")
    print("  - Create learning packs")
    print("  - Answer questions about lessons")
    print("  - Get suggestions for what to learn next")
    print("\nType 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy learning!")
                break
            
            # Process message
            response = agent.chat(user_input)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy learning!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == '__main__':
    main()

