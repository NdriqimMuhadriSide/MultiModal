"""
Audio business logic (FastAPI-facing wrapper).

Thin adapter around processors/audio/audio_processor.py's AudioProcessor
plus the LLM analysis step, mirroring the shape of
app/services/vision_service.py and app/services/rag_service.py: this is
the dependency-injection seam where configuration errors (missing
GROQ_API_KEY, etc.) get turned into clean HTTPExceptions instead of raw
500s. No audio processing or prompt-building logic lives here - that's in
processors/audio/ and prompts/audio_prompts.py respectively; this class
only sequences "process the audio, then analyze the transcript."

Layering: this is the Service layer between the API layer
(app/api/v1/endpoints/audio.py) and the Audio Processing layer
(processors/audio/audio_processor.py) + AI layer (ai/llm_service.py -
reused unchanged from chat, no new LLM client). It never talks to Whisper
directly - that's AudioProcessor's job via its Transcriber.

WHY THIS WRITES TO MEMORY

Same reason as app/services/vision_service.py: this used to be the other
modality that left no trace. It also has something vision doesn't - a
transcript, which is the single most reusable artefact any part of this
system produces. Minutes of speech become text once, at real cost, and
throwing it away with the response meant paying that cost again for every
later question about the same recording.

So the turn is recorded with the audio attached and the transcript stored
alongside the question, capped at settings.audio_transcript_memory_chars.
The cap is the honest compromise: a full hour-long transcript in a
conversation's history would swamp every subsequent prompt, and the
recording itself is kept, so the untruncated text is always one
re-transcription away.

Like vision, it writes but does not read - a conversation_id files the
analysis, it does not change it.
"""
from dataclasses import dataclass

from fastapi import HTTPException, status

from ai.llm_service import LLMService, get_llm_service
from ai.transcription_service import get_transcription_service
from app.core.config import settings
from memory.attachment_store import AttachmentStore, UnsupportedAttachmentType, get_attachment_store
from memory.conversation_memory import ConversationMemory, get_conversation_memory
from prompts.audio_prompts import format_audio_analysis_prompt
from processors.audio.audio_metadata import AudioMetadata
from processors.audio.audio_processor import AudioProcessor
from processors.audio.audio_validator import AudioValidator
from processors.audio.transcriber import Transcriber, Transcript


@dataclass
class AudioAnalysisResult:
    transcript: Transcript
    analysis: str
    metadata: AudioMetadata
    conversation_id: str = ""


# What the stored user turn says when the caller asked no question - the
# same default the analysis prompt applies, spelled out so the record reads
# as something a person said rather than as an empty field.
_DEFAULT_QUESTION = "Summarise this audio."


class AudioAnalysisService:
    """Runs the audio processing pipeline, then analyzes the resulting transcript with the LLM."""

    def __init__(
        self,
        audio_processor: AudioProcessor,
        llm_service: LLMService,
        memory: ConversationMemory,
        attachments: AttachmentStore,
        transcript_memory_chars: int = 4000,
    ) -> None:
        self._audio_processor = audio_processor
        self._llm_service = llm_service
        self._memory = memory
        self._attachments = attachments
        self._transcript_memory_chars = transcript_memory_chars

    def analyze(
        self,
        filename: str,
        mime_type: str,
        audio_bytes: bytes,
        question: str | None,
        conversation_id: str | None = None,
    ) -> AudioAnalysisResult:
        """
        Process an uploaded audio file end to end: validate -> extract
        metadata -> transcribe -> analyze the transcript with the LLM ->
        record the turn.

        `conversation_id` files the turn under an existing conversation; a
        new one is generated when it is absent, and returned either way.

        Raises:
            processors.audio.audio_validator.AudioValidationError: if the
                file fails validation.
            RuntimeError: if transcription or the LLM analysis call fails.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        result = self._audio_processor.process(
            filename=filename, mime_type=mime_type, audio_bytes=audio_bytes
        )

        if not result.transcript.text:
            # A structurally valid audio file with nothing intelligible in
            # it (silence, pure noise) - don't send an empty transcript to
            # the LLM and risk it inventing content; return an honest,
            # deterministic result instead.
            analysis = (
                "No speech was detected in this audio file, so no analysis "
                "could be generated."
            )
        else:
            prompt = format_audio_analysis_prompt(
                transcript=result.transcript.text, question=question
            )
            analysis = self._llm_service.generate_response(prompt)

        self._record(
            resolved_id,
            question=question,
            transcript=result.transcript.text,
            analysis=analysis,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
        )

        return AudioAnalysisResult(
            transcript=result.transcript,
            analysis=analysis,
            metadata=result.metadata,
            conversation_id=resolved_id,
        )

    def _record(
        self,
        conversation_id: str,
        question: str | None,
        transcript: str,
        analysis: str,
        audio_bytes: bytes,
        mime_type: str,
    ) -> None:
        """
        Write the exchange to conversation memory: the question with the
        transcript under it, then the analysis.

        The transcript rides on the user turn rather than getting one of
        its own, because it is not something either party *said* - it is
        what the attachment contains, and a turn that reads "here is my
        question, and here is what the recording says" is the shape a later
        prompt can use without knowing anything about audio.

        Storage failures are swallowed. A recording in a container this
        store has no extension for should not lose the user an analysis
        that already succeeded - the analysis is still written, just
        without the audio behind it.
        """
        try:
            ref = self._attachments.store(audio_bytes, mime_type=mime_type)
        except (UnsupportedAttachmentType, ValueError):
            ref = None

        content = question or _DEFAULT_QUESTION
        if transcript.strip():
            content = f"{content}\n\n[Transcript of the attached audio: {self._clip(transcript)}]"

        self._memory.add_message(
            conversation_id,
            role="user",
            content=content,
            modality="audio",
            attachment_ref=ref,
        )
        self._memory.add_message(
            conversation_id,
            role="assistant",
            content=analysis,
            modality="audio",
            attachment_ref=ref,
        )

    def _clip(self, transcript: str) -> str:
        """
        Cut a transcript down to what a later prompt can afford to carry.

        Truncation is announced rather than silent: a prompt containing a
        transcript that stops mid-sentence, with nothing to say it was cut,
        invites the model to answer as though that were the whole
        recording.
        """
        if len(transcript) <= self._transcript_memory_chars:
            return transcript

        return (
            f"{transcript[: self._transcript_memory_chars]}... (transcript truncated; "
            f"{len(transcript)} characters in full)"
        )


def get_audio_analysis_service() -> AudioAnalysisService:
    """
    FastAPI dependency that builds an AudioAnalysisService.

    Configuration errors (e.g. missing GROQ_API_KEY) happen here, during
    dependency resolution, so they're translated into a clean
    HTTPException instead of leaking a raw stack trace to the client.
    """
    try:
        validator = AudioValidator(max_size_mb=settings.max_audio_size_mb)
        transcriber = Transcriber(transcription_service=get_transcription_service())
        audio_processor = AudioProcessor(validator=validator, transcriber=transcriber)
        return AudioAnalysisService(
            audio_processor=audio_processor,
            llm_service=get_llm_service(),
            memory=get_conversation_memory(),
            attachments=get_attachment_store(),
            transcript_memory_chars=settings.audio_transcript_memory_chars,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Audio analysis service is not configured: {exc}",
        ) from exc
