"""Pydantic v2 schemas for conversational datasets and Neo4j graph dumps."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    """A single turn in an instruction or chat conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Non-empty message content")

    @field_validator("content")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message content cannot be empty or whitespace only")
        return stripped


class ConversationMetadata(BaseModel):
    """Provenance and context tracking for scripture dataset samples."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    scripture: str = Field(default="Hindu scriptures", description="Originating scripture / corpus")
    verse_id: Optional[str] = Field(default=None, description="Canonical verse or text identifier")
    source_entity: Optional[str] = Field(default=None, description="Primary entity name")
    target_entity: Optional[str] = Field(default=None, description="Related entity name")
    relationship_type: Optional[str] = Field(default=None, description="Graph edge label")


class ScriptureConversation(BaseModel):
    """A multi-turn conversation record ready for chat-template SFT."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    messages: List[ChatMessage] = Field(
        ..., min_length=2, description="Turn-by-turn conversation messages"
    )
    metadata: Optional[ConversationMetadata] = Field(
        default=None, description="Scripture provenance metadata"
    )

    @model_validator(mode="after")
    def validate_message_structure(self) -> "ScriptureConversation":
        # Final message must be from assistant
        if self.messages[-1].role != "assistant":
            raise ValueError(
                "The final turn in a fine-tuning conversation must be the assistant response"
            )

        # Must contain at least one user message
        user_turns = [m for m in self.messages if m.role == "user"]
        if not user_turns:
            raise ValueError("Conversation must contain at least one 'user' message")

        # If system message is present, it must be the first message (index 0)
        for idx, msg in enumerate(self.messages):
            if msg.role == "system" and idx != 0:
                raise ValueError("System message must appear strictly as the first turn (index 0)")

        return self


# =====================================================================
# Neo4j Graph Ingestion Schemas
# =====================================================================


class Neo4jNodeProperties(BaseModel):
    """Properties attached to a node in the Neo4j dump."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    translation: Optional[str] = None
    scripture: Optional[str] = None
    verse_id: Optional[str] = None
    description: Optional[str] = None


class Neo4jNode(BaseModel):
    """Node definition within exported Neo4j graph dump."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str = Field(..., description="Internal Neo4j node ID")
    labels: List[str] = Field(default_factory=list, description="Node classification labels")
    properties: Neo4jNodeProperties = Field(default_factory=Neo4jNodeProperties)


class Neo4jRelationship(BaseModel):
    """Relationship (edge) definition within exported Neo4j graph dump."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: Optional[str] = None
    type: str = Field(..., description="Relationship label")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    properties: Dict[str, Any] = Field(default_factory=dict)


class Neo4jDump(BaseModel):
    """Container for complete exported Neo4j graph data."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    metadata: Optional[Dict[str, Any]] = None
    nodes: List[Neo4jNode] = Field(..., description="All exported graph nodes")
    relationships: List[Neo4jRelationship] = Field(
        ..., description="All exported graph relationships"
    )


# =====================================================================
# Evaluation Benchmark Schemas
# =====================================================================


class BenchmarkQuestion(BaseModel):
    """A test case in the 6-category scripture benchmark suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., description="Unique test case ID (e.g. TC-BENCH-01)")
    category_id: int = Field(..., ge=1, le=6, description="Benchmark category (1-6)")
    category_name: str = Field(..., description="Name of the category")
    prompt: str = Field(..., min_length=5, description="Input query for the model")
    reference_answer: Optional[str] = Field(
        default=None, description="Ground truth answer (if applicable)"
    )
    should_refuse: bool = Field(
        default=False, description="True for Category 5 (insufficient evidence)"
    )
    expected_refusal_patterns: List[str] = Field(
        default_factory=list, description="Key refusal indicators"
    )
    required_keywords: List[str] = Field(
        default_factory=list, description="Keywords expected in valid answer"
    )


class BenchmarkEvaluationResult(BaseModel):
    """Structured scorecard generated after running evaluation benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_questions: int
    val_loss: float
    perplexity: float
    category_5_refusal_accuracy: float
    category_scores: Dict[str, float]
    passed: bool
