"""Comprehensive unit and integration test suite for VidwaanAI Unsloth QLoRA pipeline.

Tests Pydantic v2 configuration validation, dataset schemas, leak-free splitting,
dataset validator isolation, 6-category evaluation benchmark, export safety, and
error handling. All tests run headless on CPU without downloading large external model weights.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from evaluate import (
    calculate_perplexity,
    evaluate_keywords,
    evaluate_refusal,
    load_benchmark_questions,
    run_benchmark_evaluation,
    write_evaluation_reports,
)
from export_model import export_model_weights
from scripts.convert_neo4j_json_to_jsonl import (
    extract_scripture_conversations,
    load_neo4j_dump,
    partition_scriptures_leak_free,
    write_jsonl_file,
)
from src.config_schema import (
    LoraParameters,
    TrainingParameters,
    TrainingPipelineConfig,
)
from src.dataset_schema import (
    BenchmarkEvaluationResult,
    BenchmarkQuestion,
    ChatMessage,
    ConversationMetadata,
    Neo4jDump,
    Neo4jNode,
    Neo4jNodeProperties,
    Neo4jRelationship,
    ScriptureConversation,
)
from validate_local_data import (
    DatasetValidator,
    call_local_critic,
    estimate_tokens,
    find_best_context_chunk,
    find_matched_entities,
    segment_text,
)

# =====================================================================
# 1. Configuration Validation Tests (Pydantic v2)
# =====================================================================


def test_config_loading_valid():
    """Verify that config.yaml loads and passes strict Pydantic v2 validation."""
    config = TrainingPipelineConfig.from_yaml("config.yaml")
    assert "Qwen" in config.model
    assert config.max_seq_length >= 512
    assert config.load_in_4bit is True
    assert config.lora_parameters.rank == 16
    assert "q_proj" in config.lora_parameters.target_modules
    assert config.training.batch_size >= 1
    assert config.training.learning_rate > 0


def test_config_invalid_target_module():
    """Verify that unsupported target modules trigger validation errors."""
    with pytest.raises(ValidationError) as exc_info:
        LoraParameters(target_modules=["invalid_module_name"])
    assert "Invalid target module" in str(exc_info.value)


def test_config_bounds_validation():
    """Verify boundary checks on rank, batch size, learning rate, and sequence length."""
    with pytest.raises(ValidationError):
        LoraParameters(rank=0)  # ge=1

    with pytest.raises(ValidationError):
        LoraParameters(rank=129)  # le=128

    with pytest.raises(ValidationError):
        LoraParameters(scale=0.0)  # gt=0.0

    with pytest.raises(ValidationError):
        LoraParameters(dropout=0.6)  # le=0.5

    with pytest.raises(ValidationError):
        TrainingParameters(batch_size=0)  # ge=1

    with pytest.raises(ValidationError):
        TrainingParameters(batch_size=65)  # le=64

    with pytest.raises(ValidationError):
        TrainingParameters(learning_rate=-0.01)  # gt=0.0

    with pytest.raises(ValidationError):
        TrainingParameters(num_train_epochs=0)  # ge=1

    with pytest.raises(ValidationError):
        TrainingPipelineConfig(max_seq_length=128)  # ge=256


def test_config_extra_fields_forbidden():
    """Verify that arbitrary extra fields are rejected under extra='forbid'."""
    with pytest.raises(ValidationError):
        LoraParameters(unknown_field="unexpected")

    with pytest.raises(ValidationError):
        TrainingParameters(unknown_field=123)

    with pytest.raises(ValidationError):
        TrainingPipelineConfig(unexpected_flag=True)


def test_config_from_yaml_missing_file(tmp_path):
    """Verify that non-existent YAML paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TrainingPipelineConfig.from_yaml(tmp_path / "non_existent.yaml")


def test_config_from_yaml_legacy_flat_compatibility(tmp_path):
    """Verify backward compatibility: legacy flat YAML keys are correctly parsed into nested models."""
    legacy_yaml = tmp_path / "legacy_config.yaml"
    content = """
    model: "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    max_seq_length: 1024
    load_in_4bit: true
    batch_size: 4
    grad_accumulation_steps: 8
    learning_rate: 0.0003
    iters: 200
    adapter_path: "./custom_adapters"
    grad_checkpoint: false
    data: "./custom_data"
    """
    legacy_yaml.write_text(content, encoding="utf-8")

    cfg = TrainingPipelineConfig.from_yaml(legacy_yaml)
    assert cfg.max_seq_length == 1024
    assert cfg.training.batch_size == 4
    assert cfg.training.grad_accumulation_steps == 8
    assert cfg.training.learning_rate == 0.0003
    assert cfg.training.max_steps == 200
    assert str(cfg.training.adapter_path).replace("\\", "/") == "custom_adapters"
    assert cfg.training.grad_checkpoint is False
    assert str(cfg.data_dir).replace("\\", "/") == "custom_data"


# =====================================================================
# 2. Dataset Schema & Validation Tests
# =====================================================================


def test_chat_message_whitespace_rejection():
    """Verify that whitespace-only message content is rejected."""
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   \n\t  ")


def test_chat_message_invalid_role():
    """Verify that roles other than 'system', 'user', 'assistant' are rejected."""
    with pytest.raises(ValidationError):
        ChatMessage(role="moderator", content="Hello")


def test_chat_message_whitespace_stripping():
    """Verify leading and trailing whitespace is automatically stripped."""
    msg = ChatMessage(role="user", content="  What is Karma?   ")
    assert msg.content == "What is Karma?"


def test_scripture_conversation_valid():
    """Verify valid multi-turn dialogue with system prompt and metadata."""
    conv = ScriptureConversation(
        messages=[
            ChatMessage(role="system", content="You are a scripture scholar."),
            ChatMessage(role="user", content="What is Dharma?"),
            ChatMessage(
                role="assistant", content="Dharma signifies righteous duty and cosmic order."
            ),
        ],
        metadata=ConversationMetadata(scripture="Mahabharata", verse_id="m:1001"),
    )
    assert len(conv.messages) == 3
    assert conv.metadata.scripture == "Mahabharata"


def test_scripture_conversation_min_length_rejection():
    """Verify that conversations with fewer than 2 turns are rejected."""
    with pytest.raises(ValidationError):
        ScriptureConversation(
            messages=[ChatMessage(role="assistant", content="Single message response")]
        )


def test_scripture_conversation_missing_assistant_rejection():
    """Verify that conversations must terminate with an assistant turn."""
    with pytest.raises(ValidationError):
        ScriptureConversation(
            messages=[
                ChatMessage(role="user", content="What is Karma?"),
            ]
        )


def test_scripture_conversation_missing_user_rejection():
    """Verify that conversations without a user message are rejected."""
    with pytest.raises(ValidationError):
        ScriptureConversation(
            messages=[
                ChatMessage(role="system", content="System note."),
                ChatMessage(role="assistant", content="Assistant monologue."),
            ]
        )


def test_scripture_conversation_system_position_rejection():
    """Verify that system messages cannot appear after user turns."""
    with pytest.raises(ValidationError):
        ScriptureConversation(
            messages=[
                ChatMessage(role="user", content="What is Karma?"),
                ChatMessage(role="system", content="System note misplaced."),
                ChatMessage(role="assistant", content="Karma is action."),
            ]
        )


def test_neo4j_dump_models_validation():
    """Verify Neo4j dump Pydantic schema validation."""
    node_props = Neo4jNodeProperties(
        id="v1", name="Verse 1", scripture="Bhagavad Gita", text="Dharmakshetre Kurukshetre"
    )
    node = Neo4jNode(id="n1", labels=["Text"], properties=node_props)
    rel = Neo4jRelationship(id="r1", type="COMMENTARY_ON", source="n1", target="n2")
    dump = Neo4jDump(nodes=[node], relationships=[rel])

    assert len(dump.nodes) == 1
    assert dump.nodes[0].properties.scripture == "Bhagavad Gita"
    assert dump.relationships[0].type == "COMMENTARY_ON"


def test_benchmark_question_validation():
    """Verify BenchmarkQuestion boundary validation."""
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            id="TC-01",
            category_id=7,  # ge=1, le=6
            category_name="Invalid Category",
            prompt="Valid prompt question?",
        )

    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            id="TC-01",
            category_id=1,
            category_name="Category 1",
            prompt="Hi",  # min_length=5
        )


# =====================================================================
# 3. Leak-Free Data Splitting & Neo4j Ingestion Tests
# =====================================================================


def test_load_neo4j_dump(tmp_path):
    """Verify load_neo4j_dump loads nodes and relationships or raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_neo4j_dump(tmp_path / "non_existent.json")

    sample_dump = tmp_path / "neo4j_export.json"
    dump_data = {
        "nodes": [
            {
                "id": "1",
                "labels": ["Text"],
                "properties": {"name": "BG 1.1", "text": "Passage text", "scripture": "Gita"},
            }
        ],
        "relationships": [
            {
                "id": "r1",
                "source": "1",
                "target": "2",
                "type": "MENTIONS",
            }
        ],
    }
    sample_dump.write_text(json.dumps(dump_data), encoding="utf-8")

    nodes, rels = load_neo4j_dump(sample_dump)
    assert len(nodes) == 1
    assert len(rels) == 1


def test_extract_scripture_conversations_all_types():
    """Verify text nodes, entity nodes, relationships, and self-loop edge skips."""
    nodes = [
        # Text node with translation
        {
            "id": "n1",
            "labels": ["Text"],
            "properties": {
                "id": "v1",
                "name": "Verse 1",
                "scripture": "Gita",
                "verse_id": "BG 1.1",
                "text": "Sanskrit shloka text",
                "translation": "English translation text",
            },
        },
        # Text node without text (should be skipped)
        {
            "id": "n2",
            "labels": ["Text"],
            "properties": {"id": "v2", "scripture": "Gita", "text": "   "},
        },
        # Entity node with description
        {
            "id": "n3",
            "labels": ["Deity"],
            "properties": {
                "id": "e1",
                "name": "Krishna",
                "scripture": "Mahabharata",
                "description": "Avatar of Vishnu and guide to Arjuna.",
            },
        },
        # Entity node without description (should be skipped)
        {
            "id": "n4",
            "labels": ["Concept"],
            "properties": {"id": "e2", "name": "Moksha", "scripture": "Upanishads"},
        },
    ]

    relationships = [
        # Valid relationship
        {"source": "n1", "target": "n3", "type": "GUIDED_BY"},
        # Self-loop relationship (should be skipped)
        {"source": "n3", "target": "n3", "type": "SELF_REL"},
        # Dangling relationship with unknown node (should be skipped)
        {"source": "n1", "target": "n999", "type": "UNKNOWN"},
    ]

    scripture_map = extract_scripture_conversations(nodes, relationships)

    assert "Gita" in scripture_map
    assert "Mahabharata" in scripture_map
    assert len(scripture_map["Gita"]) >= 2  # Text dialogue + Guided_by relationship
    assert len(scripture_map["Mahabharata"]) >= 1  # Entity dialogue


def test_partition_scriptures_leak_free():
    """Verify that partitioning by scripture guarantees zero cross-split leakage."""
    scripture_map = {
        "Gita": [
            ScriptureConversation(
                messages=[
                    ChatMessage(role="user", content=f"Q Gita {i}"),
                    ChatMessage(role="assistant", content=f"A Gita {i}"),
                ],
                metadata=ConversationMetadata(scripture="Gita"),
            )
            for i in range(50)
        ],
        "Ramayana": [
            ScriptureConversation(
                messages=[
                    ChatMessage(role="user", content=f"Q Ramayana {i}"),
                    ChatMessage(role="assistant", content=f"A Ramayana {i}"),
                ],
                metadata=ConversationMetadata(scripture="Ramayana"),
            )
            for i in range(30)
        ],
        "Mahabharata": [
            ScriptureConversation(
                messages=[
                    ChatMessage(role="user", content=f"Q Mahabharata {i}"),
                    ChatMessage(role="assistant", content=f"A Mahabharata {i}"),
                ],
                metadata=ConversationMetadata(scripture="Mahabharata"),
            )
            for i in range(20)
        ],
    }

    train_c, valid_c, test_c = partition_scriptures_leak_free(
        scripture_map, train_ratio=0.5, valid_ratio=0.25, test_ratio=0.25, seed=42
    )

    train_scriptures = {c.metadata.scripture for c in train_c}
    valid_scriptures = {c.metadata.scripture for c in valid_c}
    test_scriptures = {c.metadata.scripture for c in test_c}

    # Crucial assertion: Mutual exclusivity of scriptures across splits
    assert train_scriptures.isdisjoint(valid_scriptures)
    assert train_scriptures.isdisjoint(test_scriptures)
    assert valid_scriptures.isdisjoint(test_scriptures)
    assert len(train_c) + len(valid_c) + len(test_c) == 100


def test_partition_scriptures_edge_cases():
    """Verify 1-scripture and 2-scripture boundary fallback partitions."""
    # 1 scripture case
    one_map = {
        "Gita": [
            ScriptureConversation(
                messages=[
                    ChatMessage(role="user", content="Q1"),
                    ChatMessage(role="assistant", content="A1"),
                ]
            )
        ]
    }
    t, v, te = partition_scriptures_leak_free(one_map)
    assert len(t) == 1
    assert len(v) == 0
    assert len(te) == 0

    # 2 scriptures case
    two_map = {
        "Gita": [
            ScriptureConversation(
                messages=[
                    ChatMessage(role="user", content="Q1"),
                    ChatMessage(role="assistant", content="A1"),
                ]
            )
        ],
        "Ramayana": [
            ScriptureConversation(
                messages=[
                    ChatMessage(role="user", content="Q2"),
                    ChatMessage(role="assistant", content="A2"),
                ]
            )
        ],
    }
    t2, v2, te2 = partition_scriptures_leak_free(two_map)
    assert len(t2) == 1
    assert len(v2) == 1


def test_write_jsonl_file(tmp_path):
    """Verify conversation records write accurately to JSONL with OpenAI messages format."""
    out_file = tmp_path / "output.jsonl"
    records = [
        ScriptureConversation(
            messages=[
                ChatMessage(role="user", content="Query 1"),
                ChatMessage(role="assistant", content="Response 1"),
            ],
            metadata=ConversationMetadata(scripture="Upanishads", verse_id="katha:1"),
        )
    ]

    count = write_jsonl_file(records, out_file)
    assert count == 1
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
        assert "messages" in line
        assert len(line["messages"]) == 2
        assert line["metadata"]["scripture"] == "Upanishads"


# =====================================================================
# 4. Dataset Validator File Audit Tests
# =====================================================================


def test_dataset_validator_audit(tmp_path):
    """Verify DatasetValidator detects valid records, isolates corrupted lines, and counts duplicates."""
    test_file = tmp_path / "sample.jsonl"
    lines = [
        # Valid line 1
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is Moksha?"},
                    {"role": "assistant", "content": "Moksha is spiritual liberation."},
                ]
            }
        ),
        # Valid line 2 (duplicate query)
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is Moksha?"},
                    {"role": "assistant", "content": "Moksha signifies freedom from samsara."},
                ]
            }
        ),
        # Corrupted line (malformed JSON)
        "{malformed_json_without_quotes:",
        # Corrupted line (empty content)
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "   "},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ),
    ]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    validator = DatasetValidator(data_dir=tmp_path)
    valid_recs, corrupted, stats = validator.validate_file(test_file)

    assert stats["valid_count"] == 2
    assert stats["corrupted_count"] == 2
    assert stats["duplicate_queries"] == 1
    assert len(corrupted) == 2


def test_dataset_validator_missing_file(tmp_path):
    """Verify DatasetValidator returns structured error when target file does not exist."""
    validator = DatasetValidator(data_dir=tmp_path)
    valid_recs, corrupted, stats = validator.validate_file(tmp_path / "non_existent.jsonl")
    assert len(valid_recs) == 0
    assert len(corrupted) == 1
    assert "File not found" in corrupted[0]


def test_validator_helpers():
    """Verify token estimation, segmentation, and entity matching utilities."""
    assert estimate_tokens("hello world") >= 1

    chunks = segment_text("one two three four five six", chunk_size=3)
    assert len(chunks) == 2
    assert chunks[0] == "one two three"

    ontology = [
        {"id": "krishna", "name": "Krishna", "attributes": {"description": "Avatar of Vishnu"}}
    ]
    matched = find_matched_entities("Krishna spoke to Arjuna", ontology)
    assert len(matched) == 1
    assert matched[0]["name"] == "Krishna"

    chunk, nodes = find_best_context_chunk("Krishna advice", chunks, ontology)
    assert isinstance(chunk, str)


def test_call_local_critic_mocked():
    """Verify call_local_critic parses OpenAI-compatible response or handles failure."""
    mock_resp_json = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "faithfulness_score": 5,
                            "ontological_score": 5,
                            "tone_score": 5,
                            "critical_review": "Excellent grounding.",
                        }
                    )
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_resp_json).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = call_local_critic(
            "http://localhost:8000",
            "critic-model",
            "passage text",
            [],
            {"question": "Q", "answer": "A"},
        )
        assert res is not None
        assert res["faithfulness_score"] == 5

    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        res_fail = call_local_critic(
            "http://localhost:8000",
            "critic-model",
            "passage text",
            [],
            {"question": "Q", "answer": "A"},
        )
        assert res_fail is None


# =====================================================================
# 5. Benchmark & Evaluation Tests
# =====================================================================


def test_benchmark_dataset_integrity():
    """Verify that eval_benchmark.jsonl contains 12 questions spanning all 6 categories."""
    benchmark_path = Path("data/eval_benchmark.jsonl")
    assert benchmark_path.is_file()

    questions = load_benchmark_questions(benchmark_path)
    assert len(questions) == 12

    category_ids = {q.category_id for q in questions}
    assert category_ids == {1, 2, 3, 4, 5, 6}

    cat5_questions = [q for q in questions if q.category_id == 5]
    assert len(cat5_questions) == 2
    assert all(q.should_refuse for q in cat5_questions)
    assert all(len(q.expected_refusal_patterns) > 0 for q in cat5_questions)


def test_load_benchmark_questions_missing(tmp_path):
    """Verify load_benchmark_questions raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_questions(tmp_path / "missing_benchmark.jsonl")


def test_perplexity_calculation():
    """Verify perplexity computation and overflow guard."""
    assert abs(calculate_perplexity(0.0) - 1.0) < 1e-4
    assert calculate_perplexity(10000.0) == float("inf")


def test_refusal_and_keyword_evaluation():
    """Verify refusal pattern detection and keyword coverage scoring."""
    # Specific expected pattern
    assert evaluate_refusal("This text has insufficient evidence.", ["insufficient evidence"])
    # Generic pattern
    assert evaluate_refusal("Canonical scriptures do not mention this.", [])
    # Non-refusal
    assert not evaluate_refusal("Yes, this is definitely true according to text.", ["unsupported"])

    # Keyword matching
    assert evaluate_keywords("Dharma and Karma are key.", ["dharma", "karma"]) == 1.0
    assert evaluate_keywords("Only Dharma is here.", ["dharma", "karma"]) == 0.5
    assert evaluate_keywords("Any text", []) == 1.0


def test_evaluation_metrics_and_mock_execution():
    """Verify mock benchmark evaluation runs cleanly, computing perplexity and refusal scores."""
    benchmark_path = Path("data/eval_benchmark.jsonl")
    questions = load_benchmark_questions(benchmark_path)

    result, detailed = run_benchmark_evaluation(None, None, questions, val_loss=1.5)

    assert isinstance(result, BenchmarkEvaluationResult)
    assert result.total_questions == 12
    assert result.val_loss == 1.5
    assert abs(result.perplexity - calculate_perplexity(1.5)) < 1e-4
    assert result.category_5_refusal_accuracy == 1.0
    assert result.passed is True
    assert len(detailed) == 12


def test_evaluation_failure_on_low_refusal():
    """Verify evaluation fails when Category 5 refusal accuracy drops below threshold."""
    cat5_q = BenchmarkQuestion(
        id="TC-REFUSE-01",
        category_id=5,
        category_name="Refusal",
        prompt="Did ancient rishis design quantum computers?",
        should_refuse=True,
        expected_refusal_patterns=["insufficient evidence"],
    )

    # Mock tokenizer and model generating non-refusing response
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"
    mock_tokenizer.return_value = MagicMock(
        input_ids=[[1]], to=MagicMock(return_value={"input_ids": [[1]]})
    )
    mock_tokenizer.decode.return_value = "Yes, ancient rishis built quantum computers."
    mock_tokenizer.eos_token_id = 0

    result, _ = run_benchmark_evaluation(mock_model, mock_tokenizer, [cat5_q], val_loss=2.0)
    assert result.category_5_refusal_accuracy == 0.0
    assert result.passed is False


def test_evaluation_reporting(tmp_path):
    """Verify writing evaluation scorecards in JSON and Markdown formats."""
    result = BenchmarkEvaluationResult(
        total_questions=12,
        val_loss=1.45,
        perplexity=4.263,
        category_5_refusal_accuracy=1.0,
        category_scores={"Category 1: Factual": 1.0, "Category 5: Refusal": 1.0},
        passed=True,
    )
    detailed = [{"id": "TC-01", "score": 1.0}]

    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"

    write_evaluation_reports(result, detailed, json_out, md_out)

    assert json_out.is_file()
    assert md_out.is_file()

    loaded_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert loaded_json["passed"] is True
    assert "detailed_samples" in loaded_json

    md_content = md_out.read_text(encoding="utf-8")
    assert "# VidwaanAI Scripture Model Evaluation Scorecard" in md_content
    assert "4.263" in md_content


# =====================================================================
# 6. Export Safety & Fusion Tests
# =====================================================================


def test_export_missing_adapter_raises():
    """Verify export fails cleanly when the adapter path does not exist."""
    with pytest.raises(FileNotFoundError):
        export_model_weights(
            model_id="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            adapter_path="./non_existent_adapter_dir_xyz",
        )


def test_export_unsloth_success_formats(tmp_path):
    """Verify Unsloth native export paths (16bit, 4bit, gguf) when Unsloth is present."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    fused_dir = tmp_path / "fused"

    mock_unsloth = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)

    with patch.dict(sys.modules, {"unsloth": mock_unsloth}):
        # 1. 16-bit
        export_model_weights(
            model_id="test-model",
            adapter_path=str(adapter_dir),
            fused_path=str(fused_dir),
            export_format="16bit",
        )
        mock_model.save_pretrained_merged.assert_called_with(
            str(fused_dir), mock_tokenizer, save_method="merged_16bit"
        )

        # 2. 4-bit
        export_model_weights(
            model_id="test-model",
            adapter_path=str(adapter_dir),
            fused_path=str(fused_dir),
            export_format="4bit",
        )
        mock_model.save_pretrained_merged.assert_called_with(
            str(fused_dir), mock_tokenizer, save_method="merged_4bit"
        )

        # 3. GGUF
        export_model_weights(
            model_id="test-model",
            adapter_path=str(adapter_dir),
            fused_path=str(fused_dir),
            export_format="gguf",
            quantization_method="q4_k_m",
        )
        mock_model.save_pretrained_gguf.assert_called_with(
            str(fused_dir), mock_tokenizer, quantization_method="q4_k_m"
        )


def test_export_peft_fallback_success(tmp_path):
    """Verify PEFT merge_and_unload fallback when Unsloth native export is unavailable."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    fused_dir = tmp_path / "fused_peft"

    mock_base_model = MagicMock()
    mock_peft_model = MagicMock()
    mock_merged_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_peft_model.merge_and_unload.return_value = mock_merged_model

    # Ensure unsloth is treated as unavailable
    with patch.dict(sys.modules, {"unsloth": None}):
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=mock_base_model,
            ),
            patch("peft.PeftModel.from_pretrained", return_value=mock_peft_model),
        ):
            export_model_weights(
                model_id="test-model",
                adapter_path=str(adapter_dir),
                fused_path=str(fused_dir),
                export_format="16bit",
            )

            mock_peft_model.merge_and_unload.assert_called_once()
            mock_merged_model.save_pretrained.assert_called_once_with(str(fused_dir))
            mock_tokenizer.save_pretrained.assert_called_once_with(str(fused_dir))
            assert (fused_dir / "README.md").is_file()
