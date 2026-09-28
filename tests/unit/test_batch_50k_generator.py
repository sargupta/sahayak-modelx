import os
import shutil
import pytest
from synthetictutor.pipeline.batch_50k_generator import (
    Batch50kGenerator,
    GenerationConfig50k,
    STAGE_TARGETS,
    TASK_TARGETS
)
from synthetictutor.pipeline.intent_generator import UserIntent


@pytest.fixture
def temp_50k_env(tmp_path):
    out_dir = tmp_path / "test_50k_out"
    out_dir.mkdir()
    db_path = out_dir / "test_checkpoint.db"
    
    config = GenerationConfig50k(
        total_target=100,
        output_dir=str(out_dir),
        checkpoint_db=str(db_path),
        shard_size=10,
        source_pool_path="datasets/final_grounding_corpus/grounding_chunks_text_only.jsonl",
        locale_path="locale.json",
        min_fidelity_score=1
    )
    generator = Batch50kGenerator(config)
    yield generator, str(out_dir)


def test_50k_quota_definitions():
    """Validates that stage and task target distributions sum to 100% and 50,000."""
    total_stage_quota = sum(d["quota"] for d in STAGE_TARGETS.values())
    total_stage_share = sum(d["share"] for d in STAGE_TARGETS.values())
    assert total_stage_quota == 50000
    assert abs(total_stage_share - 1.0) < 1e-5

    total_task_quota = sum(d["quota"] for d in TASK_TARGETS.values())
    total_task_share = sum(d["share"] for d in TASK_TARGETS.values())
    assert total_task_quota == 50000
    assert abs(total_task_share - 1.0) < 1e-5


def test_batch_generator_synthesis_and_checkpoint(temp_50k_env):
    """Tests single record generation, gate scoring, validation, and db persistence."""
    generator, out_dir = temp_50k_env
    
    intent = UserIntent(
        query="অষ্টম শ্রেণির গণিত বইয়ের সমীকরণ গঠন অধ্যায়ের ওপর একটি সহজ পাঠপরিকল্পনা তৈরি করুন।",
        target_grade=8,
        target_subject="Mathematics",
        target_topic="অধ্যায় ২২: সমীকরণ গঠন ও সমাধান",
        task_type="LESSON_PLAN",
        requester_role="TEACHER",
        instructional_target="STUDENT",
        audience="STUDENT"
    )

    fake_chunk = {
        "chunk_id": "test_chunk_001",
        "book_id": "ganit_probha_8",
        "board": "WBBSE",
        "grade": 8,
        "subject": "Mathematics",
        "chapter": "অধ্যায় ২২",
        "topic": "সমীকরণ গঠন ও সমাধান",
        "source_text": "একটি সমীকরণের উভয় পক্ষে সমান সংখ্যা যোগ বা বিয়োগ করলে সমতা বজায় থাকে।"
    }
    generator.retriever.retrieve_multi_chunk = lambda *args, **kwargs: [fake_chunk]
    generator.retriever.retrieve = lambda *args, **kwargs: [fake_chunk]

    def mock_llm_fn(intent_obj, chunks, locale):
        return (
            "শিক্ষক মহাশয়, পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদের অষ্টম শ্রেণির গণিত পাঠ্যক্রমের 'সমীকরণ গঠন ও সমাধান' "
            "অধ্যায়টির জন্য একটি ৪০ মিনিটের পাঠপরিকল্পনা নিচে দেওয়া হলো।\n\n"
            "১. শিখন উদ্দেশ্য: শিক্ষার্থীরা চলরাশি ও সমীকরণের সম্পর্ক বুঝতে পারবে।\n"
            "২. প্রয়োজনীয় উপকরণ: ব্ল্যাকবোর্ড, চক ও বাস্তব উদাহরণ।"
        )

    record = generator.synthesize_record(intent, "test_rec_001", llm_response_fn=mock_llm_fn)
    assert record is not None
    assert record["id"] == "test_rec_001"
    assert record["stage"] == "MIDDLE_6_8"
    assert "metadata" in record
    assert "provenance" in record["metadata"]
    assert record["metadata"]["provenance"]["generator_model"] == "gemini-1.5-pro-002"

    # Verify db status
    assert generator.get_completed_count() == 1
    stage_counts = generator.get_stage_counts()
    assert stage_counts.get("MIDDLE_6_8") == 1


def test_shard_export(temp_50k_env):
    """Tests JSONL shard exporting from SQLite checkpoint database."""
    generator, out_dir = temp_50k_env

    # Seed fake validated records into DB
    for i in range(15):
        dummy_rec = {
            "id": f"rec_{i:03d}",
            "stage": "PRIMARY_1_5",
            "grade": 3,
            "subject": "Mathematics",
            "task_family": "CONCEPT_EXPLANATION",
            "curriculum_fidelity_score": 5,
            "user_prompt": f"প্রশ্ন {i}",
            "assistant_response": f"উত্তর {i}"
        }
        generator.save_record(dummy_rec, validation_passed=True)

    exported_files = generator.export_shards(shard_size=10)
    assert len(exported_files) == 2  # 10 in shard 1, 5 in shard 2
    assert os.path.exists(exported_files[0])
    assert os.path.exists(exported_files[1])
