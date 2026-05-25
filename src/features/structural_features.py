"""
Structural feature extraction for earnings call data.

Computes per-call structural features from segments:
    - qa_duration_ratio: Q&A time / total call time
    - avg_answer_length: mean word count of management answers
    - avg_question_length: mean word count of analyst questions
    - response_latency: mean gap between question end and answer start (seconds)
    - turn_taking_frequency: number of speaker switches per minute
    - speaker_diversity: number of unique speakers
    - qa_segment_count: number of Q&A segments
    - prepared_segment_count: number of prepared remark segments

Output: data/processed/structural_features.parquet
Phase 3 - Task 3.D.2
"""

import logging
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_structural_features(segments: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-call structural features from a segments DataFrame.

    Expects columns: call_id, segment_id, segment_type, speaker_role,
                     speaker_name, text, start_time, end_time.
    """
    call_ids = segments["call_id"].unique().sort().to_list()
    results = []

    for call_id in call_ids:
        call_df = segments.filter(pl.col("call_id") == call_id).sort("segment_id")

        # Total segments
        n_total = len(call_df)

        # Segment type counts
        type_counts = call_df.group_by("segment_type").len()
        n_prepared = type_counts.filter(
            pl.col("segment_type") == "prepared_remarks"
        )["len"].sum()
        n_questions = type_counts.filter(
            pl.col("segment_type") == "analyst_question"
        )["len"].sum()
        n_answers = type_counts.filter(
            pl.col("segment_type") == "management_answer"
        )["len"].sum()
        n_operator = type_counts.filter(
            pl.col("segment_type") == "operator_transition"
        )["len"].sum()

        # Word counts per segment type
        texts = call_df["text"].to_list()
        types = call_df["segment_type"].to_list()
        word_counts = [len(t.split()) for t in texts]

        answer_words = [wc for wc, st in zip(word_counts, types) if st == "management_answer"]
        question_words = [wc for wc, st in zip(word_counts, types) if st == "analyst_question"]

        avg_answer_length = sum(answer_words) / max(len(answer_words), 1)
        avg_question_length = sum(question_words) / max(len(question_words), 1)

        # Q&A duration ratio (using segment counts as proxy if no timestamps)
        has_timestamps = (
            "start_time" in call_df.columns
            and "end_time" in call_df.columns
            and call_df["start_time"].null_count() < len(call_df)
        )

        if has_timestamps:
            # Use actual timestamps
            qa_df = call_df.filter(
                pl.col("segment_type").is_in(["analyst_question", "management_answer"])
            )
            total_df = call_df.filter(pl.col("start_time").is_not_null())

            qa_duration = (qa_df["end_time"] - qa_df["start_time"]).sum()
            total_duration = (total_df["end_time"] - total_df["start_time"]).sum()
            qa_duration_ratio = float(qa_duration / max(total_duration, 1e-6))

            # Response latency: gap between question end and next answer start
            latencies = []
            prev_was_question = False
            prev_end_time = 0.0
            for row in call_df.iter_rows(named=True):
                if row["segment_type"] == "management_answer" and prev_was_question:
                    if row["start_time"] is not None and prev_end_time is not None:
                        latency = row["start_time"] - prev_end_time
                        if 0 <= latency < 30:  # Sanity check
                            latencies.append(latency)

                prev_was_question = row["segment_type"] == "analyst_question"
                if row["end_time"] is not None:
                    prev_end_time = row["end_time"]

            response_latency = sum(latencies) / max(len(latencies), 1)

            # Turn-taking frequency
            total_minutes = total_duration / 60.0 if total_duration else 1.0
        else:
            # Use word counts as proxy
            qa_words = sum(answer_words) + sum(question_words)
            total_words = sum(word_counts)
            qa_duration_ratio = qa_words / max(total_words, 1)
            response_latency = 0.0
            total_minutes = total_words / 150.0  # ~150 words per minute

        # Turn-taking: count speaker changes
        speakers = call_df["speaker_name"].to_list() if "speaker_name" in call_df.columns else []
        speaker_changes = 0
        for i in range(1, len(speakers)):
            if speakers[i] != speakers[i - 1]:
                speaker_changes += 1

        turn_taking_frequency = speaker_changes / max(total_minutes, 1.0)

        # Speaker diversity
        unique_speakers = call_df["speaker_name"].n_unique() if "speaker_name" in call_df.columns else 0

        results.append({
            "call_id": call_id,
            "qa_duration_ratio": float(qa_duration_ratio),
            "avg_answer_length": float(avg_answer_length),
            "avg_question_length": float(avg_question_length),
            "response_latency": float(response_latency),
            "turn_taking_frequency": float(turn_taking_frequency),
            "speaker_diversity": int(unique_speakers),
            "n_total_segments": int(n_total),
            "n_prepared_remarks": int(n_prepared),
            "n_analyst_questions": int(n_questions),
            "n_management_answers": int(n_answers),
            "n_operator_transitions": int(n_operator),
        })

    df_result = pl.DataFrame(results)
    logger.info("Computed structural features for %d calls", len(df_result))
    return df_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    project_root = Path(__file__).resolve().parent.parent.parent
    segments_path = project_root / "data" / "processed" / "segments_labelled.parquet"
    output_path = project_root / "data" / "processed" / "structural_features.parquet"

    if not segments_path.exists():
        # Fallback to segments.parquet
        segments_path = project_root / "data" / "processed" / "segments.parquet"

    if not segments_path.exists():
        logger.error("No segments file found.")
        return

    df = pl.read_parquet(segments_path)
    logger.info("Loaded %d segments from %s", len(df), segments_path)

    df_features = compute_structural_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved structural features to: %s", output_path)


if __name__ == "__main__":
    main()
