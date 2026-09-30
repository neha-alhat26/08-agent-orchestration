
import asyncio
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from agent_framework import Message
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import SequentialBuilder
from azure.identity import AzureCliCredential


load_dotenv()


def clean_response(text: str) -> str:
    """Keep only the first sentence to avoid repeated or lengthy output."""
    text = re.sub(r"\s+", " ", text).strip()

    # Find the end of the first sentence
    match = re.search(r"^.*?[.!?](?:\s|$)", text)

    if match:
        return match.group(0).strip()

    # Fallback if the response has no sentence-ending punctuation
    return " ".join(text.split()[:35])


def normalize_classification(text: str) -> str:
    """Normalize the classifier response to one expected label."""
    text = text.strip().lower()

    if "feature request" in text:
        return "Feature request"
    if "negative" in text:
        return "Negative"
    if "positive" in text:
        return "Positive"

    return "Uncategorized"


async def main():
    # --------------------------------------------------
    # 1. Agent instructions
    # --------------------------------------------------

    summarizer_instructions = """
    Summarize the customer's feedback in one short sentence.
    Keep it neutral, concise, and factual.
    Do not add recommendations or explanations.
    """

    classifier_instructions = """
    Classify the customer's feedback into exactly one of these labels:
    Positive, Negative, or Feature request.

    Return only the label. Do not include an explanation.
    """

    action_instructions = """
    You are a customer feedback action recommendation agent.

    Based on the customer's feedback, summary, and classification,
    recommend exactly ONE next action.

    Rules:
    - Output one short sentence only.
    - Do not repeat the recommendation.
    - Do not include headings or explanations.
    - Stop after the one sentence.
    """

    # --------------------------------------------------
    # 2. Create Azure AI chat client
    # --------------------------------------------------

    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    model = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

    if not endpoint or not model:
        raise ValueError(
            "Set AZURE_AI_PROJECT_ENDPOINT and "
            "AZURE_AI_MODEL_DEPLOYMENT_NAME in your .env file."
        )

    credential = AzureCliCredential()

    chat_client = FoundryChatClient(
        credential=credential,
        project_endpoint=endpoint,
        model=model,
    )

    # --------------------------------------------------
    # 3. Create agents
    # --------------------------------------------------

    summarizer_agent = chat_client.as_agent(
        name="summarizer",
        instructions=summarizer_instructions,
    )

    classifier_agent = chat_client.as_agent(
        name="classifier",
        instructions=classifier_instructions,
    )

    action_agent = chat_client.as_agent(
        name="action",
        instructions=action_instructions,
    )

    # --------------------------------------------------
    # 4. Load customer feedback
    # --------------------------------------------------

    file_path = Path(__file__).parent.parent / "feedback.json"

    with open(file_path, "r", encoding="utf-8") as file:
        feedback_data = json.load(file)

    if not isinstance(feedback_data, list):
        raise ValueError("feedback.json must contain a JSON list.")

    # --------------------------------------------------
    # 5. Build sequential multi-agent workflow
    # --------------------------------------------------

    workflow = SequentialBuilder(
        participants=[
            summarizer_agent,
            classifier_agent,
            action_agent,
        ],
        output_from="all",
    ).build()

    all_results = []

    # --------------------------------------------------
    # 6. Process each feedback item
    # --------------------------------------------------

    for item in feedback_data:
        feedback_id = item["id"]
        feedback = item["feedback"]

        print("\n" + "=" * 60)
        print(f"Processing Feedback ID: {feedback_id}")
        print("=" * 60)

        result = await workflow.run(
            f"Customer feedback: {feedback}"
        )

        outputs = result.get_outputs()

        # Reset responses for every feedback item
        agent_responses = {}

        for response in outputs:
            messages = cast(list[Message], response.messages)

            assistant_messages = [
                msg for msg in messages
                if msg.role == "assistant" and msg.text
            ]

            if not assistant_messages:
                continue

            latest_message = assistant_messages[-1]
            agent_name = (latest_message.author_name or "").lower()
            response_text = clean_response(latest_message.text)

            if agent_name in {"summarizer", "classifier", "action"}:
                agent_responses[agent_name] = response_text

                print(f"\n[{agent_name}]")
                print(response_text)

        # Extract the response from each named agent
        summary = agent_responses.get("summarizer", "")
        classification = normalize_classification(
            agent_responses.get("classifier", "")
        )
        recommended_action = agent_responses.get("action", "")

        # Store only complete results
        if summary and recommended_action and classification != "Uncategorized":
            record = {
                "id": feedback_id,
                "feedback": feedback,
                "summary": summary,
                "classification": classification,
                "recommended_action": recommended_action,
            }

            all_results.append(record)
        else:
            print(
                f"\nWarning: Incomplete agent response for "
                f"Feedback ID {feedback_id}."
            )
            print("Collected responses:", agent_responses)

    # --------------------------------------------------
    # 7. Display structured results
    # --------------------------------------------------

    print("\n\n" + "=" * 60)
    print("STRUCTURED FEEDBACK RESULTS")
    print("=" * 60)

    for record in all_results:
        print(json.dumps(record, indent=4, ensure_ascii=False))

    # --------------------------------------------------
    # 8. Feature 1: Feedback analytics
    # --------------------------------------------------

    classification_counts = Counter(
        record["classification"] for record in all_results
    )

    total_analyzed = len(all_results)

    print("\n" + "=" * 60)
    print("FEEDBACK ANALYTICS")
    print("=" * 60)

    print(f"Total feedback received: {len(feedback_data)}")
    print(f"Successfully analyzed: {total_analyzed}")

    for label in ["Positive", "Negative", "Feature request", "Uncategorized"]:
        count = classification_counts.get(label, 0)
        percentage = (
            (count / total_analyzed) * 100
            if total_analyzed else 0
        )

        print(f"{label}: {count} ({percentage:.1f}%)")

    # --------------------------------------------------
    # 9. Save results as JSON
    # --------------------------------------------------

    output_dir = Path(__file__).parent
    json_path = output_dir / "analysis_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nJSON results saved to: {json_path}")

    # --------------------------------------------------
    # 10. Feature 2: Export results as CSV
    # --------------------------------------------------

    csv_path = output_dir / "analysis_results.csv"

    fieldnames = [
        "id",
        "feedback",
        "summary",
        "classification",
        "recommended_action",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"CSV results saved to: {csv_path}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    asyncio.run(main())