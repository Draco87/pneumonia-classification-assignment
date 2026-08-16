import os

from ollama import chat


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "phi3:mini"


# ---------------------------------------------------------
# Load project context
# ---------------------------------------------------------

def load_project_context():

    context_path = os.path.join(
        os.path.dirname(__file__),
        "project_context.txt"
    )

    with open(
        context_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


PROJECT_CONTEXT = load_project_context()


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = f"""
You are an AI assistant for a machine-learning project involving
pneumonia classification from pediatric chest X-rays.

Answer using only the supplied project information.

PROJECT INFORMATION
-------------------
{PROJECT_CONTEXT}

INSTRUCTIONS
------------
1. Do not invent experimental results or metrics.
2. Distinguish training, validation, and test results.
3. For model comparisons, use Section 10: MODEL COMPARISON — TEST SET.
4. Before saying one numerical metric is higher or lower, compare the
   two reported values explicitly.
5. Do not declare a model universally superior when metrics show a
   trade-off. Explain the trade-off.
6. If information is absent, say it was not evaluated or recorded.
7. Do not claim that the classifier provides a medical diagnosis.
8. Keep answers concise unless more detail is requested.
"""


# ---------------------------------------------------------
# Ask local assistant
# ---------------------------------------------------------

def ask_assistant(question):

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response.message.content


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nPneumonia Classification Project Assistant"
    )

    print(
        f"Local model: {MODEL_NAME}"
    )

    print(
        "Type 'exit' to stop.\n"
    )


    while True:

        question = input("You: ").strip()

        if question.lower() in {
            "exit",
            "quit"
        }:
            break

        if not question:
            continue

        try:

            answer = ask_assistant(
                question
            )

            print(
                f"\nAssistant: {answer}\n"
            )

        except Exception as error:

            print(
                f"\nError: {error}"
            )

            print(
                "Make sure Ollama is running and "
                f"{MODEL_NAME} has been downloaded.\n"
            )