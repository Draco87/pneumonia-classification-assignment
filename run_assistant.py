import subprocess
import sys

from assistant.assistant import ask_assistant


def run_cli():
    print("\nPneumonia Classification Project Assistant")
    print("Interface: Command Line")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        answer = ask_assistant(question)

        print(f"\nAssistant: {answer}\n")


def run_streamlit():
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py"
    ])


if __name__ == "__main__":

    print("\nChoose assistant interface:")
    print("0 - Command-line interface")
    print("1 - Streamlit web interface")

    choice = input("\nEnter 0 or 1: ").strip()

    if choice == "0":
        run_cli()

    elif choice == "1":
        run_streamlit()

    else:
        print("Invalid option. Please enter 0 or 1.")