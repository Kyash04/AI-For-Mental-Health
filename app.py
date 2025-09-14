#Main Python Script

from chatbot import get_response

def main():
    print("Hello there! Im HealthMate, your AI friend. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = get_response(user_input)
        print("Bot: ", response)

if __name__ == "__main__":
    main()