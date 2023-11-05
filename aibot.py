# this function detects if the ai chatbot is being called in a text message
import re
import openai
import random
userInput = "@gpt112/generate a indian student with a unibrow"

def detectAI(user_input):
    if re.findall("^@gpt112", userInput):
        # check if command is /generate
        if re.findall("^@gpt112/generate", userInput):
            # if command is properly executed
            if userInput[:16] == '@gpt112/generate':
                return ['/generate', userInput[16:]]
        elif re.findall("^@gpt112/ask", userInput):
            if userInput[:12] == '@gpt112/generate':
                return ['/ask', userInput[12:]]
    else:
        print('Note: The generate command is executed by typing "@aibot/ask" at the start of your text.')

# function is called if detectAI(userInput)[0] is equal to '/generate' and input is detectAI(userInput)[1]
def getBotImage(input):
    openai.api_key = "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"
    response = openai.Image.create(prompt=f"whole photo of {input}", n=1, size="256x256")
    imageUrl = response['data'][0]['url']
    return imageUrl

# function is called if detectAI(userInput)[0] is equal to '/ask' and input is detectAI(userInput)[1]
def getBotAnswer(input):
    openai.api_key = "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"
    response = openai.Completion.create(engine="text-davinci-003", prompt = input, temperature=0.1, max_tokens=500)
    return response.choices[0].text