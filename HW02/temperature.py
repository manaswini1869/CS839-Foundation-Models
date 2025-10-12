from openai import OpenAI

# Replace with your actual API key

client = OpenAI(api_key="")
response = client.responses.create(
    model="gpt-4.1",
    input="Can you recognize the pattern in the following numbers and what are the next two digits: 347101621304057.",
    temperature=0.25
)

print(response.output_text)