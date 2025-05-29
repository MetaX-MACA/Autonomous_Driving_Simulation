from openai import OpenAI


use_llm_query = True

token = "your-github-token"
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"

client = OpenAI(
    api_key=token,
    base_url=endpoint, 
)


def llm_query(msg, stream=False, model_name=model_name, ):
    completion = client.chat.completions.create(
        model=model_name,
        messages=msg,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=stream
    )

    if not stream:
        res = completion.choices[0].message.content
    else:
        res = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                res += chunk.choices[0].delta.content
    return res