Apply a monkey-patch an OpenAI client to add the responses.create() method by converting calls to chat.completions.create()
(So it does GATHER -> DESPATCH -> SCATTER, in effect)

Supports all the parameters... well, you can pass them, I've not tested them all, plus I only use a few.
It's rough and ready but should is useful here and there where there's no support for 'responses'

I use it like this:
```
    from patch_openai import monkey_patch_responses_api
    llm_client = OpenAI(base_url="http://example.com:8080", api_key=LLAMA_API_KEY")
    monkey_patch_responses_api(llm_client)
```
Now llm_client has a 'responses.create' function: for example:
```
    response = llm_client.responses.create(model=LLAMA_MODEL, temperature=2, input="In one sentence, tell me about Stan Laurel")
    print(response.output_text)
```
Rememeber! It's a fake endpoint so you are not really talking to responses.create() but to chat.create()
The original code was created by claudeai but I've modified to work better. Well, to work at all but it saved a
lot of typing initially!
