=== patch_openai.py ===

(You just need to download the Python file, you haven't got to 'git' or anything fancy if you don't know how: just click on its name here, use the 'Raw' button and save the displayed file.)

This code (in the Python file) applies a monkey-patch an OpenAI client to substitute the ```responses.create()``` method by converting calls to chat.completions.create() so that LLM's that don't yet support the responses end-point will work.
(So it does GATHER -> DESPATCH -> SCATTER, in effect). You use a single line of Python code (well, two if you count the ```import```) to add it and if and when llama.cpp adds the responses end-point you can just comment it out so that OpenAI's module code will be used.

The single line is : ```monkey_patch_responses_api(llm_client)```.

It supports all the calls and parameters... well, you can pass them, I've not tested them all, plus I only use a few.
It's rough and ready but should be useful here and there where there's no support for 'responses'

I use it like this:
```
    from patch_openai import monkey_patch_responses_api
    llm_client = OpenAI(base_url="http://example.com:8080", api_key=LLAMA_API_KEY")
    monkey_patch_responses_api(llm_client)
```
Now ```llm_client``` has a 'responses.create' function that works: for example:
```
    response = llm_client.responses.create(model=LLAMA_MODEL, temperature=2, input="In one sentence, tell me about Stan Laurel")
    print(response.output_text)
```

You can also have it automatically swap role-names if you pass them in a dictionary, thus:

```
    monkey_patch_responses_api(client=llm, role_name_swaps={'developer': 'system'})
```
This reads as "If you encounter 'developer' as a role-name, swap the role-name to 'system'": my llama.cpp doesn't like 'developer' so it's useful for me.

Remember: It's a fake endpoint so you are not really talking to responses.create() but to chat.create(). It should mostly work. Mostly.
The original code was created by claudeai but I've modified to work better. Well, to work at all but it saved a
lot of typing initially!
