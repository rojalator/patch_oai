=== patch_openai.py ===

(You just need to download the Python file, you haven't got to 'git' or anything fancy if you don't know how: just click on its name here, use the 'Raw' button and save the displayed file.)

This code (in the Python file) applies a monkey-patch an OpenAI client to add the responses.create() method by converting calls to chat.completions.create()
(So it does GATHER -> DESPATCH -> SCATTER, in effect). 

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

swap_role_names is a dictionary of pairs of strings, where values should be swapped. For example: {'developer': 'system'} indicates that we want to swap 'developer' as a role to 'system' instead. This is because some installations expect 'developer' to be a role, while others expect 'system' to be a role. For example, if you need to swap change 'developer' to 'system' you do this:
```
    monkey_patch_responses_api(llm, {'developer': 'system'})

```

Remember: It's a fake endpoint so you are not really talking to responses.create() but to chat.create()
The original code was created by claudeai but I've modified to work better. Well, to work at all but it saved a
lot of typing initially!


=== render_mermaid.js ===

The ViolentMonkey script (it's an add-on to WaterFox / FireFox that allows Javascript to be inserted into a page) renders Mermaid diagrams in llama.cpp's built-in web UI, with click to zoom and raw/rendered toggle. It waits for the diagram text to be fully emitted before trying to render (prevents a lot of flickering)
