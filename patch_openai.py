def monkey_patch_responses_api(client):
    """
    Apply a monkey-patch an OpenAI client to add the responses.create() method by converting calls to chat.completions.create()
    (So it does GATHER -> DESPATCH -> SCATTER, in effect)

    Supports all the parameters... well, you can pass them, I've not tested them all, plus I only use a few.
    It's rough and ready but should is useful here and there where there's no support for 'responses'

    I use it like this:

        from patch_openai import monkey_patch_responses_api
        llm_client = OpenAI(base_url="http://example.com:8080", api_key=LLAMA_API_KEY")
        monkey_patch_responses_api(llm_client)

    Now llm_client has a 'responses.create' function: for example:

        response = llm_client.responses.create(model=LLAMA_MODEL, temperature=2, input="In one sentence, tell me about Stan Laurel")
        print(response.output_text)

    Rememeber: It's a fake endpoint so you are not really talking to responses.create() but to chat.create()
    The original code was created by claudeai but I've modified to work better. Well, to work at all but it saved a
    lot of typing initially!

    """

    class ResponsesNamespace:
        def __init__(self, chat_client):
            self.chat_client = chat_client

        def create(self, model, input, modalities=None, instructions=None, metadata=None, temperature=None, top_p=None,
                   max_output_tokens=None, store=None, reasoning_effort=None, **kwargs):
            """
            Convert responses.create() call to chat.completions.create() in a simple way

            Pass:
                model: The model to use
                input: The input text or messages (yes, yes, it clashes with the Python keyword, but's it's the right name)
                modalities: List of modalities (e.g., ["text", "audio"])
                instructions: System instructions to prepend
                metadata: Metadata dictionary for tracking
                temperature: Sampling temperature (0-2)
                top_p: Nucleus sampling parameter
                max_output_tokens: Maximum tokens to generate
                store: Whether to store the conversation
                reasoning_effort: Reasoning effort level (for o1 models: "low", "medium", "high")
                **kwargs: Additional parameters

            Returns:
                A response object matching the Responses API format
            """

            # GATHER...
            messages = []

            # Add system instructions if provided
            if instructions:
                messages.append({"role": "system", "content": instructions})
            # Convert input to messages format
            if isinstance(input, str):
                messages.append({"role": "user", "content": input})
            elif isinstance(input, list):
                # If already in messages format, extend
                messages.extend(input)
            else:
                raise ValueError("'input' value must be a string or list of messages")

            # Build chat completion parameters
            chat_params = {"model": model, "messages": messages}

            # Map Responses API parameters to Chat Completions parameters
            if temperature:
                chat_params["temperature"] = temperature
            if top_p:
                chat_params["top_p"] = top_p
            if max_output_tokens:
                chat_params["max_tokens"] = max_output_tokens
            if modalities:
                chat_params["modalities"] = modalities
            if store:
                chat_params["store"] = store
            # Handle reasoning_effort for o1 models
            if reasoning_effort:
                chat_params["reasoning_effort"] = reasoning_effort
            # Add metadata if provided (pass through)
            if metadata:
                chat_params["metadata"] = metadata

            # Add any additional kwargs
            chat_params.update(kwargs)

            # ...DESPATCH... (Call chat.completions.create)
            chat_response = self.chat_client.chat.completions.create(**chat_params)

            # ...SCATTER (Convert the chat completion response to responses format (sort of...))
            class ResponseObject:
                def __init__(self, chat_resp, meta=None):
                    self.id = chat_resp.id
                    self.model = chat_resp.model
                    self.created = chat_resp.created
                    self.choices = chat_resp.choices
                    self.usage = chat_resp.usage
                    self.metadata = meta
                    self.output_text = None
                    self.output_audio = None
                    # Extract the main output text
                    if chat_resp.choices and len(chat_resp.choices) > 0:
                        self.output_text = chat_resp.choices[0].message.content
                        # Handle audio output if present (for multimodal responses)
                        if hasattr(chat_resp.choices[0].message, 'audio'):
                            self.output_audio = chat_resp.choices[0].message.audio
                    # Keep the original response, just in case
                    self._raw_response = chat_resp

                def __repr__(self):
                    output_preview = self.output_text[:50] if self.output_text else "None"
                    return f"Response(id={self.id}, output_text={output_preview}...)"

                def to_dict(self):
                    """Convert to dictionary format"""
                    result = {"id": self.id, "model": self.model, "created": self.created, "output_text": self.output_text, "usage": self.usage}
                    # I suspect we'd do as well just to put these in anyway, but I'm not sure
                    if self.metadata:
                        result["metadata"] = self.metadata
                    if self.output_audio:
                        result["output_audio"] = self.output_audio
                    return result

            return ResponseObject(chat_response, metadata)

    # Attach the responses namespace to the client
    client.responses = ResponsesNamespace(client)

    return client
