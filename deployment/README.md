# Model Deployment

All the generations from the qualitative analysis in the paper are generated using [FastChat](https://github.com/lm-sys/FastChat)'s deployment platform.
Since FastChat does not officially support Meditron models, there are several steps required to enable the deployment

1. Clone this repository and navigate to the FastChat folder

    ```bash
    git clone https://github.com/lm-sys/FastChat.git
    cd FastChat
    ```

    If you are running on Mac:

    ```bash
    brew install rust cmake
    ```

2. Add a model adapter to `fastchat/model/model_adapter.py`

    ```python
    class MeditronAdapter(BaseModelAdapter):
    """The model adapter for Meditron"""

    def match(self, model_path: str):
        return "meditron" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot") # or meditron_one_shot, meditron
    ```

3. Register some conversation templates in `fastchat/conversation.py`

   ```python
   register_conv_template(
            Conversation(
            name="one_shot_medical",
            system_message=system_msg_meditron,
            roles=("User", "Assistant"),
            messages=(
                ("User","What happens if listeria is left untreated?"),
                (
                    "Assistant",
                    """If listeria infection, or listeriosis, is left untreated, it can lead to severe health complications, particularly in certain high-risk groups. Here's a general overview of the potential outcomes:

    1. Spread of the Infection: Untreated listeriosis can spread beyond the gut to other parts of the body, including the nervous system. This can lead to more severe conditions like meningitis (inflammation of the membranes surrounding the brain and spinal cord) and septicemia (a serious blood infection).
    2. Increased Risk for Certain Groups: Pregnant women, newborns, the elderly, and individuals with weakened immune systems are at a higher risk of severe complications. In pregnant women, listeriosis can lead to miscarriage, stillbirth, premature delivery, or life-threatening infection of the newborn.
    3. Neurological Effects: Listeriosis can cause severe neurological symptoms like headaches, stiff neck, confusion, loss of balance, and convulsions, especially when the infection spreads to the nervous system.
    4. Long-Term Health Impacts: For some, particularly those with pre-existing health conditions or weakened immune systems, the health impacts of listeriosis can be long-lasting and may not fully resolve even with treatment.
    5. Fatalities: In severe cases, particularly among high-risk groups, listeriosis can be fatal.

    It's important to note that early diagnosis and appropriate treatment, typically with antibiotics, can greatly improve the prognosis for those with listeriosis. Therefore, seeking medical attention promptly if listeriosis is suspected is crucial."""
                    )
                ),
                offset=2,
                sep_style=SeparatorStyle.ADD_COLON_SINGLE,
                sep="\n### ",
                stop_str=["###", "<|im_start|>", "<|im_end|>", "$$$", "thank you for your help"]
            )
        )
    ```

    ```python
    register_conv_template(
        Conversation(
            name="meditron",
            system_message=system_msg_meditron,
            system_template="<|im_start|> system\n{system_message}<|im_end|>\n",
            roles=("<|im_start|> user\n", "<|im_start|> assistant\n"),
            sep="<|im_end|>\n",
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            stop_str=["###", "<|im_start|>", "<|im_end|>", "$$$", "thank you for your help"]
        )
    )
    ```

    ```python
    register_conv_template(
        Conversation(
            name="one_shot",
            system_message="A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("Human", "Assistant"),
            messages=(
                (
                    "Human",
                    "Got any creative ideas for a 10 year old’s birthday?",
                ),
                (
                    "Assistant",
                    """Of course! Here are some creative ideas for a 10-year-old's birthday party:
    1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
    2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
    3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
    4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
    5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
    6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
    7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
    8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
    Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
                ),
            ),
            offset=2,
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n### ",
            stop_str=["###", "<|im_start|>", "<|im_end|>", "$$$", "thank you for your help"]
        )
    )
   ```

**For easy use, we have a version with modified content supporting medtrion.**

```bash
git clone https://github.com/eric11eca/FastChat.git
cd FastChat
git checkout meditron-deploy
```

## Serving with Web GUI

To serve using the web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. You can learn more about the architecture [here](docs/server_arch.md).

Here are the commands to follow in your terminal:

### Launch the controller

```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

### Launch the model worker(s)

For example, running `meditron-70b` on 8 GPUs.

```bash
./fastchat.sh \
    -c meditron-70b \
    -m single \
    -n 8
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

### Launch the UI web server

You can use the default gradio web server by running:

By following these steps, you will be able to serve your models using the web UI. You can open your browser and chat with a model now.
If the models do not show up, try to reboot the gradio web server.

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

Or, you can server with OpenAI-compatible APIs provided by FastChat. See documentation [here](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)

To start the API, run the following command:

```bash
python3 -m fastchat.serve.openai_api_server \
    --host localhost \
    --port 8000
```

### Third Party UI

Next, you can use a third-party UI for OpenAI APIs to interact with the model. You will need to specify the host as `http://localhost:8000/v1/chat/completions`.

Here is a UI platform we used, called [BetterChat](https://github.com/ztjhz/)
Get the version we modified for this UI:

```bash
git clone https://github.com/eric11eca/BetterChatGPT
cd BetterChatGPT
git checkout meditron-ui
```

To start the UI, simply run:

```bash
npm install
npm run dev
```

You can access the UI at `http://localhost:5173/`:

<img width=50% src="./../figures/ui-example.png" alt="BetterChatGPT UI Demo" title="BetterChatGPT UI Demo">

First, update the API to be the one provided by FastChat's OpenAI API: `http://localhost:8000/v1/chat/completions`.

<img width=50% src="./../figures/api.png" alt="update api hostname" title="update api hostname">

Next, select the proper models to start interaction:

<img width=50% src="./../figures/model.png" alt="select model" title="select model">
