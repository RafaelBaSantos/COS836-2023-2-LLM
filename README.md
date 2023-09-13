# Generative Worldbuilding with Large Language Models

This project introduces a cutting-edge approach to representing fantasy worlds in role-playing games (RPGs) through automated ontology generation. The methodology leverages both intricate chained prompts and a more straightforward single-prompt approach, utilizing the capabilities of the LangChain library combined with the ChatGPT API.

## Features

1. **Interactive Streamlit App:** The main interface is built using Streamlit, allowing users to input different parameters, such as the genre, tone, technology level, inspiration from writers and universes, and many more. These parameters form the foundation of the generated fantasy world.

2. **Ontology Creation:** With the help of the LangChain library, the application can develop an intricate network of places, characters, items, species, and various elements that form the universe of the RPG.

3. **Adaptive Game Mastering:** An embedded chat functionality mimics a Game Master, dynamically crafting and guiding the narrative based on user inputs. The application maintains the chat history and context, ensuring the flow of the RPG is maintained.

4. **Player Customization:** Users can add multiple players, each with their unique balance points and descriptions, contributing to the depth of the gameplay.

5. **Knowledge Graph Visualization:** Alongside the narrative development, a knowledge graph provides a visual representation of the world's various elements and their connections.


## Implications

The results of this research are fundamental in demonstrating how powerful natural language models can be tailored to create intricate and detailed content, exploring the boundaries of automation and creativity in the world of fiction.

## Installation & Usage

### Step-by-step Installation Guide

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/RafaelBaSantos/COS836-2023-2-LLM
    cd COS836-2023-2-LLM
    ```

2. **Set up a Virtual Environment (Optional but Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows use: .\venv\Scripts\activate
    ```

3. **Install Required Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Store OpenAI API Key**:
    You can either set it up in the Streamlit app's sidebar when prompted, or for a more permanent solution, create a `secrets.toml` in the root directory:
    ```toml
    [DEFAULT]
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    ```

## Usage

1. **Launch the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

2. Open the URL displayed in your terminal (usually `http://localhost:8501/`) in your browser.

3. In the Streamlit sidebar, fill out the relevant fields for your RPG's world settings such as genre, tone, technology level, etc.

4. Customize the number of places, buildings, vehicles, characters, items, species, organizations, and more.

5. Add player characters, providing details about each.

6. Once everything is set up, click on 'Start Adventure'. The LLM will then proceed to generate the world and story.

7. The generated narrative is presented in a chat format. The user can interact with the chat, influencing the flow of the story.

8. The knowledge graph will visually represent key elements and their relationships within the generated world.


## Contributions

Feel free to fork the repository, make your changes, and submit a pull request. We are open to any enhancements or new features!

---

If you find this tool helpful or have any feedback, please star the repository and let us know. Your support is much appreciated!