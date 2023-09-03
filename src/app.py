import streamlit as st
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain import LLMChain

import pandas as pd

# Load environment variables
load_dotenv(r'E:\Documentos\Estudos\UFRJ_PESC\3_periodo\llm\projeto final\COS836-2023-2-LLM\src\utils\openai_api_key.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

chat_t1 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k", temperature=1)

sys_nodes_edges = """You are a Professional GameMaster doing the WorldBuilding of an RPG world.

{gm_personality}

The player's characters (PCs) are:

**{name_pc_1}**: {description_pc_1}

**{name_pc_2}**: {description_pc_2}

When creating the world, consider that the player's characters live in it,  but don't let it interfere with player agency. The world is filled with NPCs (non-player characters), that can live independently of the PCs.
"""

usr_nodes_edges = """
- Step 1:
Think concisely about an original and engaging world. You should consider the following aspects, but never talk about them directly:
    - The genre is {genre};
    - The world brings {feels};
    - The tone of the story is {tone};
    - The level of technology is {technology};
    - Magic is seen as {magic};
    - The atmosphere is {atmosphere};
    - The story is inspired by {writters};
    - The story is inspired by the universe of {universes}.
    - {addicional_info}
    - The story deals with more or less 4 of the following subjects: {subjects_of_drama}
    - When describing any entity (location, object, character, species, condition, conflict, organization, title, myth, etc.), you must provide specific names and brief descriptions.

- Step 2:
Find the main topics in this story, that could be better developed, and think of a list of topics and a brief explanation of each.

- Step 3:
Identify the entities in the text. An entity can be a noun or a noun phrase that refers to a real-world object or an abstract concept. You can use a named entity recognition (NER) tool or a part-of-speech (POS) tagger to identify the entities.
Each entity should have a name, a label, and a detailed description. The label can be one of the following:

- Place: Physical places, such as continents, cities, mountains, or forests;
- Building: Natural constructions and formations. Examples: house, dungeon, cave, castle;
- Vehicle: Means of transportation. Example: car, ship, bicycle.
- Character: Individuals or animals in specific roles. Examples: hero, villain, guard, pet.
- Item: Tangible items or documents. Examples: weapons, clothes, letters, and books.
- Species: Biological or cultural groups. Examples: elves, humans, and dogs.
- Organization: Groups with common purposes. Examples: churches, governments, and sects.
- Condition: Physical or emotional states. Examples: illness, inspiration, sadness.
- Drama: Challenges, sociocultural tensions, or military conflicts. Examples: military influence, class relations, and economic problems.
- Myth: Narratives, beliefs, or prose. Examples: legends about the origin of the universe, and myths of monsters.

- Step 4
Identify the relationships between the entities. A relationship can be a verb or a prepositional phrase that connects two entities. You can use dependency parsing to identify the relationships.
The relationships should be represented by triplets of subject, predicate, and object, where the subject and object are entities mapped in the last step and the predicate can have the following labels:

\1 IS_IN: Represents location or use.
    \2 Possible subjects and objects:
        \3 Buildings => Locations
        \3 Vehicles => Locations/Buildings
        \3 Characters => Locations/Buildings/Vehicles
        \3 Items => Locations/Buildings/Vehicles/Characters
\1 OWNER_OF: Denotes direct ownership.
    \2 Possible subjects and objects:
        \3 Characters/Organizations => Locations/Buildings/Vehicles/Characters/Items
\1 POSITIVE/NEGATIVE/NEUTRAL_RELATIONSHIP: Denotes an interpersonal interaction or bond.
    \2 Possible subjects and objects:
        \3 Characters/Organizations => Characters/Organizations
\1 AFFECTED_BY: Shows entities impacted by conditions or conflicts.
    \2 Possible subjects and objects:
        \3 Characters => Conditions/Drama
        \3 Organizations => Drama
\1 MEMBER_OF: It indicates belonging to groups, categories, or organizations.
    \2 Possible subjects and objects:
        \3 Characters => Organizations/Species
\1 BASED_ON: Links entities to stories, myths, legends, or cultural influences.
    \2 Possible subjects and objects:
        \3 Anything => Myths
        \3 Organizations => Drama
\1 SUCCEEDS/PRECEDES: It represents what comes after/before in a time sequence.
    \2 Possible subjects and objects:
        \3 Myths/Drama => Myths/Drama

Find as many relationships as possible within these limitations.

- Step 5
Make a dictionary of lists, that contains a list of nodes, that represent the entities found in Step 3, and e a list of edges, that represent a knowledge graph, with the relations found in Step 4.
___
Example of list of edges:
[('Name of Entity A', 'Name of Entity B',{{'relation':'RELATIONSHIP_1'}}),
 ('Name of Entity C', "Entity Name A",{{'relation':'RELATIONSHIP_2'}})]
___
Example of list of nodes:
[('Name of Entity A', {{'label': 'Label A', 'description': 'Description A'}}),
 ('Name of Entity B', {{'label': 'Label B', 'description': 'Description B'}}),
 ('Name of Entity C', {{'label': 'Label C', 'description': 'Description C'}})]
__
Example of dictionary of lists (generate different examples but with the same structure):
{{nodes: [('Name of Entity A', {{'label': 'Label A', 'description': 'Description A'}}),
          ('Name of Entity B', {{'label': 'Label B', 'description': 'Description B'}}),
          ('Name of Entity C', {{'label': 'Label C', 'description': 'Description C'}})],
 edges: [('Name of Entity A', 'Name of Entity B',{{'relation':'RELATIONSHIP_1'}}),
         ('Name of Entity C', "Entity Name A",{{'relation':'RELATIONSHIP_2'}})]}}
__

Do not write anything about steps 1, 2, 3, 4. Return only the result of steps 5.

You must create {places} places, {buildings} buildings, {vehicles} vehicles, {characters} characters (plus the player's characters), {items} items, {species} species, {organizations} organizations, {conditions} conditions, {dramas} dramas, {myths} myths, and as many relationships (edges) as you need.
Items, Buildings, Vehicles, and Characters need at least one relationship of type IS_IN.
Myths need at least one BASED_ON relationship.
Species and Organizations need at least one MEMBER_OF relationship.

You must never explain your code.
"""

usr_adventure = """
You should consider the following aspects, but never talk about them directly:
    - The genre is {genre};
    - The world brings {feels};
    - The tone of the story is {tone};
    - The level of technology is {technology};
    - Magic is seen as {magic};
    - The atmosphere is {atmosphere};
    - The story is inspired by {writters};
    - The story is inspired by the universe of {universes}.
    - {addicional_info}
    - The story deals with more or less 4 of the following subjects: {subjects_of_drama}
    - When describing any entity (location, object, character, species, condition, conflict, organization, title, myth, etc.), you must provide specific names and brief descriptions.
__

The following dictionary describes an RPG world, where entities are represented by nodes and relationships are represented by edges.
{nodes_edges}
__

In role-playing games (RPGs), an "adventure" comprises a complete story arc with a beginning, middle, and end. The concept of time in these adventures is abstracted into "scenes," rather than precise units. Scenes begin with the introduction of a new location or situation and end when characters depart from that location or resolve the situation. These scenes can be interrupted, transitioning into new scenes, allowing the game master to manage pacing and character abilities effectively. Scenes are categorized into three types: action scenes, exploration scenes, and role-playing scenes.

- **Action Scenes**: These scenes focus on physical objectives and often feature battles, chases, or traps. The gameplay is fast-paced, broken down into rounds, and characterized by frequent unexpected events. The stakes are high as the characters' lives are often in jeopardy.

- **Exploration Scenes**: These scenes involve intellectual engagement with the environment or objects rather than with characters. Examples include examining crime scenes for clues, solving puzzles, or navigating unknown territories. Although rules and skill checks may be applicable, creative problem-solving by the players is crucial.

- **Role-playing Scenes**: These scenes are characterized by dialogue and emotional interactions among characters. They may not always have a specific goal but usually contribute to story progression. Players get the chance to delve into their characters' personalities and motivations. 

___
The adventure follows those scenes:

- Scene 1: Normality
    - Role-playing scene.
    - Start with a normal situation with no imminent danger.
    - Allow the players to get comfortable before the action begins.

- Scene 2: Motivation
    - Role-playing scene
    - Present a plot hook, something to get the characters out of their normal situation and start the action.
    - A clear goal and convincing reasons to act are needed.

- Scene 3: Partial Victory
    - Action scene.
    - The characters encounter the first challenge.
    - They must have a clear but not total victory to recognize their abilities.

- Scene 4: Information and Development
    - Exploration scene.
    - Characters figure out a plan to solve the main problem.
    - Use role-play and exploration scenes to come to a conclusion about what to do next.

- Scene 5: Partial Defeat
    - Action scene.
    - Plan a setback for the characters to make the narrative more interesting.
    - It could be a more difficult fight, a failure of another kind, or even a betrayal.
    - Characters should be encouraged to think and strategize.

- Scene 6: Conditions for Victory
    - Exploration scene.
    - Characters figure out how to win once and for all.
    - There may be preparation for the final confrontation through allies, equipment, or strategies.

- Scene 7: Total Victory
    - Action scene.
    - This is the climax of the adventure, where the adventurers face the threat with a chance to win.
    - Value the players' tactics, but the villains must be dangerous.
    - Previous preparation must have been crucial to victory.

- Scene 8: Resolution, Rewards, and Hooks
    - Role-playing scene
    - Everything returns to normal after the heroes' victory.
    - Characters evolve in terms of powers and equipment.
    - Rewards can be distributed and hooks can be inserted for future campaigns.

___
Write an adventure with 8 scenes, as the structure above, based on the world described by the dictionary of entities and relationships. You can create new entities (characters, items, etc.) when needed.

Each Scene should have:
- Title;
- Location (place or building where the scene happens);
- Scene description (each scene should have a 3-paragraph long description);
- NPCs (Characters present on the scene);
- Battlemap Description (A description of the location of each entity - characters and objects - on the scene);
- Decisions (A list of three decisions that players can make).

Each NPC should have:
Description;
Appearance;
Special Hability.

Here is an example of a scene. You should follow this structure to create other scenes, but not follow the same content:

### Scene X: 
- **Title:** Acquaintances and Secrets.
- **Location:** Night Owl Cafe, a popular hub in the city's heart, Varona.
- **Scene Description:**
    The campaign kicks off at the Night Owl Cafe in Varona City, a dimly lit establishment filled with the aroma of synth-coffee and the muffled chatter of patrons from all walks of life. The players are friends or acquaintances, meeting to catch up or discuss matters both mundane and intriguing.
    During the meeting, an elusive character named Zephyr approaches the table, seemingly out of nowhere. Zephyr's eyes scan the room nervously as he whispers about an ancient artifact, The Orb of Seraphina, believed to be a forgotten relic of magical power, hidden within the Maze of Shadows.
    Zephyr is interrupted by a sudden commotion. Security drones patrol outside, and a sense of danger lingers. The urgency in Zephyr's voice grows as he implores the characters to find the Orb before the tyrannical government, the Techno Dominion, does.
- **NPCs**:
    > Zephyr:
- Description: Enigmatic and shrouded in a cloak of mystery, Zephyr is a seasoned treasure hunter and antiquarian with an uncanny knack for sniffing out powerful artifacts. Though his appearance is unassuming—clad in a hooded, tattered robe that obscures most of his face, save for piercing blue eyes—he moves with the silent grace of someone who's spent a lifetime evading capture. Tattooed sigils glimmer faintly on his exposed forearms, a testament to a hidden magical aptitude. Zephyr is deeply versed in the legends and lore surrounding relics like the Orb of Seraphina, making him a valuable, albeit elusive, guide. His motives for wanting the Orb remain ambiguous, and he's the type of character who always seems to have one more trick up his sleeve.
- Special Ability: "Temporal Flicker" - Can momentarily phase out of reality to evade harm or to teleport short distances.
- Appearance: Zephyr stands at 5'10", his lean, wiry frame shrouded in a dark gray cloak frayed at the edges. His hood is usually pulled low, obscuring his face and allowing only his piercing, glowing blue eyes to be seen. Subtle, enchanted sigils tattoo his pale skin, visible on his forearms and hands, which are adorned with various rings. When his hood is pulled back, short, sandy hair and a finely braided beard frame his sharp, enigmatic features. Equipped with silent utility boots and a belt of mysterious pouches, Zephyr moves with a calculated grace, his presence inviting curiosity yet caution.

- **Battlemap Description**:
- Zephyr is sitting at a table, near the bottom left corner.
- Security drones are patrolling inside the cafe.
- Other patrons are seated at tables throughout the cafe.
- The bar counter is at the top of the building.

- **Decisions:**
    1. Agree to embark on the mission to find the Orb of Seraphina.
    2. Interrogate Zephyr for more information about the Orb and the Maze of Shadows.
    3. Reject the mission and leave the cafe, risking the wrath of Zephyr and possibly drawing attention from the Techno Dominion.
"""

usr_update_nodes_edges = """The following dictionary describes a graph of an RPG world, where entities are represented by nodes and relationships are represented by edges.

{nodes_edges}

__
Content Text:

{context_text}

__
- Step 1:
Identify the entities in the Content Text. An entity can be a noun or a noun phrase that refers to a real-world object or an abstract concept. You can use a named entity recognition (NER) tool or a part-of-speech (POS) tagger to identify the entities.
Each entity should have a name, a label, and a detailed description. The label can be one of the following:

- Place: Physical places, such as continents, cities, mountains, or forests;
- Building: Natural constructions and formations. Examples: house, dungeon, cave, castle;
- Vehicle: Means of transportation. Example: car, ship, bicycle.
- Character: Individuals or animals in specific roles. Examples: hero, villain, guard, pet.
- Item: Tangible items or documents. Examples: weapons, clothes, letters, and books.
- Species: Biological or cultural groups. Examples: elves, humans, and dogs.
- Organization: Groups with common purposes. Examples: churches, governments, and sects.
- Condition: Physical or emotional states. Examples: illness, inspiration, sadness.
- Drama: Challenges, sociocultural tensions, or military conflicts. Examples: military influence, class relations, and economic problems.
- Myth: Narratives, beliefs, or prose. Examples: legends about the origin of the universe, and myths of monsters.

- Step 2
Identify the relationships between the entities. A relationship can be a verb or a prepositional phrase that connects two entities. You can use dependency parsing to identify the relationships.
The relationships should be represented by triplets of subject, predicate, and object, where the subject and object are entities mapped in the last step and the predicate can have the following labels:

\1 IS_IN: Represents location or use.
    \2 Possible subjects and objects:
        \3 Buildings => Locations
        \3 Vehicles => Locations/Buildings
        \3 Characters => Locations/Buildings/Vehicles
        \3 Items => Locations/Buildings/Vehicles/Characters
\1 OWNER_OF: Denotes direct ownership.
    \2 Possible subjects and objects:
        \3 Characters/Organizations => Locations/Buildings/Vehicles/Characters/Items
\1 POSITIVE/NEGATIVE/NEUTRAL_RELATIONSHIP: Denotes an interpersonal interaction or bond.
    \2 Possible subjects and objects:
        \3 Characters/Organizations => Characters/Organizations
\1 AFFECTED_BY: Shows entities impacted by conditions or conflicts.
    \2 Possible subjects and objects:
        \3 Characters => Conditions/Drama
        \3 Organizations => Drama
\1 MEMBER_OF: It indicates belonging to groups, categories, or organizations.
    \2 Possible subjects and objects:
        \3 Characters => Organizations/Species
\1 BASED_ON: Links entities to stories, myths, legends, or cultural influences.
    \2 Possible subjects and objects:
        \3 Anything => Myths
        \3 Organizations => Drama
\1 SUCCEEDS/PRECEDES: It represents what comes after/before in a time sequence.
    \2 Possible subjects and objects:
        \3 Myths/Drama => Myths/Drama

Find as many relationships as possible within these limitations.

- Step 3

If an entity is not yet present in the dictionary, add a new node along with its relevant attributes.
Establish new relationships between nodes as needed and update existing ones with fresh informations, which may involve altering attributes or reconfiguring relationships.
Remove nodes or connections from the dictionary when required.

Items, Buildings, Vehicles, and Characters need at least one relationship of type IS_IN.
Myths need at least one BASED_ON relationship.
Species and Organizations need at least one MEMBER_OF relationship.

- Step 4
Return the updated dictionary of lists, with the entities and relationships found in Step 3.

__
Example of dictionary of lists:
{{nodes: [('Name of Entity A', {{'label': 'Label A', 'description': 'Description A'}}),
          ('Name of Entity B', {{'label': 'Label B', 'description': 'Description B'}}),
          ('Name of Entity C', {{'label': 'Label C', 'description': 'Description C'}})],
 edges: [('Name of Entity A', 'Name of Entity B',{{'relation':'RELATIONSHIP_1'}}),
         ('Name of Entity C', "Entity Name A",{{'relation':'RELATIONSHIP_2'}})]}}
__

Do not write anything about steps 1, 2, or 3. Return only the result of steps 4.

You must never explain your code."""

st.title('Generative Worldbuilding with Large Language Models')

# User Input
genre = st.text_input("WHAT IS THE GENRE?", "cyberpunk fantasy")
feels = st.text_input("HOW DOES THE WORLD FEEL?", "a hopeless feeling")
tone = st.text_input("WHAT IS THE TONE OF THE WORLD?", "dark")
technology = st.text_input("WHAT IS THE LEVEL OF TECHNOLOGY?", "futuristic")
magic = st.text_input("WHAT IS THE LEVEL OF MAGIC?", "something ultra-rare and forgotten")
atmosphere = st.text_input("WHAT IS THE ATMOSPHERE?", "dystopic")
writters = st.text_input("WHAT WRITTERS THE STORY IS INSPITED BY?",
                         "Isaac Asimov, Greg Bear, Frank Herbert, William Gibson, Philip K. Dick")
universes = st.text_input("WHAT UNIVERSES THE STORY IS INSPITED BY?",
                          "Cyberpunk 2077, Lancer RPG, BladeRunner, Ghost in the Shell, Cowboy Bebop")
subjects_of_drama = st.text_area("WHAT SUBJECTS OF DRAMA THE STORY IS BASED ON?",
                                  "Governance and its influence, Legal principles and rule of law, Provision of social services, Economic dynamics, Equity in wealth distribution, Agricultural practices and trade, Interpersonal relationships influenced by race, class, gender or sexual orientation, Military power and influence, Role and influence of religion, Technological advances and influences, Impact of arts and cultural expressions, Geographical features, Interactions between civilizations, their histories and conflicts, Foundations of the laws of nature, Theories about the origin of the universe, History of species and cultures that have inhabited the world, Needs and interactions of different species and cultures")
addicional_info = st.text_area("FEEL FREE TO ADD ANY ADDICIONAL REQUESTS")

gm_personality = st.text_area("GM Personality")
name_pc_1 = st.text_input("name_pc_1")
description_pc_1 = st.text_area("description_pc_1")
name_pc_2 = st.text_input("name_pc_2")
description_pc_2 = st.text_area("description_pc_2")


places = st.slider("Places:", value=1, min_value=0, max_value=5)
buildings = st.slider("Buildings:", value=8, min_value=0, max_value=20)
vehicles = st.slider("Vehicles:", value=1, min_value=0, max_value=10)
characters = st.slider("Characters:", value=16, min_value=0, max_value=50)
items = st.slider("Items:", value=2, min_value=0, max_value=20)
species = st.slider("Species:", value=3, min_value=0, max_value=10)
organizations = st.slider("Organizations:", value=3, min_value=0, max_value=10)
conditions = st.slider("Conditions:", value=1, min_value=0, max_value=5)
dramas = st.slider("Dramas:", value=2, min_value=0, max_value=5)
myths = st.slider("Myths:", value=2, min_value=0, max_value=20)

if st.button('Generate Adventure'):

    sys_nodes_edges_msg = SystemMessagePromptTemplate.from_template(sys_nodes_edges)
    urs_nodes_edges_msg = HumanMessagePromptTemplate.from_template(usr_nodes_edges)
    chat_prompt = ChatPromptTemplate.from_messages([sys_nodes_edges_msg,
                                                    urs_nodes_edges_msg])
    chain = LLMChain(llm=chat_t1, prompt=chat_prompt)
    nodes_edges = chain.run(genre=genre, feels=feels, tone=tone, technology=technology, magic=magic,
                            atmosphere=atmosphere, writters=writters, universes=universes,
                            subjects_of_drama=subjects_of_drama, addicional_info=addicional_info,
                            places=places, buildings=buildings, vehicles=vehicles, characters=characters, items=items,
                            species=species, organizations=organizations, conditions=conditions, dramas=dramas,
                            myths=myths,
                            gm_personality=gm_personality, name_pc_1=name_pc_1, description_pc_1=description_pc_1,
                            name_pc_2=name_pc_2, description_pc_2=description_pc_2)

    st.write(nodes_edges)

    usr_adventure_msg = HumanMessagePromptTemplate.from_template(usr_adventure)

    chat_prompt_adv = ChatPromptTemplate.from_messages([usr_adventure_msg])
    chain_adv = LLMChain(llm=chat_t1, prompt=chat_prompt_adv)
    adventure = chain_adv.run(genre=genre, feels=feels, tone=tone, technology=technology, magic=magic,
                              atmosphere=atmosphere, writters=writters, universes=universes,
                              subjects_of_drama=subjects_of_drama, addicional_info=addicional_info,
                              nodes_edges=nodes_edges)

    st.write(adventure)