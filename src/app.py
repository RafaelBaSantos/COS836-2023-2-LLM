import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain import LLMChain

import random as rd

from pyvis.network import Network
import ast


def generate_nodes_edges(chat_openai):
    sys_nodes_edges = """You are a Professional GameMaster doing the WorldBuilding of an RPG world.
    Your personality is defined by: {gm_personality}"""

    usr_nodes_edges = """
    The player's characters (PCs) are:

    {description_pcs}

    The world is filled with NPCs (non-player characters), that can live independently of the PCs.
    
    Balance Point represent a point of balance between the character's physical and mental strengths). A high number means they are better at Mental. A low number means they are better at Physical.    
    __
    - Step 1:
    Think concisely about an original and engaging world. You should consider the following aspects, but never talk about them directly:
        - The genre is {genre};
        - The world brings {feels};
        - The tone of the story is {tone};
        - The level of technology is {technology};
        - Magic is seen as {magic};
        - The atmosphere is {atmosphere};
        - The story is inspired by {writers};
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

    1) IS_IN: Represents location or use.
    1.1) Possible subjects and objects:
            - Buildings => Locations
            - Vehicles => Locations/Buildings
            - Characters => Locations/Buildings/Vehicles
            - Items => Locations/Buildings/Vehicles/Characters
    2) OWNER_OF: Denotes direct ownership.
        2.1) Possible subjects and objects:
            - Characters/Organizations => Locations/Buildings/Vehicles/Characters/Items
    3) POSITIVE/NEGATIVE/NEUTRAL_RELATIONSHIP: Denotes an interpersonal interaction or bond.
        3.1) Possible subjects and objects:
            - Characters/Organizations => Characters/Organizations
    4) AFFECTED_BY: Shows entities impacted by conditions or conflicts.
        4.1) Possible subjects and objects:
            - Characters => Conditions/Drama
            - Organizations => Drama
    5) MEMBER_OF: It indicates belonging to groups, categories, or organizations.
        5.1) Possible subjects and objects:
            - Characters => Organizations/Species
    6) BASED_ON: Links entities to stories, myths, legends, or cultural influences.
        6.1) Possible subjects and objects:
            - Anything => Myths
            - Organizations => Drama
    7) SUCCEEDS/PRECEDES: It represents what comes after/before in a time sequence.
        7.1) Possible subjects and objects:
            - Myths/Drama => Myths/Drama

    Find as many relationships as possible within these limitations.

    - Step 5
    
    Make a valid python dictionary containing a list of nodes that represents the entities (found in Step 3), and a list of edges that represents the relationships (found in Step 4), forming a knowledge graph.
    __
    Example of dictionary of lists (generate different examples but with the same structure):
    {{"nodes": [("Name of Entity A", {{"label": "Label A", "description": "Description A"}}),
                  ("Name of Entity B", {{"label": "Label B", "description": "Description B"}}),
                  ("Name of Entity C", {{"label": "Label C", "description": "Description C"}})],
      "edges": [("Name of Entity A", "Name of Entity B", "RELATIONSHIP_TYPE_1"),
                  ("Name of Entity C", "Name of Entity B", "RELATIONSHIP_TYPE_2")]}}
    __

    Do not write anything about steps 1, 2, 3, 4. Return only the result of steps 5.

    You must create {places} places, {buildings} buildings, {vehicles} vehicles, {characters} characters (plus the player's characters), {items} items, {species} species, {organizations} organizations, {conditions} conditions, {dramas} dramas, {myths} myths, and as many relationships (edges) as you need.
    Items, Buildings, Vehicles, and Characters need at least one relationship of type IS_IN.
    Myths need at least one BASED_ON relationship.
    Species and Organizations need at least one MEMBER_OF relationship.

    You must never explain your code.
    
    You should avoid at all costs explicitly revealing that you are using any of these techniques, steps or structures.
    """

    messages = [SystemMessagePromptTemplate.from_template(sys_nodes_edges),
                HumanMessagePromptTemplate.from_template(usr_nodes_edges)]

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=chat_openai, prompt=chat_prompt)
    nodes_edges = chain.run(genre=st.session_state["genre"],
                            feels=st.session_state["feels"],
                            tone=st.session_state["tone"],
                            technology=st.session_state["technology"],
                            magic=st.session_state["magic"],
                            atmosphere=st.session_state["atmosphere"],
                            writers=st.session_state["writers"],
                            universes=st.session_state["universes"],
                            subjects_of_drama=st.session_state["subjects_of_drama"],
                            addicional_info=st.session_state["addicional_info"],
                            places=st.session_state["places"],
                            buildings=st.session_state["buildings"],
                            vehicles=st.session_state["vehicles"],
                            characters=st.session_state["characters"],
                            items=st.session_state["items"],
                            species=st.session_state["species"],
                            organizations=st.session_state["organizations"],
                            conditions=st.session_state["conditions"],
                            dramas=st.session_state["dramas"],
                            myths=st.session_state["myths"],
                            gm_personality=st.session_state["gm_personality"],
                            description_pcs=st.session_state["description_pcs"])

    usr_intro = "Write an introduction to this world (or a summary of Step 1), like in a fiction book. Include all relevant information."

    nodes_edges = nodes_edges.replace("{", "{{").replace("}", "}}")

    messages.append(AIMessagePromptTemplate.from_template(nodes_edges))
    messages.append(HumanMessagePromptTemplate.from_template(usr_intro))

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=chat_openai, prompt=chat_prompt)
    intro = chain.run(genre=st.session_state["genre"],
                      feels=st.session_state["feels"],
                      tone=st.session_state["tone"],
                      technology=st.session_state["technology"],
                      magic=st.session_state["magic"],
                      atmosphere=st.session_state["atmosphere"],
                      writers=st.session_state["writers"],
                      universes=st.session_state["universes"],
                      subjects_of_drama=st.session_state["subjects_of_drama"],
                      addicional_info=st.session_state["addicional_info"],
                      places=st.session_state["places"],
                      buildings=st.session_state["buildings"],
                      vehicles=st.session_state["vehicles"],
                      characters=st.session_state["characters"],
                      items=st.session_state["items"],
                      species=st.session_state["species"],
                      organizations=st.session_state["organizations"],
                      conditions=st.session_state["conditions"],
                      dramas=st.session_state["dramas"],
                      myths=st.session_state["myths"],
                      gm_personality=st.session_state["gm_personality"],
                      description_pcs=st.session_state["description_pcs"])

    return nodes_edges, intro


def generate_scene(chat_openai):

    sys_scene = """You are a Professional GameMaster.
    Your personality is defined by: {gm_personality}"""

    usr_scene = """
    The player's characters (PCs) are:

    {description_pcs}

    The world is filled with NPCs (non-player characters), that can live independently of the PCs.
    
    Balance Point represent a point of balance between the character's physical and mental strengths). A high number means they are better at Mental. A low number means they are better at Physical.    
    __
        You should consider the following aspects, but never talk about them directly:
        - The genre is {genre};
        - The world brings {feels};
        - The tone of the story is {tone};
        - The level of technology is {technology};
        - Magic is seen as {magic};
        - The atmosphere is {atmosphere};
        - The story is inspired by {writers};
        - The story is inspired by the universe of {universes}.
        - {addicional_info}
        - The story deals with more or less 4 of the following subjects: {subjects_of_drama}
        - When describing any entity (location, object, character, species, condition, conflict, organization, title, myth, etc.), you must provide specific names and brief descriptions.
    __
    The following dictionary describes the RPG world, where entities are represented by nodes and relationships are represented by edges.

    {nodes_edges}
    __
    In role-playing games (RPGs), an "adventure" comprises a complete story arc with a beginning, middle, and end. The concept of time in these adventures is abstracted into "scenes," rather than precise units. Scenes begin with the introduction of a new location or situation and end when characters depart from that location or resolve the situation. These scenes can be interrupted, transitioning into new scenes, allowing the game master to manage pacing and character abilities effectively.
    Scenes are categorized into three types:
    - **Action Scenes**: These scenes focus on physical objectives and often feature battles, chases, or traps. The gameplay is fast-paced, broken down into rounds, and characterized by frequent unexpected events. The stakes are high as the characters' lives are often in jeopardy.
    - **Exploration Scenes**: These scenes involve intellectual engagement with the environment or objects rather than with characters. Examples include examining crime scenes for clues, solving puzzles, or navigating unknown territories. Although rules and skill checks may be applicable, creative problem-solving by the players is crucial.
    - **Role-playing Scenes**: These scenes are characterized by dialogue and emotional interactions among characters. They may not always have a specific goal but usually contribute to story progression. Players get the chance to delve into their characters' personalities and motivations. 
    ___
    The adventure follows this structure:

    {story_structure}
    ___
    Scenes should follow this structure:

    ### Scene X: 
    - **Title:** [The title of the scene]
    - **Location:** [place or building where the scene happens]
    - **Scene Description:**
        [A 3-paragraph long description]
    - **NPCs**:
    1) [Name of the first character]:
    1.1) Description: [Description of the first character]
    1.2) Special Ability: [A Special ability that the first character has]
    1.3) Appearance: [First character appearance]
    2) [Name of the second character]:
    2.1) Description: [Description of the second character]
    2.2) Special Ability: [A Special ability that the second character has]
    2.3) Appearance: [Second character appearance]
    3. (...)
    - **Battlemap Description**:
    [A bullet point list with the description of the location of each entity (characters and objects) on the scene.]
    - **End Condition:**
    [A bullet point list of things that players can make that would conclude this scene. If any of these conditions are met, the scene ends.]
    __
    **Story Summary (Here is what happened in the story until now)**:

    {current_context}
    __
    Create scene {current_scene_num}, with the correct structure, based on the world described by the dictionary of entities and relationships, and on the Story Summary. You can create new entities (characters, items, etc.) when needed.
    
    You should avoid at all costs explicitly revealing that you are using any of these techniques, steps or structures.
    """

    messages = [SystemMessagePromptTemplate.from_template(sys_scene),
                HumanMessagePromptTemplate.from_template(usr_scene)]

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=chat_openai, prompt=chat_prompt)
    current_scene = chain.run(gm_personality=st.session_state["gm_personality"],
                              description_pcs=st.session_state["description_pcs"],
                              genre=st.session_state["genre"],
                              feels=st.session_state["feels"],
                              tone=st.session_state["tone"],
                              technology=st.session_state["technology"],
                              magic=st.session_state["magic"],
                              atmosphere=st.session_state["atmosphere"],
                              writers=st.session_state["writers"],
                              universes=st.session_state["universes"],
                              subjects_of_drama=st.session_state["subjects_of_drama"],
                              addicional_info=st.session_state["addicional_info"],
                              nodes_edges=st.session_state["nodes_edges"],
                              story_structure=st.session_state["story_structure"],
                              current_context=st.session_state["current_context"],
                              current_scene_num=st.session_state["current_scene_num"])

    return current_scene


def generate_response(chat_openai: ChatOpenAI) -> str:
    sys_game_master = """
    You are a Professional GameMaster who is running an RPG campaign.
    Your personality is defined by: {gm_personality}
    __
    The player's characters (PCs) are:

    {description_pcs}

    The world is filled with NPCs (non-player characters), that can live independently of the PCs.
    
    Balance Point represent a point of balance between the character's physical and mental strengths). A high number means they are better at Mental. A low number means they are better at Physical.
    __
    You should consider the following aspects, but never talk about them directly:
        - The genre is {genre};
        - The world brings {feels};
        - The tone of the story is {tone};
        - The level of technology is {technology};
        - Magic is seen as {magic};
        - The atmosphere is {atmosphere};
        - The story is inspired by {writers};
        - The story is inspired by the universe of {universes}.
        - {addicional_info}
        - The story deals with more or less 4 of the following subjects: {subjects_of_drama}
        - When describing any entity (location, object, character, species, condition, conflict, organization, title, myth, etc.), you must provide specific names and brief descriptions.
    __
    In role-playing games (RPGs), an "adventure" comprises a complete story arc with a beginning, middle, and end. The concept of time in these adventures is abstracted into "scenes," rather than precise units. Scenes begin with the introduction of a new location or situation and end when characters depart from that location or resolve the situation. These scenes can be interrupted, transitioning into new scenes, allowing the game master to manage pacing and character abilities effectively.
    ___
    We are now at:

    {scene}
    __
    The game mechanic of this RPG is closely related to the Laser and Feelings system but with some changes.
    
    Players have a Balance Point, a number from 2 to 5. A high number means they are better at Mental. A low number means they are better at Physical.
    
    The Mental attribute covers the character's intelligence, wisdom, perception, willpower and other mental and spiritual abilities. Some situations that can be tested with the Mental Attribute:
    Knowledge: Remembering information, deciphering codes or solving puzzles.
    Perception: Noticing something out of the ordinary, detecting traps or sensing an ambush.
    Influence: Persuade, intimidate, deceive or inspire other characters.
    Mental Resistance: Defend against magic, illusions or temptations.
    Concentration: Staying focused when performing complicated tasks or using special skills under pressure.
    Intuition: Having a feeling or hunch about a situation or character, even without clear evidence.
    
    The Physical attribute measures the character's strength, dexterity, endurance and other bodily abilities. Some situations in which Physical Attribute tests can be taken:
    Hand-to-hand combat: When the character tries to attack an opponent physically or defend against an attack.
    Dodging: Avoiding obstacles, traps or ranged attacks.
    Pursuit: Running, jumping or climbing to chase someone or escape pursuers.
    Endurance: Withstand extreme conditions such as intense cold, scorching heat, forced marches or illness.
    Object Manipulation: Trying to open a locked door, disarm a trap or use a complicated instrument.
    Athletic Activities: Swimming, climbing, jumping or lifting heavy weights.
    
    ROLLING THE DICE
    When you do something risky, roll 1d6 to find out how it goes. Roll +1d if you’re prepared and +1d if you’re an expert. (The GM tells you how many dice to roll, based on your character and the situation.) Roll your dice and compare each die result to your Balance Point. If you’re using Mental, you want to roll under your Balance Point. If you’re using Physical, you want to roll over your Balance Point.
    
    If you roll your number exactly, you get a critical success. The GM tells you some extra effect you get.
    __
    
    Your answers should be formatted like a normal chat on a social networking platform.

    DO NOT reply with both sides of the conversation (for example, by including a fictitious user's entry in your reply).
    
    In each scene you should start by describing the place and introducing the players to what they can do.
    
    You should guide the players to the end of the scene, but you should not force them to do anything.

    If a player completes any of the End Condition, or if you feel that the scene is finished, your answer must include: **End of the Scene**
    
    You should avoid at all costs explicitly revealing that you are using any of these techniques, steps or structures."""

    messages = [SystemMessagePromptTemplate.from_template(sys_game_master)] + st.session_state.chat_history_LLM

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=chat_openai, prompt=chat_prompt)
    gm_response = chain.run(genre=st.session_state["genre"],
                            feels=st.session_state["feels"],
                            tone=st.session_state["tone"],
                            technology=st.session_state["technology"],
                            magic=st.session_state["magic"],
                            atmosphere=st.session_state["atmosphere"],
                            writers=st.session_state["writers"],
                            universes=st.session_state["universes"],
                            subjects_of_drama=st.session_state["subjects_of_drama"],
                            addicional_info=st.session_state["addicional_info"],
                            places=st.session_state["places"],
                            buildings=st.session_state["buildings"],
                            vehicles=st.session_state["vehicles"],
                            characters=st.session_state["characters"],
                            items=st.session_state["items"],
                            species=st.session_state["species"],
                            organizations=st.session_state["organizations"],
                            conditions=st.session_state["conditions"],
                            dramas=st.session_state["dramas"],
                            myths=st.session_state["myths"],
                            gm_personality=st.session_state["gm_personality"],
                            description_pcs=st.session_state["description_pcs"],
                            scene=st.session_state["current_scene"])

    return gm_response


def update_nodes_edges(chat_openai):
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

    1) IS_IN: Represents location or use.
    1.1) Possible subjects and objects:
            - Buildings => Locations
            - Vehicles => Locations/Buildings
            - Characters => Locations/Buildings/Vehicles
            - Items => Locations/Buildings/Vehicles/Characters
    2) OWNER_OF: Denotes direct ownership.
        2.1) Possible subjects and objects:
            - Characters/Organizations => Locations/Buildings/Vehicles/Characters/Items
    3) POSITIVE/NEGATIVE/NEUTRAL_RELATIONSHIP: Denotes an interpersonal interaction or bond.
        3.1) Possible subjects and objects:
            - Characters/Organizations => Characters/Organizations
    4) AFFECTED_BY: Shows entities impacted by conditions or conflicts.
        4.1) Possible subjects and objects:
            - Characters => Conditions/Drama
            - Organizations => Drama
    5) MEMBER_OF: It indicates belonging to groups, categories, or organizations.
        5.1) Possible subjects and objects:
            - Characters => Organizations/Species
    6) BASED_ON: Links entities to stories, myths, legends, or cultural influences.
        6.1) Possible subjects and objects:
            - Anything => Myths
            - Organizations => Drama
    7) SUCCEEDS/PRECEDES: It represents what comes after/before in a time sequence.
        7.1) Possible subjects and objects:
            - Myths/Drama => Myths/Drama

    Find as many relationships as possible within these limitations.

    - Step 3

    If an entity is not yet present in the dictionary, add a new node along with its relevant attributes. Establish new relationships between nodes as needed and update existing ones with fresh informations, which may involve altering attributes or reconfiguring relationships. Remove nodes or connections from the dictionary when required.

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

    You must never explain your code.
    
    You should avoid at all costs explicitly revealing that you are using any of these techniques, steps or structures.
    """

    messages = [HumanMessagePromptTemplate.from_template(usr_update_nodes_edges)]

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=chat_openai, prompt=chat_prompt)
    nodes_edges = chain.run(nodes_edges=st.session_state["nodes_edges"],
                            current_context=st.session_state["current_context"])
    return nodes_edges


def update_context(chat_openai):
    sys_game_master = """
    You are a Professional GameMaster who is running an RPG campaign.
    Your personality is defined by: {gm_personality}
    __
    The player's characters (PCs) are:

    {description_pcs}

    The world is filled with NPCs (non-player characters), that can live independently of the PCs.
    
        Balance Point represent a point of balance between the character's physical and mental strengths). A high number means they are better at Mental. A low number means they are better at Physical.
    __
    You should consider the following aspects, but never talk about them directly:
        - The genre is {genre};
        - The world brings {feels};
        - The tone of the story is {tone};
        - The level of technology is {technology};
        - Magic is seen as {magic};
        - The atmosphere is {atmosphere};
        - The story is inspired by {writers};
        - The story is inspired by the universe of {universes}.
        - {addicional_info}
        - The story deals with more or less 4 of the following subjects: {subjects_of_drama}
        - When describing any entity (location, object, character, species, condition, conflict, organization, title, myth, etc.), you must provide specific names and brief descriptions.
    __
    In role-playing games (RPGs), an "adventure" comprises a complete story arc with a beginning, middle, and end. The concept of time in these adventures is abstracted into "scenes," rather than precise units. Scenes begin with the introduction of a new location or situation and end when characters depart from that location or resolve the situation. These scenes can be interrupted, transitioning into new scenes, allowing the game master to manage pacing and character abilities effectively.
    ___
    We are now at:

    {scene}
    __
    The game mechanic of this RPG is closely related to the Laser and Feelings system but with some changes.

    Players have to choose a number from 2 to 5. A high number means they are better at Mental. A low number means they are better at Physical.

    The Mental attribute covers the character's intelligence, wisdom, perception, willpower and other mental and spiritual abilities. Some situations that can be tested with the Mental Attribute:
    Knowledge: Remembering information, deciphering codes or solving puzzles.
    Perception: Noticing something out of the ordinary, detecting traps or sensing an ambush.
    Influence: Persuade, intimidate, deceive or inspire other characters.
    Mental Resistance: Defend against magic, illusions or temptations.
    Concentration: Staying focused when performing complicated tasks or using special skills under pressure.
    Intuition: Having a feeling or hunch about a situation or character, even without clear evidence.

    The Physical attribute measures the character's strength, dexterity, endurance and other bodily abilities. Some situations in which Physical Attribute tests can be taken:
    Hand-to-hand combat: When the character tries to attack an opponent physically or defend against an attack.
    Dodging: Avoiding obstacles, traps or ranged attacks.
    Pursuit: Running, jumping or climbing to chase someone or escape pursuers.
    Endurance: Withstand extreme conditions such as intense cold, scorching heat, forced marches or illness.
    Object Manipulation: Trying to open a locked door, disarm a trap or use a complicated instrument.
    Athletic Activities: Swimming, climbing, jumping or lifting heavy weights.

    ROLLING THE DICE
    When you do something risky, roll 1d6 to find out how it goes. Roll +1d if you’re prepared and +1d if you’re an expert. (The GM tells you how many dice to roll, based on your character and the situation.) Roll your dice and compare each die result to your number. If you’re using Mental, you want to roll under your number. If you’re using Physical, you want to roll over your number.

    If you roll your number exactly, you get a critical success. The GM tells you some extra effect you get.
    __

    Your answers should be formatted like a normal chat on a social networking platform.

    DO NOT reply with both sides of the conversation (for example, by including a fictitious user's entry in your reply).

    In each scene you should start by describing the place and introducing the players to what they can do.

    You should guide the players to the end of the scene, but you should not force them to do anything.

    If a player completes any of the End Condition, or if you feel that the scene is finished, your answer must include: **End of the Scene**
        
    You should avoid at all costs explicitly revealing that you are using any of these techniques, steps or structures."""

    usr_context = "Write an summary of the story up until this point, like in a fiction book. Include all relevant information."

    messages = [SystemMessagePromptTemplate.from_template(sys_game_master)]\
               + st.session_state.chat_history_LLM\
               + [HumanMessagePromptTemplate.from_template(usr_context)]

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=chat_openai, prompt=chat_prompt)
    context = chain.run(genre=st.session_state["genre"],
                            feels=st.session_state["feels"],
                            tone=st.session_state["tone"],
                            technology=st.session_state["technology"],
                            magic=st.session_state["magic"],
                            atmosphere=st.session_state["atmosphere"],
                            writers=st.session_state["writers"],
                            universes=st.session_state["universes"],
                            subjects_of_drama=st.session_state["subjects_of_drama"],
                            addicional_info=st.session_state["addicional_info"],
                            places=st.session_state["places"],
                            buildings=st.session_state["buildings"],
                            vehicles=st.session_state["vehicles"],
                            characters=st.session_state["characters"],
                            items=st.session_state["items"],
                            species=st.session_state["species"],
                            organizations=st.session_state["organizations"],
                            conditions=st.session_state["conditions"],
                            dramas=st.session_state["dramas"],
                            myths=st.session_state["myths"],
                            gm_personality=st.session_state["gm_personality"],
                            description_pcs=st.session_state["description_pcs"])
    return context


def display_knowledge_graph():
    # Convert the nodes_edges string into a dictionary
    data = ast.literal_eval(st.session_state.nodes_edges.replace("{{", "{").replace("}}", "}"))

    # Create a new network instance without heading to prevent duplication
    g = Network(height='600px', width='100%', bgcolor="#333333", font_color="white")

    # Define a color scheme for nodes based on labels
    node_colors = {
        'Place': 'red',
        'Building': 'blue',
        'Character': 'green',
        'Vehicle': 'yellow',
        'Organization': 'purple',
        'Item': 'orange',
        'Species': 'cyan'
    }

    # Define a color scheme for edges based on relationship types
    edge_colors = {
        'IS_IN': 'lightgray',
        'OWNER_OF': 'black',
        'POSITIVE_RELATIONSHIP': 'green',
        'NEGATIVE_RELATIONSHIP': 'red',
        'AFFECTED_BY': 'yellow',
        'MEMBER_OF': 'blue',
        'BASED_ON': 'cyan',
        'SUCCEEDS': 'purple',
        'PRECEDES': 'pink'
    }

    # Process nodes and edges from the data dictionary
    for item in data['nodes']:
        node, attributes = item
        color = node_colors.get(attributes['label'], 'gray')  # Default to gray if not found
        g.add_node(node, label=node, title=attributes['description'], color=color)

    for item in data['edges']:
        source, target, relationship = item
        color = edge_colors.get(relationship, 'gray')  # Default to gray if not found
        g.add_edge(source, target, title=relationship, color=color)

    # Save the graph to a temporary file
    temp_file_name = "temp_graph.html"
    g.save_graph(temp_file_name)

    # Use streamlit to display the HTML
    st.components.v1.html(open(temp_file_name, 'r').read(), height=600)


def roll_atribute(attribute: int, test_type: str):
    dice = rd.randint(1, 6)

    if test_type == "physical":
        if dice < attribute:
            return "Success"
        elif dice == attribute:
            return "Critical Success"
        else:
            return "Failure"
    if test_type == "mental":
        if dice < attribute:
            return "Failure"
        elif dice == attribute:
            return "Critical Success"
        else:
            return "Success"
    else:
        return "Invalid type of dice"


def roll_d6():
    """
    Simulates a roll of a standard 6-sided dice (d6) and returns the result.

    Returns:
    - str: The outcome of the dice roll, a value between "1" and "6" inclusive.
    """
    dice = rd.randint(1, 6)
    return str(dice)


st.title('Generative Worldbuilding with Large Language Models')

with st.sidebar:
    st.title('API Key')
    if ('OPENAI_API_KEY' in st.secrets):
        st.success('OpenAI API Key already provided!', icon='✅')
        OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
    else:
        OPENAI_API_KEY = st.text_input('Enter OpenAI API key:', type='password')
        if not (OPENAI_API_KEY):
            st.warning('Please enter your key!', icon='⚠️')
        else:
            st.success('Proceed!', icon='👉')

if 'adventure_started' not in st.session_state:
    st.session_state.adventure_started = False

if not st.session_state.adventure_started:
    # User Input
    st.session_state.genre = st.text_input("WHAT IS THE GENRE?", "cyberpunk fantasy")
    st.session_state.feels = st.text_input("HOW DOES THE WORLD FEEL?", "a hopeless feeling")
    st.session_state.tone = st.text_input("WHAT IS THE TONE OF THE WORLD?", "dark")
    st.session_state.technology = st.text_input("WHAT IS THE LEVEL OF TECHNOLOGY?", "futuristic")
    st.session_state.magic = st.text_input("WHAT IS THE LEVEL OF MAGIC?", "something ultra-rare and forgotten")
    st.session_state.atmosphere = st.text_input("WHAT IS THE ATMOSPHERE?", "dystopic")
    st.session_state.writers = st.text_input("WHAT WRITERS THE STORY IS INSPITED BY?",
                  "Isaac Asimov, Greg Bear, Frank Herbert, William Gibson, Philip K. Dick")
    st.session_state.universes = st.text_input("WHAT UNIVERSES THE STORY IS INSPITED BY?",
                  "Cyberpunk 2077, Lancer RPG, BladeRunner, Ghost in the Shell, Cowboy Bebop")
    st.session_state.subjects_of_drama = st.text_area("WHAT SUBJECTS OF DRAMA THE STORY IS BASED ON?",
                 "Governance and its influence, Legal principles and rule of law, Provision of social services, Economic dynamics, Equity in wealth distribution, Agricultural practices and trade, Interpersonal relationships influenced by race, class, gender or sexual orientation, Military power and influence, Role and influence of religion, Technological advances and influences, Impact of arts and cultural expressions, Geographical features, Interactions between civilizations, their histories and conflicts, Foundations of the laws of nature, Theories about the origin of the universe, History of species and cultures that have inhabited the world, Needs and interactions of different species and cultures")
    st.session_state.addicional_info = st.text_area("FEEL FREE TO ADD ANY ADDICIONAL REQUESTS")

    st.session_state.num_scenes = st.slider("Number of Scenes:", value=7, min_value=1, max_value=20)

    st.session_state.story_structure = st.text_area("WHAT IS THE STRUCTURE OF THE ADVENTURE? DESCRIBE EACH SCENE", """- Scene 1: Motivation
        - Role-playing scene.
        - Start with a normal situation with no imminent danger.
        - Allow the players to get comfortable before the action begins.
        - Role-playing scene
        - Present a plot hook, something to get the characters out of their normal situation and start the action.
        - A clear goal and convincing reasons to act are needed.
    
    - Scene 2: Partial Victory
        - Action scene.
        - The characters encounter the first challenge.
        - They must have a clear but not total victory to recognize their abilities.
    
    - Scene 3: Information and Development
        - Exploration scene.
        - Characters figure out a plan to solve the main problem.
        - Use role-play and exploration scenes to come to a conclusion about what to do next.
    
    - Scene 4: Partial Defeat
        - Action scene.
        - Plan a setback for the characters to make the narrative more interesting.
        - It could be a more difficult fight, a failure of another kind, or even a betrayal.
        - Characters should be encouraged to think and strategize.
    
    - Scene 5: Conditions for Victory
        - Exploration scene.
        - Characters figure out how to win once and for all.
        - There may be preparation for the final confrontation through allies, equipment, or strategies.
    
    - Scene 6: Total Victory
        - Action scene.
        - This is the climax of the adventure, where the adventurers face the threat with a chance to win.
        - Value the players' tactics, but the villains must be dangerous.
        - Previous preparation must have been crucial to victory.
    
    - Scene 7: Resolution, Rewards, and Hooks
        - Role-playing scene
        - Everything returns to normal after the heroes' victory.
        - Characters evolve in terms of powers and equipment.
        - Rewards can be distributed and hooks can be inserted for future campaigns.""")

    st.session_state.places = st.slider("Places:", value=1, min_value=0, max_value=5)
    st.session_state.buildings = st.slider("Buildings:", value=8, min_value=0, max_value=20)
    st.session_state.vehicles = st.slider("Vehicles:", value=1, min_value=0, max_value=10)
    st.session_state.characters = st.slider("Characters:", value=16, min_value=0, max_value=50)
    st.session_state.items = st.slider("Items:", value=2, min_value=0, max_value=20)
    st.session_state.species = st.slider("Species:", value=3, min_value=0, max_value=10)
    st.session_state.organizations = st.slider("Organizations:", value=3, min_value=0, max_value=10)
    st.session_state.conditions = st.slider("Conditions:", value=1, min_value=0, max_value=5)
    st.session_state.dramas = st.slider("Dramas:", value=2, min_value=0, max_value=5)
    st.session_state.myths = st.slider("Myths:", value=2, min_value=0, max_value=20)

    st.session_state.gm_personality = st.text_area("GM Personality", "")

    # Initialize the players count if it's not already in session_state
    if 'num_players' not in st.session_state:
        st.session_state.num_players = 0

    # Button to add more players
    if st.button('Add Player'):
        st.session_state.num_players += 1

    if st.button('Remove Player') and st.session_state.num_players > 0:
        st.session_state.num_players -= 1

    # Display input fields for each player based on the current count
    for i in range(st.session_state.num_players):
        st.text_input(f"Name of Player {i+1}", key=f"name_pc_{i+1}")
        st.text_area(f"Description of Player {i+1}", key=f"description_pc_{i+1}")
        st.slider(f"Balance Point of Player {i+1}", value=3, min_value=2, max_value=5, key=f"balance_point_pc_{i+1}")

    if st.button('Start Adventure'):
        description_pcs = ""
        for i in range(st.session_state.num_players):
            player_name = st.session_state[f"name_pc_{i + 1}"]
            player_description = st.session_state[f"description_pc_{i + 1}"]
            balance_point = st.session_state[f"balance_point_pc_{i + 1}"]
            description_pcs += "**{player_name}**: {player_description}\n" \
                               "Balance Point:{balance_point}\n\n".format(player_name=player_name,
                                                                      player_description=player_description,
                                                                      balance_point=balance_point)

        st.session_state.description_pcs = description_pcs
        st.session_state.adventure_started = True

else:
    # Initialize session state variables if they don't exist
    if 'current_scene' not in st.session_state:
        st.session_state.current_scene_num = 1
        st.session_state.current_scene = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history_LLM = []
    if "nodes_edges" not in st.session_state:
        st.session_state.nodes_edges = ""

    if not st.session_state.nodes_edges:
        chat_openai = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k")
        #, temperature=0.6, top_p=0.7)

        # dice_desc = """
        # Simulates a dice roll for an RPG, determining outcomes based on a player's balance point and the type of test.
        #
        # Parameters:
        # - balance_point (int): The character's number that indicates the balance between their Mental and Physical abilities.
        # - test_type (str): The type of test being taken, either "physical" or "mental".
        #
        # Returns:
        # - str: The outcome of the dice roll. Possible values include "Success", "Critical Success", "Failure",
        #        or "Invalid type of dice".
        #
        # Outcomes:
        # - For a "physical" test:
        #   * Roll < attribute: "Success"
        #   * Roll = attribute: "Critical Success"
        #   * Roll > attribute: "Failure"
        # - For a "mental" test:
        #   * Roll < attribute: "Failure"
        #   * Roll = attribute: "Critical Success"
        #   * Roll > attribute: "Success"
        # """
        # tools = [Tool(name="RollDice", func=roll_dice, description=dice_desc)]
        # agent = initialize_agent(tools, llm=chat_openai, agent=AgentType.OPENAI_FUNCTIONS)

        nodes_edges, intro = generate_nodes_edges(chat_openai)
        st.session_state.nodes_edges = nodes_edges
        st.session_state.current_context = intro

        st.session_state.chat_history.append(("Game Master", intro))

        st.session_state.current_scene = generate_scene(chat_openai)

        st.session_state.chat_openai = chat_openai

    display_knowledge_graph()

    # Display chat messages
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    prompt = st.chat_input("Your Turn!")
    if prompt:
        st.session_state.chat_history.append(("User", prompt))
        with st.chat_message("User"):
            st.write(prompt)

        st.session_state.chat_history_LLM.append(HumanMessagePromptTemplate.from_template(prompt))

    # Generate a new response if last message is not from assistant
    if st.session_state.chat_history[-1][0] == "User":
        with st.chat_message("Game Master"):
            with st.spinner("Thinking..."):
                response = generate_response(st.session_state.chat_openai)
                st.session_state.chat_history.append(("Game Master", response))
                with st.chat_message("Game Master"):
                    st.write(response)

                st.session_state.chat_history_LLM.append(AIMessagePromptTemplate.from_template(response))
                if "**End of the Scene**" in response:
                    # Generate a summary using GPT-3 or predefined logic
                    if st.session_state.current_scene_num == st.session_state.num_scenes:
                        st.session_state.chat_history.append(("Game Master", "**End of the Adventure**"))
                        with st.chat_message("Game Master"):
                            st.write("**End of the Adventure**")
                        # todo: fazer alguma coisa com isso aqui:
                        # st.session_state.end_adventure = True
                        # st.session_state.adventure_started = False
                    else:
                        st.session_state.current_context = update_context(st.session_state.chat_openai)
                        st.session_state.nodes_edges = update_nodes_edges(st.session_state.chat_openai)

                        st.session_state.current_scene_num += 1
                        st.session_state.chat_history_LLM = []

                        st.session_state.current_scene = generate_scene(st.session_state.chat_openai)
