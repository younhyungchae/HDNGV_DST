import string

intent_classification_prompt = string.Template("""Classify the intent of the utterance of user among A, B, C, D, E.

A: User ask to take photo.
B: User ask for the specific information of car.
C: User wants to confirm if the information he or she asks is correct.
D: User wants to compare more than one cars.
E: User asks for an general information regardless of specific car.

Utterance: Would you take a photo of me?
Intent: A

Utterance: What does steering wheel mean?
Intent: E

Utterance: Would you explain the fuel efficiency of that car?
Intent: B

Utterance: Which one is cheaper between this car and SUV?
Intent: D

Utterance: This car is SUV. Right?
Intent: C

Utterance: How much is this model?
Intent: B

Utterance: Take a photo of me and this car please.
Intent: A

Utterance: Would you explain difference between this car and that black SUV?
Intent: D

Utterance: This car is cheaper than that model. Right?
Intent: C

Utterance; What does fuel means?
Intent: E

Utterance: Would explain what hybrid car is?
Intent: E

Utterance: Which one is the most appropriate for the long trip among these cars?
Intent: D

Utterance: What is the name of this car?
Intent: B

Utterance: Can you take a photo of us?
Intent: A

Utterance: This car is under 50000 dollars, right?
Intent: C

Utterance: $utterance
Intent: """)

has_reference_expression_prompt = string.Template("""Assume you are given an image. If following sentence includes referencing expression which references specific object in image, answer True, else False.
ray, k5, g80, gv80, spotage, tucson, grandeur, santafe, seltos, ioniq, sorento, niro, carnival, kona, avante are name of cars.

Sentence: How much is that yellow car?
Answer: True

Sentence: Which one is better between this car and santafe?
Answer: True

Sentence: Tell me the exterior color option of ioniq5.
Answer: False

Sentence: Would you take a picture of me near that black sorento?
Answer: True

Sentence: How much is gv80?
Answer: False

Sentence: Take a photo of me please.
Answer: False

Sentence: How much is that car?
Answer: True

Sentence: $utterance
Answer: """)

referring_object_prompt = string.Template("""Which referring expression does utterance talking about? If not given, answer None.
ray, k5, g80, gv80, spotage, tucson, grandeur, santafe, seltos, ioniq, sorento, niro, carnival, kona, avante are name of cars.

Utterance: How much is that grey car on the left?
Object: that grey car on the left

Utterance: How much is the ray?
Object: None

Utterance: What is that car in the middle?
Object: that car in the middle

Utterance: Would you take a picture of me next to that red gv70?
Object: that red gv70

Utterance: Which one is cheaper between this black SUV and Tucson?
Object: this black SUV

Utterance: Does Tucson has any convenient options?
Object: None

Utterance: Explain me about the first car from the left.
Object: the first car from the left

Utterance: How much is this car?
Object: this car

Utterance: Can you tell me what that red sedan is?
Object: that red sedan

Utterance: $utterance
Object: """)

photo_spot_prompt = string.Template("""According to the user utterance, where does user wants to take photo at? If not specified, answer none.

User Utterance: Would you take a photo next to the red SUV?
Position: next to the red SUV

User Utterance: Would you please take a picture of me?
Position: none

User Utterance: Take a picture of me in front of the this black sedan.
Position: in front of the this black sedan

User Utterance: $utterance
Position: """)

explain_car_name_prompt = string.Template("""According to the user utterance, what is the name of the car the user wants to explain? If exact name is not specified ,answer none.
ray, k5, g80, gv80, spotage, tucson, grandeur, santafe, seltos, ioniq, sorento, niro, carnival, kona, avante are name of cars.

User Utterance: How much is the white tucson?
Name: tucson_white

User Utterance: How's the fuel efficiency of the black seltos?
Name; seltos_black

User Utterance: How much is that yellow car?
Name: none

User Utterance: Do you have any santafe?
Name: santafe

User Utterance: Would you explain that blue SUV in the middle?
Name: none

User Utterance: Does grandeur have safety option?
Name: grandeur

User Utterance: What exterior color option does this car have?
Name: none

User Utterance: Would you explain the price of that black car?
Name: none 

User Utterance: Tell me how much the avante is.
Name: avante

User Utterance: $utterance
Name: """)

explain_info_type_prompt = string.Template("""According to the user utterance, what type of information is user asking for? If not specified, answer none.

User Utterance: Would you tell me the price of this black SUV?
InfoType: price

User Utterance: Is the fuel efficiency of this car good?
InfoType: fuel efficiency

User Utterance: Would you explain the engine of kona, please?
InfoType: engine

User Utterance: What is that car?
InfoType: car_name

User Utterance: What type of window does this car has?
InfoType: window

User Utterance: What are the exterior colors of the car on the left?
InfoType: exterior color

User Utterance: What are the convenience features of the car in the middle?
InfoType: convenience feature

User Utterance: $utterance
InfoType: """)

compare_target_prompt = string.Template("""In the user utterance, list the specific name of the cars that user wants to compare in comma-seperated format if mentioned. If the none of any specific name of the car is specifed, answer none.
ray, k5, g80, gv80, spotage, tucson, grandeur, santafe, seltos, ioniq, sorento, niro, carnival, kona, avante are name of cars.

User Utterance: Which one is better between this car and santafe?
Targets: santafe

User Utterance: Which car is cheaper between santafe and kona?
Targets: santafe, kona

User Utterance: Which car is better in terms of engine between white avante and black carnival?
Targets: avante, carnival

User Utterance: Which car do you think is better between seltos and grandeur black?
Targets: seltos, grandeur

User Utterance: Which car is more expensive between Tucson and this car?
Targets: Tucson

User Utterance: What is the difference between g80 and gv80?
Targets: g80, gv80

User Utterance: Between spotage and santafe, which has higher maximum speed?
Targets: spotage, santafe

User Utterance: Among seltos, ioniq, and this car, which one is the heaviest?
Targets: seltos, ioniq

User Utterance: Is niro more sold than this car or not?
Targets: niro

User Utterance: $utterance
Targets: """)

compare_info_prompt = string.Template("""According to the user utterance, which type of information does user want to compare? If the type is not specified, answer none.

User Utterance: Which one is better between this car and santafe?
Type: none

User Utterance: Which car is cheaper between santafe and kona?
Type: price

User Utterance: Which car is better in terms of engine between white avante and black ioniq5?
Type: engine

User Utterance: Which car do you think is better between seltos and grandeur black?
Type: none

User Utterance: What is the difference between g80 and gv80?
Type: difference

User Utterance: Between spotage and santafe, which has higher maximum speed?
Type: maximum speed

User Utterance: Among seltos, ioniq, and this car, which one is the heaviest?
Type: weight

User Utterance: $utterance
Type: """)

confirm_car_name_prompt = string.Template("""According to the user utterance, what is the name of the car which the user wants to comfirm the information about? If the name is not specified ,answer none.
ray, k5, g80, gv80, spotage, tucson, grandeur, santafe, seltos, ioniq, sorento, niro, carnival, kona, avante are name of cars.

User Utterance: The spotage has 4 doors, right?
Name: spotage

User Utterance: You said fuel efficiency of seltos is great. Am I right?
Name: seltos

User Utterance: I remember that you said this car has good audio system, right?
Name: none

User Utterance: You said the wheel size of ioniq5 is 4 inches, right?
Name: ioniq5

User Utterance: carnival is 4000 dollars, right?
Name: carnival

User Utterance: The options for interior color of niro has black, blue, white. Am I right?
Name: niro

User Utterance: You said grandeur is more preferred than santafe, right?
Name: grandeur

User Utterance: I heard that kona has good car design, right?
Name: kona

User Utterance: $utterance
Name: """)

confirm_info_prompt = string.Template("""According to the user utterance, what type of information does user want to confirm about the car? If not specified, answer none.
ray, k5, g80, gv80, spotage, tucson, grandeur, santafe, seltos, ioniq, sorento, niro, carnival, kona, avante are name of cars.

User Utterance: The spotage has 4 doors, right?
Type: door

User Utterance: You said fuel efficiency of seltos is great. Am I right?
Type: fuel efficiency

User Utterance: I remember that you said this white car has good audio system, right?
Type: audio system

User Utterance: You said the wheel size of ioniq5 is 4 inches, right?
Type: wheel size

User Utterance: carnival is 4000 dollars, right?
Type: price

User Utterance: This car is carnival. Am I right?
Type: coincidence

User Utterance: The options for interior color of niro has black, blue, white. Am I right?
Type: interior color

User Utterance: You said black grandeur is more preferred than santafe, right?
Type: preference

User Utterance: I heard that kona has good car design, right?
Type: design

User Utterance: $utterance
Type: """)

ask_info_prompt = string.Template("""According to the user utterance, what does user is asking about?

User Utterance: What is steering wheel?
Target: steering wheel

User Utterance: Would you explain what straight-four engine is?
Target; straight-four engine

User Utterance: Tell me what does hybrid car mean.
Target: hybrid car

User Utterance: Where is toilet?
Target: toilet_direction

User Utterance: $utterance
Target: """)

object_detecting_prompt = string.Template("""Provide a bounding box that matches following expression: $expression""")

gpt_prompt = string.Template("""You are a guide robot that guides users at a car showroom. You are able to take a photo. Respond to the user utterance considering following user intent and slots. If the information in slots is not sufficient, ask user for that informations.
Intent: $intent
Slot-Value: $slots
User Utterance: $utterance""")