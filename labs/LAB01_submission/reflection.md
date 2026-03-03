# Reflection: OOP Design Decisions

Write 2-3 paragraphs reflecting on your object-oriented design. Some questions to consider:

- Why did you structure your classes the way you did?
- What inheritance relationships did you use and why?
- What was challenging about managing multiple interacting objects?
- If you had more time, what would you refactor or add?
- How does this experience connect to working with OOP in analytics/ML codebases?

---

Following the flow of Lecture 4, I was able to create the class structure in a linear fashion. This allowed for a simplified understanding of what needed to be created and why, fleshing out the haunted mansion idea by taking one class at a time. Creating a class one at a time allowed me to build complex inheritance hierarchies for each individual Enemy. This allowed me to flesh out each character further and personalize them, and generalize the traits they are meant to maintain when needed. This helped simplify the code needed to create each individual character and granted personalization when needed.

Although creating a haunted house game inspired by my girlfriend's love of haunted movies (and my fear of them) was a challenge, it did call for some challenges along the way. Ensuring the right definition was called when going through the house was difficult, as it had to connect correctly with the other definitions without anything crashing. However, with the help of some vibe coding and reviewing our lectures, we were able to create a well-formatted game. If given the chance, I would have loved to expand the map significantly into a maze-like system that feels like a never-ending loop (think of Mario going up the stairs nonstop in Super Mario 64), while also adding more enemies or side quests. Additionally, I would have loved to play further into the haunted-house theme by adding some really scary elements, like a random jump scare. 

Building this game also gave me a clearer picture of how OOP shows up in real analytics and ML work. Managing the various classes, such as Game, Player, Combat, and Location, and having them all talking to each other felt a lot like how a data pipeline is structured, rather than the traditional queries and dashboard work it is believed analysts do. This hands-on approach made the comprehension of structuring data and how they interact with one another significantly more engaging. The more I got into the weeds of the program, the more I fleshed out the idea for the game I wanted to create.  

