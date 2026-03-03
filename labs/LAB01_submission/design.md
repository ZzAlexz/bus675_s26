# Game Design Document

## Theme / Setting
- An abandoned Victorian mansion filled with restless spirits and cursed relics.

## Player's Goal
- Collecting ritual items and escaping before sunrise.

## Locations (4-6)
- Entry hall
- Living Room
- Dining Room
- Kitchen
- Study/Library
- Cellar

```
   [  Study / Library]
            |  (north / south)
   [X Dining Room    ]--- [X Living Room    ]--->[  Entry Hall     ]--- [  Kitchen        ]
                                                                                 |  (down / up)
                                                                        [X Cellar         ]
```

## Enemies (2-4 types)
- Wandering Spirit (weak/beginnger level enemy) - ghost like
- Possessed Doll (medium enemy) - inspired to be Chucky like (never watched the movie but know the doll)
- Shadow Figure - (Final boss) - imagined something along the lines of Stranger Things

## Win Condition
- Collect the artifacts: Ancient Tome (found in the Living Room) and Silver Locket (found in the Study), perform ritual in correct room (Cellar), escape through entry hall.

## Lose Condition
- Players are not able to complete ritual by runrise (start 8pm, Sunrise 6am). Every round costs 0.5hr meaning that there is max 20 rounds.
- Health meter (0-100) if it reaches 0, player loses
- Fear meter (0-100) if it reaches 100, player loses

## Class Hierarchy
```
Character (base class)
├── Player
└── Enemy
    ├── Possessed Doll (weak enemy)
    ├── Wandering Spirit (medium enemy)
    └── Shadow Figure (final challenge)

Location
├── Entry Hall
├── Living Room
├── Dining Room
├── Kitchen
├── Study/Library
├── Cellar

Item 
├── Ancient Tome
├── Silver Locket

Game (manages game state and loop)
├── Time ticker
├── Health meter
├── Fear meter
```

## Additional Notes
[Any other design decisions, ideas, or plans]
