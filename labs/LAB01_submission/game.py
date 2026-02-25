"""
Lab 1: Text-Based Adventure RPG
================================
YOUR NAME HERE

Build your game here! This file contains all the starter code from the lab notebook.
Fill in the TODOs, add your own classes, and make it your own.

Run with: python game.py
"""

import random


# =============================================================================
# Dice Utilities
# =============================================================================

def roll_d20():
    """Roll a 20-sided die."""
    return random.randint(1, 20)


def roll_dice(num_dice, sides):
    """Roll multiple dice and return the total. E.g., roll_dice(2, 6) for 2d6."""
    return sum(random.randint(1, sides) for _ in range(num_dice))


# =============================================================================
# Character Classes
# =============================================================================

class Character:
    """Base class for all characters in the game."""

    def __init__(self, name, health, strength, defense):
        # TODO: Initialize attributes
        pass

    def is_alive(self):
        # TODO: Return True if health > 0
        pass

    def take_damage(self, amount):
        # TODO: Reduce health, but don't go below 0
        # Print a message about the damage taken
        pass

    def attack(self, target):
        # TODO: Implement d20 combat
        # 1. Roll d20, add strength
        # 2. Compare to target's defense
        # 3. If hit, deal damage to target
        # 4. Print combat messages!
        pass

    def __str__(self):
        return f"{self.name} (HP: {self.health}/{self.max_health})"


class Player(Character):
    """The player character."""

    def __init__(self, name):
        # TODO: Call parent __init__ with appropriate starting stats
        # TODO: Initialize inventory as empty list
        pass

    def pick_up(self, item):
        # TODO: Add item to inventory
        pass

    def show_inventory(self):
        # TODO: Display all items in inventory
        pass


class Enemy(Character):
    """Base class for enemies."""

    def __init__(self, name, health, strength, defense, xp_value=10):
        # TODO: Call parent __init__
        # TODO: Set xp_value
        pass


# TODO: Create specific enemy types (Minion, Elite, Boss)
# Example:
# class Goblin(Enemy):
#     def __init__(self):
#         super().__init__("Goblin", health=15, strength=3, defense=8, xp_value=5)


# =============================================================================
# Location Class
# =============================================================================

class Location:
    """A location in the game world."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.connections = {}  # {"north": Location, "south": Location, etc.}
        self.enemies = []      # List of enemies in this location
        self.items = []        # List of items in this location

    def describe(self):
        """Print a full description of the location."""
        print(f"\n{'='*50}")
        print(f"📍 {self.name}")
        print(f"{'='*50}")
        print(self.description)

        # TODO: Show enemies if present
        # TODO: Show items if present
        # TODO: Show available exits

    def get_exits(self):
        """Return a list of available directions."""
        return list(self.connections.keys())

    def add_connection(self, direction, location):
        """Connect this location to another."""
        self.connections[direction] = location


# =============================================================================
# World Builder
# =============================================================================

def create_world():
    """Create and connect all locations. Returns the starting location."""

    # Create locations
    # village = Location(
    #     "The Village",
    #     "A peaceful village with thatched-roof cottages. Smoke rises from chimneys."
    # )

    # TODO: Create more locations (4-6 total)

    # Connect locations (remember to connect both ways!)
    # village.add_connection("north", castle)
    # castle.add_connection("south", village)

    # TODO: Add enemies to locations
    # dungeon.enemies.append(Goblin())

    # TODO: Add items to locations (optional)

    # Return the starting location
    # return village
    pass


# =============================================================================
# Combat System
# =============================================================================

class Combat:
    """Manages turn-based combat between player and enemy."""

    # Combat states
    PLAYER_TURN = "player_turn"
    ENEMY_TURN = "enemy_turn"
    COMBAT_END = "combat_end"

    def __init__(self, player, enemy):
        self.player = player
        self.enemy = enemy
        self.state = Combat.PLAYER_TURN
        self.combat_log = []

    def start(self):
        """Begin combat and run until someone wins/loses/flees."""
        print(f"\n⚔️ COMBAT BEGINS! ⚔️")
        print(f"{self.player.name} vs {self.enemy.name}!")

        while self.state != Combat.COMBAT_END:
            if self.state == Combat.PLAYER_TURN:
                self.player_turn()
            elif self.state == Combat.ENEMY_TURN:
                self.enemy_turn()

        return self.get_result()

    def player_turn(self):
        """Handle player's turn in combat."""
        print(f"\n{self.player} | {self.enemy}")
        print("What do you do? (attack / run)")

        action = input("> ").lower().strip()

        if action == "attack":
            self.player.attack(self.enemy)
            if not self.enemy.is_alive():
                print(f"\n🎉 {self.enemy.name} has been defeated!")
                self.state = Combat.COMBAT_END
            else:
                self.state = Combat.ENEMY_TURN

        elif action == "run":
            # 50% chance to escape
            if random.random() < 0.5:
                print("You successfully fled!")
                self.state = Combat.COMBAT_END
            else:
                print("Couldn't escape!")
                self.state = Combat.ENEMY_TURN

        else:
            print("Invalid action. Try 'attack' or 'run'.")

    def enemy_turn(self):
        """Handle enemy's turn in combat."""
        print(f"\n{self.enemy.name}'s turn...")
        self.enemy.attack(self.player)

        if not self.player.is_alive():
            print(f"\n💀 {self.player.name} has fallen!")
            self.state = Combat.COMBAT_END
        else:
            self.state = Combat.PLAYER_TURN

    def get_result(self):
        """Return the combat result: 'victory', 'defeat', or 'fled'."""
        if not self.enemy.is_alive():
            return "victory"
        elif not self.player.is_alive():
            return "defeat"
        else:
            return "fled"


# =============================================================================
# Main Game Class
# =============================================================================

class Game:
    """Main game controller."""

    # Game states
    EXPLORING = "exploring"
    IN_COMBAT = "in_combat"
    GAME_OVER = "game_over"
    VICTORY = "victory"

    def __init__(self):
        self.player = None
        self.current_location = None
        self.state = Game.EXPLORING
        self.game_running = True

    def start(self):
        """Initialize and start the game."""
        self.show_intro()
        self.create_player()
        self.current_location = create_world()  # Your function from earlier
        self.current_location.describe()

        # Main game loop
        while self.game_running:
            if self.state == Game.EXPLORING:
                self.exploration_loop()
            elif self.state == Game.GAME_OVER:
                self.show_game_over()
                break
            elif self.state == Game.VICTORY:
                self.show_victory()
                break

    def show_intro(self):
        """Display the game introduction."""
        print("\n" + "="*60)
        print("         YOUR GAME TITLE HERE")
        print("="*60)
        print("\nYour epic intro text goes here...")
        print("Set the scene! What's happening? Why is the player here?")
        print("\n" + "="*60)

    def create_player(self):
        """Create the player character."""
        print("\nWhat is your name, adventurer?")
        name = input("> ")
        self.player = Player(name)
        print(f"\nWelcome, {name}! Your adventure begins...")

    def exploration_loop(self):
        """Handle player input during exploration."""
        print("\nWhat do you do? (type 'help' for commands)")
        command = input("> ").lower().strip()

        # Parse the command
        parts = command.split()
        if not parts:
            return

        action = parts[0]

        if action == "help":
            self.show_help()

        elif action == "look":
            self.current_location.describe()

        elif action == "go" and len(parts) > 1:
            direction = parts[1]
            self.move(direction)

        elif action in ["north", "south", "east", "west", "up", "down"]:
            self.move(action)

        elif action in ["fight", "attack"]:
            self.initiate_combat()

        elif action in ["inventory", "i"]:
            self.player.show_inventory()

        elif action == "quit":
            print("Thanks for playing!")
            self.game_running = False

        else:
            print("I don't understand that command. Type 'help' for options.")

    def move(self, direction):
        """Move the player in the specified direction."""
        if direction in self.current_location.connections:
            self.current_location = self.current_location.connections[direction]
            self.current_location.describe()
            # TODO: Check for automatic combat triggers?
        else:
            print(f"You can't go {direction} from here.")

    def initiate_combat(self):
        """Start combat with an enemy in the current location."""
        if not self.current_location.enemies:
            print("There's nothing to fight here.")
            return

        enemy = self.current_location.enemies[0]  # Fight first enemy
        battle = Combat(self.player, enemy)
        result = battle.start()

        if result == "victory":
            self.current_location.enemies.remove(enemy)
            # TODO: Check for victory condition (e.g., boss defeated)
        elif result == "defeat":
            self.state = Game.GAME_OVER

    def show_help(self):
        """Display available commands."""
        print("\n📜 AVAILABLE COMMANDS:")
        print("  go [direction] - Move in a direction (north, south, east, west)")
        print("  look          - Examine your surroundings")
        print("  fight         - Attack an enemy in this location")
        print("  inventory     - Check your inventory")
        print("  help          - Show this help message")
        print("  quit          - Exit the game")

    def show_game_over(self):
        """Display game over message."""
        print("\n" + "="*60)
        print("                    GAME OVER")
        print("="*60)
        print("\nYou have fallen. The adventure ends here...")
        print("\n(But you can always try again!)")

    def show_victory(self):
        """Display victory message."""
        print("\n" + "="*60)
        print("                    🎉 VICTORY! 🎉")
        print("="*60)
        print("\nCongratulations! You have completed your quest!")
        # TODO: Add your custom victory text


# =============================================================================
# Run the Game
# =============================================================================

if __name__ == "__main__":
    game = Game()
    game.start()
