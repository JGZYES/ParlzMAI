# -*- coding: utf-8 -*-
"""生成一份多轮闲聊对话语料（User:/Assistant: 格式），用于训练聊天模型。"""
import random


def build_pools():
    P = {}
    P["names"] = ["Alice", "Bob", "Sam", "Luna", "Max", "Emma", "Kai", "Nora", "Leo", "Mia"]
    P["hobbies"] = ["reading", "hiking", "cooking", "painting", "chess", "yoga", "photography", "fishing", "gaming", "gardening"]
    P["foods"] = ["pizza", "sushi", "ramen", "tacos", "pasta", "salad", "biryani", "dumplings", "curry", "pancakes"]
    P["cities"] = ["Tokyo", "Paris", "Berlin", "Lisbon", "Seoul", "Toronto", "Sydney", "Rome", "Kyoto", "Prague"]
    P["weather"] = ["sunny", "rainy", "cloudy", "snowy", "windy", "clear"]
    P["movies"] = ["sci-fi", "comedy", "drama", "horror", "adventure", "romance"]
    P["music"] = ["pop", "jazz", "rock", "classical", "hip-hop", "folk"]
    P["ports"] = ["pets", "cats", "dogs", "birds", "fish"]
    return P


def gen_conversation(P, rng):
    n_turns = rng.randint(3, 6)
    turns = []
    for i in range(n_turns):
        if i % 2 == 0:  # user
            r = rng.random()
            if r < 0.20:
                u = rng.choice(["Hi", "Hello", "Hey", "Good morning", "What's up", "Hi there"])
            elif r < 0.35:
                u = f"How are you doing" + ("?" if rng.random() < 0.7 else "")
            elif r < 0.5:
                u = f"I like {rng.choice(P['hobbies'])}. What about you?"
            elif r < 0.62:
                u = f"Have you been to {rng.choice(P['cities'])}?"
            elif r < 0.72:
                u = f"What kind of {rng.choice(['food', 'music', 'movies'])} do you like?"
            elif r < 0.8:
                u = f"Do you have any {rng.choice(P['ports'])}?"
            elif r < 0.9:
                u = f"Can you tell me a {rng.choice(['joke', 'fact', 'story'])}?"
            else:
                u = "That's nice. Tell me more."
            turns.append(("User", u))
        else:  # assistant
            r = rng.random()
            if r < 0.2:
                a = rng.choice(["Hi! How can I help you today?", "Hello! It's good to hear from you.", "Hey there! What's on your mind?"])
            elif r < 0.35:
                a = rng.choice(["I'm doing great, thanks for asking!", "Pretty good! How about you?", "Can't complain. How are you?"])
            elif r < 0.5:
                a = f"That sounds fun! I enjoy {rng.choice(P['hobbies'])} too."
            elif r < 0.62:
                a = f"I haven't been to {rng.choice(P['cities'])}, but I've heard it's lovely."
            elif r < 0.72:
                a = f"I'd say I like {rng.choice(P['music'])} and {rng.choice(P['movies'])}. You?"
            elif r < 0.8:
                a = rng.choice(["I do! I really like dogs and cats.", "I think pets are wonderful companions."])
            elif r < 0.9:
                a = rng.choice(["Here's one: why did the programmer quit? Because he didn't get arrays.", "Did you know honey never spoils?", "Let me tell you a short story about a curious little robot."])
            else:
                a = rng.choice(["That makes sense.", "I see what you mean.", "Interesting point!", "Thanks for sharing that."])
            turns.append(("Assistant", a))
    return turns


def main(out="data/chat.txt", n=4000, seed=7):
    random.seed(seed)
    P = build_pools()
    lines = []
    for _ in range(n):
        for role, text in gen_conversation(P, random):
            lines.append(f"{role}: {text}")
        lines.append("")  # 空行分隔对话
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"chat corpus -> {out} : {sum(len(l) for l in lines)} chars, {n} conversations")


if __name__ == "__main__":
    import sys
    main(out=sys.argv[1] if len(sys.argv) > 1 else "data/chat.txt")
