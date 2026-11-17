"""Per-category aspect vocabularies.

Used twice: (1) to write campaign briefs / LLM prompts during injection, and
(2) as the keyword-based aspect extractor for the AO (aspect overlap) feature.
"""
import re

ASPECTS = {
    "Musical_Instruments": {
        "product": "musical instrument accessory",
        "aspects": {
            "sound": ["sound", "tone", "tonal", "audio"],
            "build quality": ["build", "construction", "sturdy", "solid", "flimsy"],
            "price": ["price", "value", "cheap", "expensive", "worth", "money"],
            "strings": ["string", "strings"],
            "tuning stability": ["tuning", "in tune", "tuner", "tune"],
            "packaging": ["packaging", "package", "box", "arrived", "shipping"],
            "durability": ["durable", "durability", "lasted", "last", "broke"],
            "noise": ["noise", "hum", "buzz", "hiss", "quiet"],
            "volume": ["volume", "loud", "louder"],
            "comfort": ["comfortable", "comfort", "feel", "ergonomic"],
            "setup": ["setup", "set up", "install", "easy to use", "plug"],
            "battery": ["battery", "batteries", "charge", "charging"],
            "cable": ["cable", "cord", "wire", "connector", "jack"],
            "pedal": ["pedal", "effect", "effects"],
            "size": ["size", "compact", "small", "portable", "light", "weight"],
            "finish": ["finish", "color", "colour", "looks", "design"],
            "customer service": ["seller", "customer service", "support", "refund", "return"],
            "beginner friendly": ["beginner", "learn", "learning", "student", "kid"],
        },
    },
    "All_Beauty": {
        "product": "beauty product",
        "aspects": {
            "scent": ["scent", "smell", "fragrance", "smells"],
            "texture": ["texture", "consistency", "creamy", "greasy", "sticky"],
            "price": ["price", "value", "cheap", "expensive", "worth", "money"],
            "packaging": ["packaging", "package", "bottle", "box", "arrived", "shipping"],
            "skin feel": ["skin", "soft", "smooth", "moistur", "hydrat"],
            "hair": ["hair", "curls", "frizz", "shine"],
            "long lasting": ["lasting", "lasts", "all day", "long"],
            "color": ["color", "colour", "shade", "pigment"],
            "irritation": ["irritat", "sensitive", "rash", "burn", "allerg"],
            "ingredients": ["ingredient", "natural", "organic", "chemical"],
            "application": ["apply", "application", "applies", "brush", "easy to use"],
            "size": ["size", "small", "travel", "large", "amount"],
            "results": ["result", "results", "difference", "works", "worked"],
            "customer service": ["seller", "customer service", "support", "refund", "return"],
        },
    },
}
# Yelp (if the user supplies the Yelp Open Dataset manually)
ASPECTS["Yelp"] = {
    "product": "local business (restaurant)",
    "aspects": {
        "food": ["food", "dish", "meal", "taste", "delicious", "flavor"],
        "service": ["service", "staff", "waiter", "waitress", "server", "friendly"],
        "price": ["price", "value", "cheap", "expensive", "worth"],
        "ambience": ["ambience", "atmosphere", "decor", "music", "vibe"],
        "wait time": ["wait", "waited", "slow", "quick", "fast"],
        "cleanliness": ["clean", "dirty", "hygien"],
        "portion size": ["portion", "size", "huge", "small"],
        "location": ["location", "parking", "located"],
        "drinks": ["drink", "drinks", "cocktail", "beer", "wine", "coffee"],
    },
}


def get(category):
    return ASPECTS.get(category, ASPECTS["Musical_Instruments"])


def extract(text, category):
    """Keyword aspect extractor: returns the set of aspect names mentioned in text."""
    t = text.lower()
    out = set()
    for name, kws in get(category)["aspects"].items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw), t):
                out.add(name)
                break
    return frozenset(out)
