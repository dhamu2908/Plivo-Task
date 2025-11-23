"""
Generate synthetic noisy STT transcripts with PII entities.
Reflects real speech-to-text patterns: spelled-out numbers, "dot", "at", no punctuation.
"""
import json
import random
from typing import List, Dict, Tuple

random.seed(42)

# Templates for noisy STT patterns
FIRST_NAMES = ["ramesh", "priya", "amit", "sneha", "rajesh", "kavya", "vijay", "anita", 
               "suresh", "deepa", "manoj", "pooja", "arun", "neha", "kiran", "sonal",
               "rahul", "meera", "ajay", "ritu", "john", "mary", "david", "sarah"]

LAST_NAMES = ["sharma", "kumar", "patel", "singh", "reddy", "krishnan", "mehta", "joshi",
              "gupta", "verma", "rao", "nair", "iyer", "das", "bose", "smith", "johnson"]

CITIES = ["chennai", "bangalore", "mumbai", "delhi", "hyderabad", "pune", "kolkata",
          "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal",
          "new york", "london", "paris", "tokyo", "singapore"]

LOCATIONS = ["mg road", "park street", "main street", "nehru nagar", "gandhi market",
             "railway station", "airport", "bus stand", "city center", "downtown",
             "central plaza", "shopping mall", "tech park", "industrial area"]

EMAIL_DOMAINS = ["gmail dot com", "yahoo dot com", "hotmail dot com", "outlook dot com",
                 "company dot co dot in", "tech dot com", "mail dot com"]

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

MONTH_SHORT = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

def number_to_words(num: int, digit_by_digit: bool = False) -> str:
    """Convert number to spoken words."""
    if digit_by_digit:
        digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        return " ".join(digits[int(d)] for d in str(num))
    
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
             "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if num < 10:
        return ones[num]
    elif num < 20:
        return teens[num - 10]
    elif num < 100:
        return tens[num // 10] + ("" if num % 10 == 0 else " " + ones[num % 10])
    elif num < 1000:
        return ones[num // 100] + " hundred" + ("" if num % 100 == 0 else " " + number_to_words(num % 100))
    else:
        return str(num)

def generate_credit_card() -> Tuple[str, str]:
    """Generate credit card in noisy STT format."""
    # Generate groups of 4 digits
    groups = []
    for _ in range(4):
        group = "".join(str(random.randint(0, 9)) for _ in range(4))
        groups.append(number_to_words(int(group), digit_by_digit=True))
    
    patterns = [
        f"{' '.join(groups)}",
        f"card number {' '.join(groups)}",
        f"credit card {' '.join(groups)}",
    ]
    text = random.choice(patterns)
    # Find actual number position
    cc_start = text.find(groups[0].split()[0])
    cc_end = text.rfind(groups[-1].split()[-1]) + len(groups[-1].split()[-1])
    return text, (cc_start, cc_end)

def generate_phone() -> Tuple[str, str]:
    """Generate phone number in noisy STT format."""
    digits = [random.randint(0, 9) for _ in range(10)]
    spoken = number_to_words(sum(d * (10 ** i) for i, d in enumerate(reversed(digits))), digit_by_digit=True)
    
    patterns = [
        f"my number is {spoken}",
        f"call me on {spoken}",
        f"phone number {spoken}",
        f"contact me at {spoken}",
        spoken,
    ]
    text = random.choice(patterns)
    # Find phone position
    phone_words = spoken.split()
    phone_start = text.find(phone_words[0])
    phone_end = text.rfind(phone_words[-1]) + len(phone_words[-1])
    return text, (phone_start, phone_end)

def generate_email() -> Tuple[str, str]:
    """Generate email in noisy STT format."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    domain = random.choice(EMAIL_DOMAINS)
    
    patterns = [
        f"{first} dot {last} at {domain}",
        f"{first} underscore {last} at {domain}",
        f"{first}{random.randint(1,99)} at {domain}",
        f"{first} {last} at {domain}",
    ]
    email_text = random.choice(patterns)
    
    templates = [
        f"my email is {email_text}",
        f"email me at {email_text}",
        f"contact {email_text}",
        f"send it to {email_text}",
        email_text,
    ]
    text = random.choice(templates)
    
    # Find email position
    email_start = text.find(email_text.split()[0])
    email_end = text.rfind(email_text.split()[-1]) + len(email_text.split()[-1])
    return text, (email_start, email_end)

def generate_person_name() -> Tuple[str, str]:
    """Generate person name."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    
    patterns = [
        f"{first} {last}",
        f"{first}",
        f"mister {first} {last}",
        f"miss {first} {last}",
        f"doctor {first} {last}",
    ]
    name = random.choice(patterns)
    
    templates = [
        f"my name is {name}",
        f"this is {name}",
        f"call me {name}",
        f"i am {name}",
        f"speaking with {name}",
        name,
    ]
    text = random.choice(templates)
    
    name_start = text.find(name.split()[0])
    name_end = text.rfind(name.split()[-1]) + len(name.split()[-1])
    return text, (name_start, name_end)

def generate_date() -> Tuple[str, str]:
    """Generate date in noisy STT format."""
    day = random.randint(1, 28)
    month = random.choice(MONTHS if random.random() > 0.3 else MONTH_SHORT)
    year = random.randint(1980, 2025)
    
    patterns = [
        f"{number_to_words(day)} {month} {number_to_words(year, digit_by_digit=True)}",
        f"{month} {number_to_words(day)} {number_to_words(year, digit_by_digit=True)}",
        f"{day} {month} {year}",
        f"{month} {day} {year}",
        f"{number_to_words(day)} of {month}",
    ]
    date = random.choice(patterns)
    
    templates = [
        f"on {date}",
        f"date is {date}",
        f"scheduled for {date}",
        f"meeting on {date}",
        date,
    ]
    text = random.choice(templates)
    
    date_start = text.find(date.split()[0])
    date_end = text.rfind(date.split()[-1]) + len(date.split()[-1])
    return text, (date_start, date_end)

def generate_city() -> Tuple[str, str]:
    """Generate city mention."""
    city = random.choice(CITIES)
    
    templates = [
        f"i live in {city}",
        f"from {city}",
        f"located in {city}",
        f"city is {city}",
        f"based in {city}",
        city,
    ]
    text = random.choice(templates)
    
    city_start = text.find(city)
    city_end = city_start + len(city)
    return text, (city_start, city_end)

def generate_location() -> Tuple[str, str]:
    """Generate location mention."""
    location = random.choice(LOCATIONS)
    
    templates = [
        f"at {location}",
        f"near {location}",
        f"on {location}",
        f"location is {location}",
        location,
    ]
    text = random.choice(templates)
    
    loc_start = text.find(location)
    loc_end = loc_start + len(location)
    return text, (loc_start, loc_end)

def generate_utterance(use_multiple: bool = True) -> Dict:
    """Generate a complete noisy STT utterance with multiple entities."""
    generators = {
        "CREDIT_CARD": generate_credit_card,
        "PHONE": generate_phone,
        "EMAIL": generate_email,
        "PERSON_NAME": generate_person_name,
        "DATE": generate_date,
        "CITY": generate_city,
        "LOCATION": generate_location,
    }
    
    # Choose 1-4 entity types
    num_entities = random.randint(1, 4 if use_multiple else 2)
    entity_types = random.sample(list(generators.keys()), num_entities)
    
    # Generate segments
    segments = []
    entities = []
    current_pos = 0
    
    for i, ent_type in enumerate(entity_types):
        # Add connector words occasionally
        if i > 0 and random.random() > 0.3:
            connector = random.choice(["and", "also", "my", "the", "plus", ""])
            if connector:
                segments.append(connector)
                current_pos += len(connector) + 1
        
        text_segment, (rel_start, rel_end) = generators[ent_type]()
        segments.append(text_segment)
        
        # Calculate absolute positions
        abs_start = current_pos + rel_start
        abs_end = current_pos + rel_end
        
        entities.append({
            "start": abs_start,
            "end": abs_end,
            "label": ent_type
        })
        
        current_pos += len(text_segment) + 1
    
    full_text = " ".join(segments)
    
    # Sort entities by start position
    entities.sort(key=lambda x: x["start"])
    
    return {
        "text": full_text,
        "entities": entities
    }

def generate_dataset(num_samples: int, id_prefix: str = "utt") -> List[Dict]:
    """Generate a complete dataset."""
    dataset = []
    for i in range(num_samples):
        utterance = generate_utterance(use_multiple=random.random() > 0.2)
        utterance["id"] = f"{id_prefix}_{i+1:04d}"
        dataset.append(utterance)
    return dataset

def save_jsonl(data: List[Dict], filepath: str, include_labels: bool = True):
    """Save data as JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            if not include_labels:
                # Remove entities for test set
                item = {"id": item["id"], "text": item["text"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("Generating PII NER datasets...")
    
    # Generate datasets
    train_data = generate_dataset(800, "train")
    dev_data = generate_dataset(150, "dev")
    test_data = generate_dataset(100, "test")
    
    # Save datasets
    save_jsonl(train_data, "data/train.jsonl", include_labels=True)
    save_jsonl(dev_data, "data/dev.jsonl", include_labels=True)
    save_jsonl(test_data, "data/test.jsonl", include_labels=False)
    
    print(f"✓ Generated {len(train_data)} training examples → data/train.jsonl")
    print(f"✓ Generated {len(dev_data)} dev examples → data/dev.jsonl")
    print(f"✓ Generated {len(test_data)} test examples (unlabeled) → data/test.jsonl")
    
    # Show sample
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
