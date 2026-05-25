
tesla_keywords = [
    'tesla','tsla','elon','musk',
    'cybertruck','model 3','model y',
    'gigafactory','fsd'
]

def tesla_relevance(text):
    text = str(text).lower()
    score = 0

    for keyword in tesla_keywords:
        if keyword in text:
            score += 1

    return score

def detect_topic(text):
    text = str(text).lower()

    if 'earnings' in text:
        return 'Earnings'
    elif 'elon' in text or 'musk' in text:
        return 'Elon Musk'
    elif 'delivery' in text:
        return 'Deliveries'
    elif 'bitcoin' in text:
        return 'Bitcoin'
    elif 'autopilot' in text:
        return 'Autopilot'
    elif 'stock' in text or 'price' in text:
        return 'Stock Market'
    else:
        return 'General Tesla'
