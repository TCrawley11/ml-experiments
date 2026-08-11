import requests

def get_wiki_plaintext(title, filename=None):
    """Fetches a Wikipedia article as clean plain text."""

    HEADERS = {
        'User-Agent': 'MyLLMTrainingProject (your.email@example.com)'
    }
    
    # 1. Define the API endpoint and parameters
    API_URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,  # This is the key for plain text
        "formatversion": 2    # Makes the JSON easier to parse
    }

    # 2. Make the API request
    try:
        response = requests.get(API_URL, params=params, headers=HEADERS)
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
    except requests.RequestException as e:
        print(f"Error making API request: {e}")
        return

    # 3. Extract the text content
    if 'query' in data and 'pages' in data['query']:
        page = data['query']['pages'][0] # Should be the first (and only) page

        if 'extract' in page:
            text_content = page['extract']

            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"Successfully saved '{title}' to {filename}")
            else:
                return text_content
        else:
            print(f"Error: Could not find text for page '{title}'. Check the title is correct.")
    else:
        print("Error: Invalid response from API.")


get_wiki_plaintext("History_of_China", "historyOfChina.txt")