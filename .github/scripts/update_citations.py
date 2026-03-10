import re
import requests
from bs4 import BeautifulSoup


def get_citation_count():
    url = (
        "https://scholar.google.com/scholar?oi=bibs&hl=en"
        "&cites=4325808232671020176,16284230108871951652&as_sdt=5"
    )

    # Use a real browser User-Agent to avoid being blocked
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the total results count
        results_div = soup.find('div', {'id': 'gs_ab_md'})
        if results_div:
            text = results_div.get_text()
            match = re.search(r'About\s+(\d+)\s+results', text)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error fetching citations: {e}")
    return None


def update_readme(citation_count):
    if citation_count is None:
        return

    readme_path = "README.md"
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update the citations badge
    new_content = re.sub(
        r'\[\!\[Citations\]\(https://img\.shields\.io/badge/Citations-\d+-blue\)\]',
        f'[![Citations](https://img.shields.io/badge/Citations-{citation_count}-blue)]',
        content
    )

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


if __name__ == "__main__":
    citations = get_citation_count()
    if citations:
        update_readme(citations)
        print(f"Updated citation count to: {citations}")
    else:
        print("Failed to update citations")
