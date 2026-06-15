import arxiv

search = arxiv.Search(
    query="electromagnetics and electrostatics",
    max_results=5
)

client = arxiv.Client()

for result in client.results(search):
    print("Title:", result.title)
    print("Summary:", result.summary)
    print("-" * 50)