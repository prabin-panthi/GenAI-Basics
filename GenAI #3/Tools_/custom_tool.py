from langchain.tools import tool

@tool
def wish_morning(name : str):
    """A custom tool which is used to wish good morning message to {name} provided"""
    return f"Good morning {name}, hope you will have a great day today. Be happy and stay positive."

result = wish_morning.invoke({"name": "Bill"})
print(result)

print(wish_morning.name)
print(wish_morning.description)
print(wish_morning.args)