import chromadb

# Connection
client = chromadb.HttpClient(host='localhost', port=8000)

# Nanosecond heartbeat.
client.heartbeat()

# Reset?
# client.reset()
