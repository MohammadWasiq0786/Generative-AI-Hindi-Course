from langserve import RemoteRunnable

chain=RemoteRunnable("http://localhost:8000/chain/c/N4XyA")
res=chain.invoke({"language":"hindi" ,"text":"what is generative ai"})
print(res)