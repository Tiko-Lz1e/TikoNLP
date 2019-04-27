import pickle

f1 = pickle.load(open('./data/pos_words.pkl', 'rb'))
f2 = pickle.load(open('./data/neg_words.pkl', 'rb'))
f = pickle.load(open('./data/sentences.pkl', 'rb'))

print(len(f1))
print(len(f2))
print(len(f))
print(f1)
print(f2)
print(f)