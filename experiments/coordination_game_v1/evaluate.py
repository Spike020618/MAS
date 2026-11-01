import re,string

def normalize(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',str(s).lower().translate(str.maketrans('','',string.punctuation))).split())
def answer_score(prediction,gold):
    from collections import Counter
    p=normalize(prediction).split();g=normalize(gold).split();em=int(p==g)
    if set(p+g)&{'yes','no','noanswer'}and p!=g:return {'em':em,'f1':0.}
    overlap=sum((Counter(p)&Counter(g)).values());f1=2*overlap/(len(p)+len(g))if p or g else 1.
    return {'em':em,'f1':f1}
