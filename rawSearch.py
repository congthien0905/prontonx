import json
from collections import defaultdict
import string
import math
class TFIDF:
    def __init__(self):
        with open('docs.json', 'r') as docs_file:
            self.docs = json.load(docs_file)
      
    def search(self, q, k):
        # def get_tf(num):
        #     return rounding(math.log10(num+1))

        # def rounding(num):
        #     return math.floor(num * 1000) / 1000
        
        # Initialize stats dictionary
        stats = {
            "words": {},
            "docs": {}
        }
        
        # Process documents and collect word occurrences
        for i, doc in enumerate(self.docs):
            if i not in stats['docs']:
                stats['docs'][i] = defaultdict(int)

            for word in doc.split(' '):
                if word not in stats['words']:
                    stats['words'][word] = {i}
                else:
                    stats['words'][word].add(i)

                stats['docs'][i][word] += 1

                
        words = stats['words'].keys()

        # Calculate IDF
        idf = defaultdict(float)
        N = len(self.docs)
        for word in words:
            df = len(stats['words'][word])
            idf[word] = math.log10(N / df)

        tf_idf_list = defaultdict(lambda: defaultdict(float))
        ds = defaultdict(float)

        # Calculate TF-IDF and document norms
        for doc in stats['docs']:
            d = 0
            for word in words:
                #tf = get_tf(stats['docs'][doc][word])
                #return rounding(math.log10(num+1))
                tf=math.floor(math.log10((stats['docs'][doc][word])+1)*1000)/1000
                #tf = round(math.log10((stats['docs'][doc][word])+1))
                tf_idf = tf * idf[word]
                d += tf_idf ** 2
                tf_idf_list[word][doc] = tf_idf

            # Phép tính dưới mẫu của hình bên dưới
            d_ = d ** (1/2)

            # Lưu các giá trị
            #math.floor(num * 1000) / 1000
            ds[doc] = math.floor(d_*1000)/1000
            
        # Score documents based on query
        
        
        results = []
        for i in range(len(self.docs)):
            score = 0
            for t in q.split():
                t = t.lower()
                try:
                    score += tf_idf_list[t][i] / ds[i]
                except ZeroDivisionError:
                    score += 0
            results.append((score, i))

        # Sort and return top k results
        results.sort(reverse=True)
        top_k_docs = []
        for score, i in results[:k]:  # Get only the top k results
            top_k_docs.append(self.docs[i])  # Store the actual documents
        return top_k_docs
    

    

