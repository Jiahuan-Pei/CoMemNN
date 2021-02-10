import faiss # faiss use cpu version since cudatookit issue
import torch
import os
from whoosh.index import create_in
from whoosh.fields import *
import whoosh.index as index
from whoosh.qparser import QueryParser
from whoosh.qparser import syntax

class Vector_KNN(object):
    def __init__(self, hidden_size, nlist=100, nprobe=100):
        '''
        :param hidden_size:
        :param nlist: number of cluster center
        :param nprobe:
        '''
        self.nprobe=nprobe
        # if torch.cuda.is_available():
        #     ngpus = faiss.get_num_gpus()
        #     print('num GPUs: ', ngpus)
        quantizer = faiss.IndexFlatIP(hidden_size)
        self.cpu_index = faiss.IndexIVFFlat(quantizer, hidden_size, nlist, faiss.METRIC_INNER_PRODUCT)
        self.act_index=self.cpu_index

    def index(self, vector_to_index):
        if torch.cuda.is_available():
            vector_to_index = vector_to_index.cpu()
        vector_to_index=vector_to_index.numpy()
        # if torch.cuda.is_available():
        #     self.act_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        self.act_index.train(vector_to_index)
        self.act_index.add(vector_to_index)
        print('#entities: ', self.act_index.ntotal)

    # def index_size(self):
    #     return self.act_index.ntotal

    def search(self, vector, k):
        self.act_index.nprobe = self.nprobe
        if torch.cuda.is_available():
            vector = vector.cpu()
        # (n, x, k, distances, labels)
        D, I = self.act_index.search(vector.numpy(), k)
        I = torch.tensor(I)

        return I


class Text_KNN(object):
    def __init__(self,):
        self.ix = None
        self.searcher=None

    def index(self, text_to_index):
        if not os.path.exists("index"):
            os.mkdir("index")
        if index.exists_in("index"):
            self.ix = index.open_dir("index")
            self.searcher=self.ix.searcher()
            return self
        schema = Schema(content=TEXT(stored=True), id=ID(stored=True))
        self.ix = create_in("index", schema)
        writer = self.ix.writer()
        for i in range(len(text_to_index)):
            writer.add_document(content=text_to_index[i], id=i)
        writer.commit()
        self.searcher = self.ix.searcher()

    def search(self, text, k):
        query = QueryParser('content', self.ix.schema, group=syntax.OrGroup).parse(text)
        results = self.searcher.search(query, limit=k, scored=True)
        labels=set([r['id'] for r in results])
        return list(labels)