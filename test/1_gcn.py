import time

import dgl
import numpy as np
import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import citation_graph as citegrh
import networkx as nx
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = lambda nodes:{'h':torch.sum(nodes.mailbox['m'],1)+nodes.data['h']}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self,dropout=0.5):
        super(Net, self).__init__()
        self.dropout = dropout
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.layer2(g, x)
        return x

def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask,mode='eval'):
    if mode == 'eval':
        model.eval()
    else:
        model.train()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(net):
    g, features, labels, train_mask, test_mask = load_cora_data()
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    dur = []
    for epoch in range(50):
        if epoch >= 3:
            t0 = time.time()
        
        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch >= 3:
            dur.append(time.time() - t0)
            
        train_acc = evaluate(net,g, features, labels, train_mask,mode='train')
        test_acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.2f} % | Test Acc {:.2f} % | Time(s) {:.4f}".format(epoch, loss.item(), train_acc*100, test_acc*100,np.mean(dur)))

if __name__ == "__main__":
    net = Net()
    train(net)