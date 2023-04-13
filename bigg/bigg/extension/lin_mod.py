### For Linear Regression Prediction
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt 


class EdgeWeightLinearModel:
    
    def __init__(self, args):
        super().__init__(args)
        self.lin_mod = make_pipeline(StandardScaler(),lm.SGDRegressor(max_iter=1000, tol=1e-3, warm_start = True))
    
    def train(features, weights):
        self.lin_mod.fit(features, weights)
    
    def predict(features):
        weights = self.lin_mod.predict(features)
        return features










class BiggWithEdgeLen(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 2])
        self.edgelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 2]) ## Changed
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        self.edge_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim) ## ADDED




d1 = {'col1': np.floor(np.random.uniform(20,30,50)), 'col2': np.floor(np.random.uniform(15,25,50)), 'col3': np.floor(np.random.uniform(10,30,50))}
df1 = pd.DataFrame(data = d1)
y1 = 1.5 * df.col1 - 0.25 * df.col2  - 0.75 * df.col3 + np.random.normal(0, 1, 50)


d2 = {'col1': np.floor(np.random.uniform(20,30,50)), 'col2': np.floor(np.random.uniform(15,25,50)), 'col3': np.floor(np.random.uniform(10,30,50))}
df2 = pd.DataFrame(data = d2)
y2 = 1.5 * df.col1 - 0.25 * df.col2  - 0.75 * df.col3 + np.random.normal(0, 1, 50)

d3 = {'col1': np.floor(np.random.uniform(20,30,50)), 'col2': np.floor(np.random.uniform(15,25,50)), 'col3': np.floor(np.random.uniform(10,30,50))}
df3 = pd.DataFrame(data = d2)
y3 = 1.5 * df.col1 - 0.25 * df.col2  - 0.75 * df.col3 + np.random.normal(0, 1, 50)


reg1 = lm.SGDRegressor(max_iter = 1000, tol=1e-3)
reg2 = lm.SGDRegressor(max_iter = 1000, tol=1e-3)


scaled_df1 = scale(df1)
scaled_df2 = scale(df2)
scaled_df3 = scale(df3)

reg1.partial_fit(scaled_df1, y1)
reg1.partial_fit(scaled_df2, y2)
#reg2.partial_fit(scaled_df1, y1)
reg2.partial_fit(scaled_df2, y2)

reg1.predict(scaled_df3) - reg2.predict(scaled_df3)













###
reg2 = make_pipeline(StandardScaler(),lm.SGDRegressor(max_iter=1000, tol=1e-3, warm_start = True))
reg2.fit(df1, y1)
reg2.fit(df2, y2)
reg2.predict(df3) - y3


reg2.fit(df1, y1)
reg2.predict(df3) - y3






def lin_reg_weights(edges): 
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    
    






## LOSS: (w - w-hat)^2
## Idea: input graphs one at a time. Get path matrix from graph. 



if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    has_node_feats = False

    #with open(os.path.join(cmd_args.data_dir, 'Group202A.dat'), 'rb') as f:
    #    train_graphs = cp.load(f)
    #train_graphs = nx.read_gpickle('/content/drive/MyDrive/Projects/Data/Bigg-Data/Yeast.dat')    
    #train_graphs = nx.readwrite.read_gpickle()
    
    
    path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    print(path)
    with open(path, 'rb') as f:
        train_graphs = cp.load(f)
    
    [TreeLib.InsertGraph(g) for g in train_graphs]

    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    if max_num_nodes < 100:
        print(train_graphs[0].edges(data=True))
    
    #list_node_feats = [torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs]    
    list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    

    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        model.load_state_dict(torch.load(cmd_args.model_dump))
    
    #########################################################################################################
    if cmd_args.phase != 'train':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        with open(path, 'rb') as f:
            gt_graphs = cp.load(f)
        print('# gt graphs', len(gt_graphs))
        gen_graphs = []
        with torch.no_grad():
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, pred_edges, _, pred_node_feats, pred_edge_feats = model(num_nodes)
                #print("edges: ", pred_edges)
                #print("edge feats:",  pred_edge_feats)
                if has_node_feats:
                    pred_g = nx.Graph()
                    pred_g.add_nodes_from(range(num_nodes))
                    #print(pred_node_feats)
                    sys.exit()
                
                if cmd_args.has_edge_feats:
                    weighted_edges = []
                    for e, w in zip(pred_edges, pred_edge_feats):
                        #print("e: ", e)
                        assert e[0] > e[1]
                        w = w.item()
                        w = np.round(w, 4)
                        edge = (e[1], e[0], w)
                        #print("edge:", edge)
                        weighted_edges.append(edge)
                    #print("weighted edges: ", weighted_edges)
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                else:
                    pred_g = nx.Graph()
                    pred_g.add_edges_from(pred_edges)
                    gen_graphs.append(pred_g)
         
        counter = 0
        for g in gen_graphs:
            if counter <= 10:
                print("edges:", g.edges(data=True))
                counter += 1
        
        if cmd_args.has_edge_feats:
            print("Generating Statistics for ", cmd_args.file_name)
            final_graphs = graph_stat_gen(gen_graphs, train_graphs, gt_graphs, kind = cmd_args.file_name)
            print("final_g len: ", len(final_graphs))
        
        else:
            print("Testing for Tree Structures...")
            trees = 0
            for T in gen_graphs:
                if nx.is_tree(T):
                    leaves = [n for n in T.nodes() if T.degree(n) == 1]
                    internal = [n for n in T.nodes() if T.degree(n) == 3]
                    root = [n for n in T.nodes() if T.degree(n) == 2]
                    if 2*len(leaves) - 1 == len(T) and len(leaves) == len(internal) + 2 and len(root) == 1 and len(leaves) + len(internal)+ len(root) == len(T):
                        trees += 1
            print("Number of Trees: ", trees)
            print("Out of....: ", len(gen_graphs))
            final_graphs = gen_graphs
        
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(final_graphs, f, cp.HIGHEST_PROTOCOL)
        print('graph generation complete')
        
        sys.exit()
    #########################################################################################################
    
    # debug_model(model, train_graphs[0], list_node_feats[0], list_edge_feats[0])
    serialized = False

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) ##added
    indices = list(range(len(train_graphs)))
    
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.epoch_save))

        optimizer.zero_grad()
        for idx in pbar:
            random.shuffle(indices)
            batch_indices = indices[:cmd_args.batch_size]
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])

            #node_feats = torch.cat([list_node_feats[i] for i in batch_indices], dim=0)
            #node_feats = node_feats[1:]
            
            #print(node_feats)
            #sys.exit()
            edge_feats = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
            
            
            if serialized:
                for ind in batch_indices:
                    g = train_graphs[ind]
                    n = len(g)
                    m = len(g.edges())
                    
                    ### Obtaining edge list 
                    edgelist = []
                    for e in g.edges():
                        if e[0] < e[1]:
                            e = (e[1], e[0])
                        edgelist.append((e[0], e[1]))
                    edgelist.sort(key = lambda x: x[0])
                    
                    ### Obtaining weights list
                    weightdict = dict()
                    for node1, node2, data in g.edges(data=True):
                        if node1 < node2:
                            e = (node2, node1)
                        else:
                            e = (node1, node2)
                        weightdict[e] = data['weight']
                    
                    ### Compute log likelihood, loss
                    ll, _, _, _ = model(node_end = n, edge_list = edgelist, weights = weightdict)
            else:
                ll, _ = model.forward_train(batch_indices, node_feats=None, edge_feats = edge_feats)
            loss = -ll / num_nodes
            loss.backward()
            loss = loss.item()

            if (idx + 1) % cmd_args.accum_grad == 0:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))
        #scheduler.step()
        print('epoch complete')
        cur = epoch+1
        if cur % 10 == 0 or cur == cmd_args.num_epochs: #save every 10th / last epoch
            print('saving epoch')
            torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
        #_, pred_edges, _, pred_node_feats, pred_edge_feats = model(len(train_graphs[0]))
        #print(pred_edges)
        #print(pred_node_feats)
        #print(pred_edge_feats)
    print("Model training complete.")
    