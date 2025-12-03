import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score

# =================================================================
# 1. DIFF POOLING BLOCK (SỬA ĐỔI CHO EVIDENTIAL LEARNING)
# =================================================================
class DiffPoolingBlock(torch.nn.Module):
    def __init__(self, in_dim, n_clusters, tau=2, dim=-1, n_layer=1):
        super().__init__()
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_dim, n_clusters))
        for i in range(n_layer-1):
            self.linear.append(nn.Linear(n_clusters, n_clusters))
        
        # [S²E CHANGE] Thay Softmax bằng Softplus để tạo Evidence không âm
        self.activation_head = nn.Softplus()
        
        self.activation = nn.ReLU()
        self.tau = tau
        self.dim = dim

    def forward(self, h, adj):
        out = h
        for i in range(len(self.linear)):
            out = self.activation(torch.mm(adj, self.linear[i](out)) + 1e-15)
        
        # [S²E CHANGE] Trả về Alpha (Dirichlet parameters) thay vì xác suất
        evidence = self.activation_head(out)
        alpha = evidence + 1
        return alpha


# =================================================================
# 2. STANDARD GCN & READOUTS (GIỮ NGUYÊN TỪ CARE)
# =================================================================
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None: out += self.bias
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self): super(AvgReadout, self).__init__()
    def forward(self, seq): return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self): super(MaxReadout, self).__init__()
    def forward(self, seq): return torch.max(seq, 1).values

class MinReadout(nn.Module):
    def __init__(self): super(MinReadout, self).__init__()
    def forward(self, seq): return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self): super(WSReadout, self).__init__()
    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


# =================================================================
# 3. S²E-CARE MODEL (LOGIC CHÍNH)
# =================================================================
class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, readout, config):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, 2 * n_h, activation)
        self.gcn2 = GCN(2 * n_h, n_h, activation)
        self.act = nn.PReLU()
        self.fc1 = nn.Linear(n_h, 2 * n_h, bias=False)
        
        # [S²E] DiffPool trả về Alpha
        self.diffpool = DiffPoolingBlock(n_in, n_clusters=config['clusters'], dim=-1)
        
        self.ReLU = nn.ReLU()
        if readout == 'max': self.read = MaxReadout()
        elif readout == 'min': self.read = MinReadout()
        elif readout == 'avg': self.read = AvgReadout()
        elif readout == 'weighted_sum': self.read = WSReadout()
            
        self.tau = 1
        self.lamb = config['lamb']
        self.alpha_coeff = config['alpha'] # Đổi tên để tránh trùng với tham số Dirichlet alpha
        self.beta = config['beta']
        self.norm = config['norm']
        self.n_clusters = config['clusters']

    # --- HELPER: EVIDENTIAL LOSS & UNCERTAINTY ---
    def calc_uncertainty(self, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        return self.n_clusters / S

    def evidential_loss(self, alpha, pseudo_labels):
        """Hàm Loss đặc trưng của S²E-CARE"""
        S = torch.sum(alpha, dim=1, keepdim=True)
        y = F.one_hot(pseudo_labels, num_classes=self.n_clusters).float()
        
        # Bayes Risk (MSE)
        prob = alpha / S
        loss_risk = torch.sum((y - prob) ** 2, dim=1)
        
        # KL Divergence Regularization
        alpha_tilde = y + (1 - y) * alpha
        var_term = torch.lgamma(torch.sum(alpha_tilde, dim=1)) - \
                   torch.lgamma(torch.sum(torch.ones_like(alpha), dim=1)) - \
                   torch.sum(torch.lgamma(alpha_tilde), dim=1) + \
                   torch.sum(torch.lgamma(torch.ones_like(alpha)), dim=1)
                   
        return torch.mean(loss_risk + 0.01 * var_term)

    def normalize_adj_tensor(self, adj):
        row_sum = torch.sum(adj, 0)
        r_inv = torch.pow(row_sum, -0.5).flatten()
        r_inv = torch.nan_to_num(r_inv, posinf=0, neginf=0)
        adj = torch.mm(adj, torch.diag_embed(r_inv))
        adj = torch.mm(torch.diag_embed(r_inv), adj)
        return adj

    # --- FORWARD PASS ---
    def forward(self, seq, adj, raw_adj, adj2=None, raw_adj2=None, sparse=False):
        # 1. GCN Encoder
        if adj2 is not None:
            feat1 = self.gcn2(self.gcn1(seq, adj[0]), adj[0])
            feat2 = self.gcn2(self.gcn1(seq, adj2[0]), adj2[0])
            feat = (feat1 + feat2)/2
            s_adj = (adj[0] + adj2[0])/2
        else:
            feat = self.gcn2(self.gcn1(seq, adj[0]), adj[0])
            s_adj = adj[0]
            raw_adj2 = raw_adj

        # 2. Evidential Clustering (DiffPool)
        # Output là Alpha (Dirichlet parameters)
        alpha = self.diffpool(seq, s_adj)
        
        # Chuyển Alpha về Probability để dùng cho các loss cũ của CARE
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob_assign = alpha / S # Expected Probability
        
        # 3. Tính toán các Loss thành phần
        raw_adj = (raw_adj[0] + raw_adj2[0])/2
        
        # a) Graph Contrastive Loss (Dùng Prob) - Đã có Nystrom Sampling bên trong
        con_loss = self.graph_contrastive_loss(feat, feat, prob_assign, raw_adj)
        
        # b) Cluster Affinity Loss
        cluster_sim = self.alpha_coeff * torch.mm(prob_assign, prob_assign.T) + raw_adj * (1 - self.alpha_coeff)
        if self.norm:
            cluster_sim = self.normalize_adj_tensor(cluster_sim)
        
        # c) View Consistency
        if adj2 is None: view_consistency = 0
        else: view_consistency = torch.norm(feat1 - feat2, dim=1, p=2).mean()
        
        # d) Max Message Loss (CARE gốc)
        loss_msg, _ = self.max_message(feat, cluster_sim)
        
        # e) [S²E NEW] Evidential Loss
        # Tự giám sát bằng nhãn giả tự tin nhất
        with torch.no_grad():
            pseudo_labels = torch.argmax(alpha, dim=1)
        loss_edl = self.evidential_loss(alpha, pseudo_labels)

        # f) Regularization
        fc1 = self.fc1(feat)
        loss_reg = 0
        if self.beta != 0:
            loss_reg = self.beta * self.reg_edge(fc1, adj[0])
            
        # Tổng hợp Loss
        total_loss = loss_msg + \
                     self.lamb * (con_loss + view_consistency) + \
                     loss_reg + \
                     0.1 * loss_edl # Thêm trọng số cho EDL
                     
        # Quan trọng: Trả về 4 giá trị để khớp với main.py mới
        return feat, cluster_sim, total_loss, alpha

    # --- NYSTRÖM APPROXIMATION LOSS ---
    def graph_contrastive_loss(self, v1, v2, assignment, adj, test_phase=False):
        # Lấy mẫu ngẫu nhiên (Sampling) để giảm độ phức tạp từ O(N^2) -> O(Nm)
        if v1.shape[0] > 10000 and not test_phase:
            idx = torch.randperm(v1.shape[0])[:5000] # Landmarks
        else:
            idx = torch.arange(v1.shape[0])
            
        v1, v2, assignment = v1[idx], v2[idx], assignment[idx]
        adj = adj[idx, :][:, idx]
        
        v = (v1 + v2)/2
        sim = self.cosine_sim(v, v, self.tau)
        O = self.alpha_coeff * torch.matmul(assignment, assignment.T) + (1 - self.alpha_coeff) * adj
        
        # Tránh lỗi chia cho 0
        O_sum = O.sum(dim=1, keepdim=True).clamp(min=1e-6)
        loss = torch.norm(O / O_sum - sim, 2, dim=1).mean()
        return loss

    def cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)

    def reg_edge(self, emb, adj):
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        sim_u_u = torch.mm(emb, emb.T)
        adj_inverse = (1 - adj)
        sim_u_u = sim_u_u * adj_inverse
        sim_u_u_no_diag = torch.sum(sim_u_u, 1)
        row_sum = torch.sum(adj_inverse, 1)
        r_inv = torch.pow(row_sum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        sim_u_u_no_diag = sim_u_u_no_diag * r_inv
        loss_reg = torch.sum(sim_u_u_no_diag)
        return loss_reg

    def max_message(self, feature, adj_matrix):
        feature = feature / torch.norm(feature, dim=-1, keepdim=True)
        sim_matrix = torch.mm(feature, feature.T)
        sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
        sim_matrix[torch.isinf(sim_matrix)] = 0
        sim_matrix[torch.isnan(sim_matrix)] = 0
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        return - torch.sum(message), message

    # --- S²E-CARE INFERENCE ---
    def inference(self, feature, alpha):
        # 1. Base Score từ CARE (Affinity-based)
        # Sử dụng lại logic của max_message để tính độ liên kết
        feature_norm = feature / torch.norm(feature, dim=-1, keepdim=True)
        
        # Trong CARE, message cao = normal, message thấp = anomaly
        # Ta lấy âm của message để: cao = anomaly
        _, message = self.max_message(feature, torch.mm(feature_norm, feature_norm.T)) 
        base_score = 1 - message # Giả định message nằm trong khoảng [0,1]
        
        # 2. Uncertainty Score
        uncertainty = self.calc_uncertainty(alpha).squeeze()
        
        # 3. Kết hợp: Base Score * (1 + Uncertainty)
        # Node vừa có affinity thấp, vừa có uncertainty cao sẽ có score cao nhất
        final_score = base_score * (1 + uncertainty)
        
        return final_score

    def evaluation(self, score, ano_label):
        auc = roc_auc_score(ano_label, score)
        AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('   >>> [EVAL] AUC:{:.4f}, AP:{:.4f}'.format(auc, AP))