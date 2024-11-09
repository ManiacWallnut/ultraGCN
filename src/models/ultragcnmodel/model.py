#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class UltraGCN(torch.nn.Module):
    def __init__(self,
                 hyper_param,
                 ii_constraint_mat,
                 ii_neighbor_mat):
        super(UltraGCN, self).__init__()

        # initialize variables
        self.user_num = hyper_param['user_num']
        self.item_num = hyper_param['item_num']
        self.embedding_dim = hyper_param['embedding_dim']
        self.w1 = hyper_param['w1']
        self.w2 = hyper_param['w2']
        self.w3 = hyper_param['w3']
        self.w4 = hyper_param['w4']

        self.negative_weight = hyper_param['negative_weight']
        self.gamma = hyper_param['gamma']
        self.lambda_ = hyper_param['lambda']

        self.user_embeds = torch.nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = torch.nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = hyper_param['constraint_mat']
        
        # 모델 초기화 시 ii_constraint_mat와 ii_neighbor_mat을 GPU로 이동
        device = self.get_device()
        self.ii_constraint_mat = ii_constraint_mat.to(device)
        self.ii_neighbor_mat = ii_neighbor_mat.to(device)

        self.initial_weight = hyper_param['initial_weight']
        self.initial_weights()

    def initial_weights(self):
        torch.nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        torch.nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()

        # constraint_mat의 'beta_uD'와 'beta_iD'를 users와 동일한 디바이스로 이동
        beta_uD = self.constraint_mat['beta_uD'].to(device)
        beta_iD = self.constraint_mat['beta_iD'].to(device)

        if self.w2 > 0:
            pos_weight = torch.mul(beta_uD[users], beta_iD[pos_items])
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(beta_uD[users], neg_items.size(1)),
                                   beta_iD[neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)
        
        weight = torch.cat((pos_weight, neg_weight))
        return weight
    
    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        # batch_size
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) 

        user_embeds = user_embeds.unsqueeze(1)

        # batch_size * negative_num
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) 

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, 
                                                      neg_labels, 
                                                      weight = omega_weight[len(pos_scores):].view(neg_scores.size()),
                                                      reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores,
                                                      pos_labels,
                                                      weight = omega_weight[:len(pos_scores)],
                                                      reduction='none')
        
        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()
    
    def cal_loss_I(self, users, pos_items):
        device = self.get_device()

        pos_items = pos_items.to(device)

        # ii_neighbor_mat과 ii_constraint_mat은 이미 GPU로 이동했으므로 to(device) 호출을 제거
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items])

        sim_scores = self.ii_constraint_mat[pos_items]

        user_embeds = self.user_embeds(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        return loss.sum()
    
    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def get_device(self):
        return self.user_embeds.weight.device

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss
    
    def test_forward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)

        return user_embeds.mm(item_embeds.t())

    # def predict(self, features):
    #     scores = self.linear(features)
    #     return torch.nn.functional.softmax(scores, dim=1)

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import torch

# import torch.nn.functional as F


# class UltraGCN(torch.nn.Module):
#     def __init__(self,
#                  hyper_param,
#                  ii_constraint_mat,
#                  ii_neighbor_mat):
#         super(UltraGCN, self).__init__()

#         # initialize variables
#         self.user_num = hyper_param['user_num']
#         self.item_num = hyper_param['item_num']
#         self.embedding_dim = hyper_param['embedding_dim']
#         self.w1 = hyper_param['w1']
#         self.w2 = hyper_param['w2']
#         self.w3 = hyper_param['w3']
#         self.w4 = hyper_param['w4']

#         self.negative_weight = hyper_param['negative_weight']
#         self.gamma = hyper_param['gamma']
#         self.lambda_ = hyper_param['lambda']

#         self.user_embeds = torch.nn.Embedding(self.user_num, self.embedding_dim)
#         self.item_embeds = torch.nn.Embedding(self.item_num, self.embedding_dim)

#         self.constraint_mat = hyper_param['constraint_mat']
#         self.ii_constraint_mat = ii_constraint_mat.to(self.get_device())
#         self.ii_neighbor_mat = ii_neighbor_mat.to(self.get_device())

#         # self.constraint_mat = self.constraint_mat.to(device)
#         # self.ii_constraint_mat = self.ii_constraint_mat.to(device)
#         # self.ii_neighbor_mat = self.ii_neighbor_mat.to(device)


#         self.initial_weight = hyper_param['initial_weight']
#         self.initial_weights()

#         # # initialize layers
#         # self.linear = torch.nn.Linear(self.in_dim, self.out_dim, bias=True)
#         # torch.nn.init.normal_(self.linear.weight)
#         # self.criterion = torch.nn.CrossEntropyLoss()

#     def initial_weights(self):
#         torch.nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
#         torch.nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

#     def get_omegas(self, users, pos_items, neg_items):
#         device = self.get_device()

#         # constraint_mat의 'beta_uD'와 'beta_iD'를 users와 동일한 디바이스로 이동
#         beta_uD = self.constraint_mat['beta_uD'].to(device)
#         beta_iD = self.constraint_mat['beta_iD'].to(device)

#         if self.w2 > 0:
#             pos_weight = torch.mul(beta_uD[users], beta_iD[pos_items])
#             pos_weight = self.w1 + self.w2 * pos_weight
#         else:
#             pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
#         # users = (users * self.item_num).unsqueeze(0)
#         if self.w4 > 0:
#             neg_weight = torch.mul(torch.repeat_interleave(beta_uD[users], neg_items.size(1)),
#                                    beta_iD[neg_items.flatten()]).to(device)
#             neg_weight = self.w3 + self.w4 * neg_weight
#         else:
#             neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)
        
#         weight = torch.cat((pos_weight, neg_weight))
#         return weight
    
#     def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
#         device = self.get_device()
#         user_embeds = self.user_embeds(users)
#         pos_embeds = self.item_embeds(pos_items)
#         neg_embeds = self.item_embeds(neg_items)

#         # batch_size
#         pos_scores = (user_embeds * pos_embeds).sum(dim=-1) 

#         user_embeds = user_embeds.unsqueeze(1)

#         # batch_size * negative_num
#         neg_scores = (user_embeds * neg_embeds).sum(dim=-1) 

#         neg_labels = torch.zeros(neg_scores.size()).to(device)
#         neg_loss = F.binary_cross_entropy_with_logits(neg_scores, 
#                                                       neg_labels, 
#                                                       weight = omega_weight[len(pos_scores):].view(neg_scores.size()),
#                                                       reduction='none').mean(dim = -1)
        
#         pos_labels = torch.ones(pos_scores.size()).to(device)
#         pos_loss = F.binary_cross_entropy_with_logits(pos_scores,
#                                                       pos_labels,
#                                                       weight = omega_weight[:len(pos_scores)],
#                                                       reduction='none')
        
#         loss = pos_loss + neg_loss * self.negative_weight

#         return loss.sum()
    
#     def cal_loss_I(self, users, pos_items):
#         device = self.get_device()

#         ii_neighbor_mat = self.ii_neighbor_mat.to(device)

#         # len(pos_items) * num_neighbors * dim
#         neighbor_embeds = self.item_embeds(ii_neighbor_mat[pos_items])

#         # len(pos_items) * num_neighbors
#         sim_scores = self.ii_constraint_mat[pos_items].to(device)

#         user_embeds = self.user_embeds(users).unsqueeze(1)

#         loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

#         # loss = loss.sum(-1)
#         return loss.sum()
    
#     def norm_loss(self):
#         loss = 0.0
#         for parameter in self.parameters():
#             loss += torch.sum(parameter ** 2)
#         return loss / 2
    
#     def get_device(self):
#         return self.user_embeds.weight.device

#     def forward(self, users, pos_items, neg_items):
#         omega_weight = self.get_omegas(users, pos_items, neg_items)

#         loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
#         loss += self.gamma * self.norm_loss()
#         loss += self.lambda_ * self.cal_loss_I(users, pos_items)
#         return loss
    
#     def test_forward(self, users):
#         items = torch.arange(self.item_num).to(users.device)
#         user_embeds = self.user_embeds(users)
#         item_embeds = self.item_embeds(items)

#         return user_embeds.mm(item_embeds.t())

#     # def predict(self, features):
#     #     scores = self.linear(features)
#     #     return torch.nn.functional.softmax(scores, dim=1)