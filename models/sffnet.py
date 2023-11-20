import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP, found_affinity_unseen_paris, Simple_gcn
from .word_embedding import load_word_embeddings
from .graph_method import GraphFull
from collections import OrderedDict
from os.path import join as ospj
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes=512, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 2)),
            ("relu", nn.ReLU()),
            ("drop", nn.Dropout(0.5)),
            ("c_proj", nn.Linear(d_model * 2, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_x = nn.LayerNorm(d_model)
        self.ln_y = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 2)),
            ("relu", nn.ReLU()),
            ("c_proj", nn.Linear(d_model * 2, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_x(x), self.ln_y(y))
        x = x + self.mlp(self.ln_2(x))
        return x


class Feats_fusion(nn.Module):
    def __init__(self, width_res: int, width_gfn: int, smp_weight: float, fuse_weight: float,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width_res = width_res
        self.width_gfn = width_gfn
        self.smp_weight = smp_weight
        self.fuse_weight = fuse_weight
        self.res2gfn_layer = nn.Linear(width_res, width_gfn)
        self.gfn2res_layer = nn.Linear(width_gfn, width_res)
        self.conv_last = nn.Sequential(
            nn.Conv2d(2 * width_res, 2 * width_res, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.3)
        self.cross_block_res = CrossResidualAttentionBlock(width_res, width_res // 64, attn_mask)
        self.cross_block_gfn = CrossResidualAttentionBlock(width_gfn, width_gfn // 64, attn_mask)
        self.residual_blocks_res = nn.Sequential(
            *[ResidualAttentionBlock(width_res, width_res // 64, attn_mask)])
        self.residual_blocks_gfn = nn.Sequential(
            *[ResidualAttentionBlock(width_gfn, width_gfn // 64, attn_mask)])
        self.cbam_res = CBAM(in_planes=width_res)
        self.cbam_gfn = CBAM(in_planes=width_gfn)

    def res2gfn(self, x: torch.Tensor):
        x = self.res2gfn_layer(x)
        x = self.dropout(x)
        return x

    def gfn2res(self, x: torch.Tensor):
        x = self.gfn2res_layer(x)
        x = self.dropout(x)
        return x

    def forward(self, x_res: torch.Tensor, x_gfn: torch.Tensor):
        b, c, d, _ = x_res.shape
        x_smp = x_res.view(b, c, -1).mean(2)
        g_smp = x_gfn.view(b, c, -1).mean(2)
        smp_fuse = torch.cat([x_smp, g_smp], dim=1)
        # b*c*d*d -> b*c*L ->L*b*c
        x_res = x_res.reshape(b, c, -1).permute(2, 0, 1)
        x_gfn = x_gfn.reshape(b, c, -1).permute(2, 0, 1)
        res_cross = self.cross_block_res(x_res, self.gfn2res(x_gfn))
        res_ft = self.residual_blocks_res(res_cross)
        gfn_cross = self.cross_block_gfn(x_gfn, self.res2gfn(x_res))
        gfn_ft = self.residual_blocks_gfn(gfn_cross)

        res_ft = res_ft.permute(1, 2, 0).reshape(b, c, d, d)
        res_ft = self.cbam_res(res_ft)
        gfn_ft = gfn_ft.permute(1, 2, 0).reshape(b, c, d, d)
        gfn_ft = self.cbam_gfn(gfn_ft)
        ft_cat = torch.cat([res_ft, gfn_ft], dim=1).reshape(b, 2 * c, -1).mean(2)
        feats_fusion = self.smp_weight * smp_fuse + self.fuse_weight * ft_cat
        return feats_fusion


def get_sub_classes():
    return None


def get_all_ids(relevant_pairs, attr2idx, obj2idx):
    # Precompute validation pairs
    attrs, objs = zip(*relevant_pairs)
    attrs = [attr2idx[attr] for attr in attrs]
    objs = [obj2idx[obj] for obj in objs]
    pairs = [a for a in range(len(relevant_pairs))]
    pairs = torch.LongTensor(pairs).to(device)
    attrs = torch.LongTensor(attrs).to(device)
    objs = torch.LongTensor(objs).to(device)

    return attrs, objs, pairs


def get_word_dim(args):
    if args.emb_init == 'glove' or args.emb_init == 'word2vec' or args.emb_init == 'fasttext':
        word_dim = 300
    elif args.emb_init == 'ft+w2v+gl':
        word_dim = 900
    else:
        word_dim = 600
    return word_dim


class STF(nn.Module):
    def __init__(self, dset, args):
        super(STF, self).__init__()
        self.args = args
        self.dset = dset
        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        self.cos_scale = self.args.cosine_scale
        self.pairs = dset.pairs
        self.num_attrs, self.num_objs, self.num_pairs = len(self.dset.attrs), len(self.dset.objs), len(dset.pairs)
        self.num_attr_range = torch.arange(self.num_attrs).to(device)
        self.num_obj_range = torch.arange(self.num_objs).to(device)
        self.word_dim = get_word_dim(self.args)

        self.unseen_pairs = []
        unseen_pairs_idx = []
        for idx, i in enumerate(self.dset.pairs):
            if i not in self.dset.train_pairs:
                self.unseen_pairs.append(i)
                unseen_pairs_idx.append(idx)
        self.unseen_pairs_idx = torch.tensor(unseen_pairs_idx)
        self.unseen_attrs, self.unseen_objs, _ = get_all_ids(self.unseen_pairs, dset.attr2idx, dset.obj2idx)
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs, dset.attr2idx, dset.obj2idx)

        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs, dset.attr2idx,
                                                                              dset.obj2idx)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        self.fusion = Feats_fusion(dset.feat_dim, dset.feat_dim, attn_mask=None,
                                   smp_weight=args.smp_weight, fuse_weight=args.fuse_weight)
        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        if args.nlayers:
            self.image_embedder = MLP(2 * dset.feat_dim, args.emb_dim, num_layers=args.nlayers,
                                      dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)
        self.resnet_attr_mlp = MLP(inp_dim=2 * dset.feat_dim, out_dim=args.emb_dim, num_layers=2,
                                   dropout=True,
                                   norm=self.args.norm, layers=[2048], relu=True)
        self.resnet_obj_mlp = MLP(inp_dim=2 * dset.feat_dim, out_dim=args.emb_dim, num_layers=2,
                                  dropout=True,
                                  norm=self.args.norm, layers=[2048], relu=True)

        self.attr2sharing_space = nn.Sequential(
            nn.Linear(self.word_dim, args.emb_dim),
            nn.ReLU()
        )
        self.obj2sharing_space = nn.Sequential(
            nn.Linear(self.word_dim, args.emb_dim),
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(self.word_dim * 2, args.emb_dim),
            nn.ReLU()
        )

        # init with word embeddings
        self.attr_embedder = nn.Embedding(len(dset.attrs), self.word_dim).to(device)
        self.obj_embedder = nn.Embedding(len(dset.objs), self.word_dim).to(device)
        if args.load_save_embeddings:
            attr_weights = ospj("./utils/", args.dataset + "_" + args.emb_init + '_attr-weights.t7')
            obj_weights = ospj("./utils/", args.dataset + "_" + args.emb_init + '_obj-weights.t7')
            if not os.path.exists(attr_weights or obj_weights):
                print("Generating embeddings...")
                attr_w = load_word_embeddings(args.emb_init, dset.attrs)
                obj_w = load_word_embeddings(args.emb_init, dset.objs)
                torch.save(attr_w, attr_weights)
                torch.save(obj_w, obj_weights)
            else:
                attr_w = torch.load(attr_weights)
                obj_w = torch.load(obj_weights)
            self.attr_embedder.weight.data.copy_(attr_w)
            self.obj_embedder.weight.data.copy_(obj_w)

        # improved cge:
        if args.use_cge:
            graph_method = GraphFull(dset, args)
            self.gcn = graph_method.gcn
            self.embeddings = graph_method.embeddings
            self.train_idx = graph_method.train_idx

        obj2pair_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            obj2pair_mask.append(torch.BoolTensor(mask))
        self.obj2pair_mask = torch.stack(obj2pair_mask, 0)
        attr2pair_mask = []
        for _attr in dset.attrs:
            mask = [1 if _attr == attr else 0 for attr, obj in dset.pairs]
            attr2pair_mask.append(torch.BoolTensor(mask))
        self.attr2pair_mask = torch.stack(attr2pair_mask, 0)

    def compose(self, attrs, objs, use_cge=False):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output = F.normalize(output, dim=1)
        return output

    def __Label_smooth(self, inputs, targets, smoothing=0.1):
        assert 0 <= smoothing < 1
        label_all = []
        for idx in targets:
            _attr, _obj = self.dset.train_pairs[idx]  # 找到真实标签包含的attr和obj
            label = torch.LongTensor([1 if (_attr == attr or _obj == obj) else 0 for attr, obj in
                                      self.dset.train_pairs])  # 找到semi-positive
            label = (smoothing / (label.sum() - 1)) * label
            label[idx] = 1 - smoothing
            label_all.append(label)
        label_all = torch.stack(label_all, 0).to(device)
        log_logits = F.log_softmax(inputs, dim=1)
        temp = -1 * (log_logits * label_all)
        loss = (temp.sum(1).sum(0)) / temp.shape[0]
        return loss

    def __unseen_calibration(self, unseen_pred, true_targets, unseen_pairs, unseen_map, label_smooth=0.01):
        assert 0 <= label_smooth < 1
        mask = torch.full((len(true_targets),), True, dtype=torch.bool)
        label_all = []
        for i, idx in enumerate(true_targets):
            _attr, _obj = self.dset.train_pairs[idx]  # 找到真实标签包含的attr和obj
            label = torch.LongTensor([1 if (_attr == attr or _obj == obj) else 0 for attr, obj in
                                      unseen_pairs])  # 找到unseen pairs包含attr和obj的
            if label.sum() == 0:
                mask[i] = False
                continue
            else:
                label = (label_smooth / (label.sum())) * label
            label[unseen_map[idx]] = 1.0 - label_smooth
            label_all.append(label)
        label_all = torch.stack(label_all, 0).to(device)
        log_logits = F.log_softmax(unseen_pred[mask], dim=1)
        temp = -1.0 * (log_logits * label_all)
        loss = (temp.sum(1).sum(0)) / temp.shape[0]  # avg_loss
        return loss

    def train_forward_normal(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        b, c, _, _ = img.shape
        if self.args.use_gfn:
            g_img = x[4]
            img = self.fusion(img, g_img)
            # img = torch.cat([img.reshape(b, c, -1).mean(2).squeeze(), g_img.reshape(b, c, -1).mean(2).squeeze()], dim=1)
        # --pair
        if self.args.nlayers:
            img_feats = F.normalize(self.image_embedder(img), dim=1)   #self.image_embedder(img)
        else:
            img_feats = (img)
        if self.args.use_cge:
            cge_embeddings = self.gcn(self.embeddings)
            if self.args.train_only:
                pair_embed = cge_embeddings[self.train_idx]
            else:
                pair_embed = cge_embeddings[self.num_attrs + self.num_objs:self.num_attrs + self.num_objs +                                                                self.num_pairs, :]
        else:
            pair_embed = self.compose(self.train_attrs, self.train_objs, use_cge=False)  # normalize in compose
        pair_pred = torch.matmul(img_feats, pair_embed.T)
        #p_loss = F.cross_entropy(pair_pred, pairs)
        p_loss = self.__Label_smooth(self.cos_scale * pair_pred, pairs, smoothing=0.1)
        loss_all = p_loss
        # attr
        fc_img_attr = self.resnet_attr_mlp(img)
        fc_embedding_attr = self.attr2sharing_space(self.attr_embedder(self.num_attr_range))
        attr_pred = torch.matmul(F.normalize(fc_img_attr, dim=1), F.normalize(fc_embedding_attr, dim=1).T)
        loss_attr = F.cross_entropy(self.cos_scale * attr_pred, attrs)
        loss_all += self.args.attr_loss_w * loss_attr

        # --obj--
        fc_img_obj = self.resnet_obj_mlp(img)
        fc_embedding_obj = self.obj2sharing_space(self.obj_embedder(self.num_obj_range))
        obj_pred = torch.matmul(F.normalize(fc_img_obj, dim=1), F.normalize(fc_embedding_obj, dim=1).T)
        loss_obj = F.cross_entropy(self.cos_scale * obj_pred, objs)
        loss_all += self.args.obj_loss_w * loss_obj

        # --use_calibration
        if self.args.use_calibration:
            with torch.no_grad():
                if self.args.use_cge:
                    unseen_pair_embed = cge_embeddings[self.unseen_pairs_idx + self.num_attrs + self.num_objs]
                else:
                    unseen_pair_embed = self.compose(self.unseen_attrs, self.unseen_objs)

            unseen_pair_pred = torch.matmul(img_feats, unseen_pair_embed.T)
            unseen_index_map = found_affinity_unseen_paris(pair_embed.detach(), unseen_pair_embed.detach())
            loss_unseen = self.__unseen_calibration(unseen_pair_pred, pairs, self.unseen_pairs,
                                                    unseen_index_map, label_smooth=self.args.unseen_smooth)
            loss_all += self.args.calibration_weights * loss_unseen

        return loss_all, None, None, None

    def val_forward_dotpr(self, x):
        img = x[0]
        b, c, _, _ = img.shape
        if self.args.use_gfn:
            g_img = x[4]
            img = self.fusion(img, g_img)
        # --pair--
        if self.args.nlayers:
            img_feats = F.normalize(self.image_embedder(img), dim=1)
        else:
            img_feats = (img)
        if self.args.use_cge:
            pair_embeds = self.gcn(self.embeddings)[
                          self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :]
        else:
            pair_embeds = self.compose(self.val_attrs, self.val_objs)
        score = torch.matmul(img_feats, pair_embeds.T)

        fc_img_attr = F.normalize(self.resnet_attr_mlp(img), dim=1)
        fc_embedding_attr = F.normalize(self.attr2sharing_space(self.attr_embedder(self.num_attr_range)), dim=1)
        attr_score = torch.matmul(fc_img_attr, fc_embedding_attr.T)

        # --obj--
        fc_img_obj = F.normalize(self.resnet_obj_mlp(img), dim=1)
        fc_embedding_obj = F.normalize(self.obj2sharing_space(self.obj_embedder(self.num_obj_range)), dim=1)
        obj_score = torch.matmul(fc_img_obj, fc_embedding_obj.T)

        # Add scores to pair_score
        attr_score2pair = torch.matmul(attr_score, self.attr2pair_mask.float().to(device)).to(device)
        obj_score2pair = torch.matmul(obj_score, self.obj2pair_mask.float().to(device)).to(device)
        score += self.args.attr_score_weight * attr_score2pair + self.args.obj_score_weight * obj_score2pair

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]  # 取出对应每个组合，batch_size个样本的得分

        return None, None, None, scores, attr_score, obj_score

    def forward(self, x):
        if self.training:
            loss, pred, pred_attr, pred_obj = self.train_forward(x)
            return loss, pred, pred_attr, pred_obj
        else:
            with torch.no_grad():  # testing
                loss, loss_attr, loss_obj, pred, pred_attr, pred_obj = self.val_forward(x)
                return loss, loss_attr, loss_obj, pred, pred_attr, pred_obj